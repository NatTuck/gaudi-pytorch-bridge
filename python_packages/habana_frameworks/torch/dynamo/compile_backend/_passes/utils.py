###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from habana_frameworks.torch.dynamo.compile_backend.partitioner import HabanaPartitioner
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
import torch.fx

logger = get_compile_backend_logger()


class OptimizationPassPlacement(Enum):
    PRE_PLACEMENT = 1
    PRE_PARTITIONER = 2
    PARTITIONER = 3
    POST_PARTITIONER = 4


@dataclass
class OptimizerContext:
    graph_module: torch.fx.GraphModule
    graph_name: str
    example_inputs: list[torch.Tensor]
    is_training: bool
    is_backward: bool
    is_dynamic: bool
    stage: OptimizationPassPlacement
    current_partitions: list
    current_partitions_non_mergeable: list
    habana_partitioner: HabanaPartitioner = None
    is_submod: bool = False
    use_jit_fork: bool = False


class ColorGraph:
    def __init__(self):
        self.partition_colors = set()
        self.output_colors = set()
        self.colors_to_remove = set()
        self._last_color = 0
        self._graph = defaultdict(set)

    def assign_new_color(self, is_partition_color=False):
        self._last_color += 1
        if is_partition_color:
            self.partition_colors.add(self._last_color)
        else:
            self.colors_to_remove.add(self._last_color)
        return self._last_color

    def add_node(self, user_color, color):
        if user_color != color:
            self._graph[color].add(user_color)

    def add_output_node(self, color):
        # We have cases in which a node without any users belongs to a
        # partition, and that partition can have other users.
        if color not in self._graph:
            self._graph[color] = set()

    def _merge_non_partition_colors(self):
        for color in self.colors_to_remove:
            replacement_set = self._graph.pop(color, {})
            for v in self._graph.values():
                if color in v:
                    v.remove(color)
                    v.update(replacement_set)

        self.colors_to_remove = set()

    def _get_reverse_graph_dict(self):
        logger.debug("Color graph (initial): \n%s", self)
        self._merge_non_partition_colors()
        logger.debug("Color graph (partitions only): \n%s", self)

        reverse_graph_dict = defaultdict(set)
        for color, user_set in self._graph.items():
            user_frozen_set = frozenset(user_set)
            reverse_graph_dict[user_frozen_set].add(color)

        return reverse_graph_dict

    def get_parallel_blocks(self):
        reverse_graph_dict = self._get_reverse_graph_dict()

        parallel_blocks = []

        key_to_remove = frozenset()

        while reverse_graph_dict:
            if key_to_remove not in reverse_graph_dict:
                raise AssertionError("Empty set not a key in reverse_graph_dict. Graph most likely cyclic.")
            current_block = reverse_graph_dict.pop(key_to_remove)
            parallel_blocks.append(current_block)
            next_iter_dependencies = [(key - current_block, val) for key, val in reverse_graph_dict.items()]
            reverse_graph_dict = defaultdict(set)
            for next_iter_consumers, next_iter_producers in next_iter_dependencies:
                reverse_graph_dict[next_iter_consumers].update(next_iter_producers)

        return parallel_blocks

    def __str__(self):
        lines = []
        lines.append(f"Partition colors: {self.partition_colors}")
        lines.extend([f"{k} --> {v}" for k, v in self._graph.items()])
        return "\n".join(lines)


class SchedulePolicy(Enum):
    # default mode, based on original execution order
    strict = 1


# ---------------------------------------------------------------------------
# Provenance propagation for passes that mint a replacement node
# ---------------------------------------------------------------------------
#
# A pass that does
#
#     new = graph.call_function(...)
#     old.replace_all_uses_with(new)
#     graph.erase_node(old)
#
# starts ``new`` with an empty ``meta``.  Everything the graph knew about
# ``old`` -- above all the source line a profiler or a recipe-numbering consumer
# attributes it to -- is gone, and it is gone for exactly the nodes a reader most
# wants to attribute (the ones a rewrite touched).  ``torch.fx`` gives no warning:
# ``node_copy`` copies meta, ``call_function`` does not.
#
# ``PROVENANCE_META_KEYS`` is an allow-list, not "copy all of meta", and the
# distinction is deliberate.  The rest of ``meta`` describes *the computation the
# old node performed* -- ``val``, ``output_shapes``, ``output_strides``,
# ``output_contiguous``, ``output_offset``, ``dtype``, ``device``, ``placement``,
# ``partition_assigned``, ...  Those are re-derived by the pass that owns them
# (``fill_propagated_tensor_metadata_to_node`` here, ``pass_fake_propagation``
# later) and copying a stale one across a rewrite would assert something about a
# node that has not been measured yet.

PROVENANCE_META_KEYS = ("stack_trace", "source_fn_stack", "nn_module_stack")
"""The ``meta`` keys that say *where a node came from*, not *what it computes*.

``stack_trace``
    The Python stack that produced the node.  This is the only per-node source
    attribution the FX graph carries, and it is what ``region_identity`` numbers
    recipes by (``foo.1 .. foo.N`` ordered by the minimum source line among a
    recipe's nodes).  A replacement node without it makes the recipe place on
    partial evidence.

``source_fn_stack``
    dynamo's ``[(name, fn), ...]`` chain of the functions the node came from.
    Nothing in this bridge reads it (``grep -rn source_fn_stack`` over
    ``compile_backend`` returns nothing), but it is pure provenance about the
    same source expression, so it costs nothing to keep and it is what external
    tooling (``torch._dynamo``'s repro/debug helpers) attributes by.

``nn_module_stack``
    The enclosing ``nn.Module`` path.  This bridge *does* read it:
    ``propagate_module_names`` (``passes.py``, called from ``pass_compile_clusters``
    at the POST_PARTITIONER stage) filters FX nodes on ``"nn_module_stack" in
    node.meta`` before naming the JIT IR nodes, so a node inside a fused partition
    that loses it silently drops out of the IR name mapping.  ``scalar_reorder``
    runs at PRE_PLACEMENT, i.e. before the node is clustered, so this is exactly
    the key it must not drop.

``original_aten`` is deliberately NOT carried.  It is not provenance: it asserts
"this node is the decomposition of that aten op", and ``is_decomposed_from_inplace_node``
(``_helpers/helpers.py``) compares a node's own target against it.  A node minted
by a *rewrite* is not a decomposition, so carrying the key would make that
predicate lie -- ``div(scalar, tensor)`` rewritten to ``mul(reciprocal(t), scalar)``
would report itself as a decomposition of ``aten.div.Tensor``.
"""

_STACK_TRACE_SITE_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>[^\s]+)')


def stack_trace_site(trace: str | None) -> tuple[str, int] | None:
    """The innermost ``(file, line)`` of an FX ``meta["stack_trace"]``, or ``None``.

    A trace holds one frame per inlined call, outermost first.  We take the *last*
    match -- the innermost frame -- because that is the line that produced the node.
    This is the same rule ``region_identity.node_sites`` applies, which matters:
    if the two disagreed, a pass that "fixed" the trace would change the numbering.
    """
    if not trace:
        return None
    matches = list(_STACK_TRACE_SITE_RE.finditer(trace))
    if not matches:
        return None
    return (matches[-1].group("file"), int(matches[-1].group("line")))


def copy_provenance_meta(
    src: torch.fx.Node,
    dst: torch.fx.Node,
    overrides: dict | None = None,
) -> torch.fx.Node:
    """Copy the provenance keys of *src* onto *dst* (a replacement for *src*).

    ``overrides`` replaces an individual key; a value of ``None`` means "do not
    set that key at all" (used by view-chain fusion, which cannot honestly claim a
    single ``stack_trace``).  Returns *dst* so call sites read as one expression.
    """
    overrides = overrides or {}
    for key in PROVENANCE_META_KEYS:
        if key in overrides:
            continue
        if key in src.meta:
            dst.meta[key] = src.meta[key]
    for key, value in overrides.items():
        if value is not None:
            dst.meta[key] = value
    return dst

