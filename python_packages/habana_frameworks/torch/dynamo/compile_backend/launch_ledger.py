###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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
#
# Modified here. The original licence above is Apache-2.0; the changes in this
# file are Copyright Nat Tuck under the GNU AGPL-3.0-only. The conflict is
# recorded rather than hidden, as AGENTS.md requires.
###############################################################################

"""The compile-time half of the launch ledger.

The C++ half (`backend/synapse_helpers/launch_ledger.h`) writes one fixed-size
record per recipe launch, and one row per compiled recipe holding the recipe's
name and the FNV-1a key of that name. That is everything the launch path can
know cheaply. It does not know which *region* a recipe belongs to, because a
region is a dynamo frame and a recipe is what the compiler built for it.

This module supplies the missing half, at compile time:

    recipe_id  ->  (region name, region digest, `foo.N` index)

`recipe_id` is the id `graph_compile` returns. It is recoverable from the
recipe name on the C++ side without any extra plumbing, because
`GraphExecsGroup::generate_graph_index()` encodes it:

    graph_index = 1_000_000 + recipe_id * 1000 + variant

and `HabanaLaunchOpPT::syn_graph_name_` embeds that in the recipe name, e.g.

    .graph_dumps/graph_0004_fused_0_syn_1004000_1004000_5021

so `recipe_id = (1004000 - 1000000) // 1000 = 4`. The join is done at report
time, off the hot path, by `tools/launch_ledger_report.py`.

Numbering is `tools/region_identity.assign_ordinals`, unchanged: recipes are
labelled `foo.1 .. foo.N` ordered by minimum source line. Nothing here invents
a numbering.

Region *identity*: `region_identity.region_id` is the digest of a decorated
function's code object, which needs `@region` at the `torch.compile` call site.
The vLLM plugin has many such call sites and changing them would change the arm
under measurement, so the identity used here is derived from the frame's own
source position instead (file:function of the minimum-line node, plus a digest
of the partitions' structure). Both halves are stable across boots; the report
says which one it used and what the decorator transport would add.

Everything is a no-op unless `VLLM_HPU_LAUNCH_LEDGER` names a directory, the
same variable the C++ half reads.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading

_ENV_DIR = "VLLM_HPU_LAUNCH_LEDGER"
_ENV_REGION_DIR = "VLLM_HPU_REGION_IDENTITY_DIR"
_DEFAULT_REGION_DIR = "/home/nat/Qwen27B/tools/region_identity"

_lock = threading.Lock()
_regions: dict[str, "_Region"] = {}
_ri = None
_ri_tried = False
_warned = set()


def out_dir() -> str | None:
    d = os.environ.get(_ENV_DIR)
    return d if d else None


def enabled() -> bool:
    return bool(os.environ.get(_ENV_DIR))


def _warn_once(what: str, detail: str) -> None:
    if what in _warned:
        return
    _warned.add(what)
    sys.stderr.write(f"[launch-ledger] {what}: {detail}\n")
    sys.stderr.flush()


def _region_identity():
    """Import tools/region_identity, or return None and say why once."""
    global _ri, _ri_tried
    if _ri_tried:
        return _ri
    _ri_tried = True
    path = os.environ.get(_ENV_REGION_DIR, _DEFAULT_REGION_DIR)
    try:
        if path not in sys.path:
            sys.path.insert(0, path)
        import region_identity  # noqa: PLC0415

        _ri = region_identity
    except Exception as exc:  # noqa: BLE001 - the ledger must never break a boot
        _warn_once("region_identity unavailable", f"{path}: {exc!r}")
        _ri = None
    return _ri


class _Region:
    """The recipes of one dynamo frame, before and after numbering."""

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.parts: list[tuple[object, object]] = []  # (RecipeRecord, gm)
        self.sealed = False
        self.name = ""
        self.region_id = ""
        self.notes: list[str] = []
        self.n_recipes = 0


class Handle:
    """What `get_callable_recipe` hands back to `HabanaGraphModule`."""

    __slots__ = ("graph_name", "index")

    def __init__(self, graph_name: str, index: int) -> None:
        self.graph_name = graph_name
        self.index = index


def record_partition(graph_name: str, gm) -> "Handle | None":
    """Called once per partition, from `get_callable_recipe`.

    Cheap on purpose: it stores the graph module and numbers nothing. The
    numbering has to wait until the frame's partitions are all known.
    """
    if not enabled() or graph_name is None:
        return None
    with _lock:
        reg = _regions.get(graph_name)
        if reg is None:
            reg = _regions[graph_name] = _Region(graph_name)
        index = len(reg.parts)
        reg.parts.append((None, gm))
    return Handle(graph_name, index)


def _seal(reg: _Region) -> None:
    """Compute the region identity and the `foo.N` numbering, once."""
    ri = _region_identity()
    if ri is None:
        reg.name = f"unplaced:{reg.graph_name}"
        reg.region_id = reg.name
        reg.sealed = True
        return
    try:
        sites = []
        structures = []
        for _, gm in reg.parts:
            try:
                sites.extend(ri.node_sites(gm))
                structures.append(ri.structural_digest(gm))
            except Exception as exc:  # noqa: BLE001
                reg.notes.append(f"partition unplaceable: {exc!r}")
        if not sites:
            reg.name = f"unplaced:{reg.graph_name}"
            reg.region_id = reg.name
            reg.sealed = True
            reg.notes.append("no node carried a stack trace")
            return
        first = min(sites, key=lambda s: s.line)
        human = f"{os.path.basename(first.file)}:{first.func}"
        digest = hashlib.sha256("\0".join(structures).encode()).hexdigest()[:8]
        reg.name = f"{human}#{digest}"
        reg.region_id = reg.name
        records = []
        slot_of = {}
        for i, (_, gm) in enumerate(reg.parts):
            try:
                shape = ri.recipe_shape(gm)
                structure = ri.structural_digest(gm)
            except Exception as exc:  # noqa: BLE001
                reg.notes.append(f"recipe {i} unplaceable: {exc!r}")
                continue
            rec = ri.RecipeRecord(
                region_id=reg.region_id,
                name=reg.name,
                shape=shape,
                structure=structure,
                discovery_index=i,
                gm=gm,
            )
            records.append(rec)
            slot_of[id(rec)] = i
        if not records:
            reg.sealed = True
            return
        numbering = ri.assign_ordinals(records)
        reg.notes.extend(numbering.notes)
        for rec in numbering.ordered:
            reg.parts[slot_of[id(rec)]] = (rec, rec.gm)
        reg.n_recipes = len(records)
    except Exception as exc:  # noqa: BLE001
        reg.name = f"unplaced:{reg.graph_name}"
        reg.region_id = reg.name
        reg.notes.append(f"seal failed: {exc!r}")
    finally:
        reg.sealed = True


def note_recipe(recipe_id, handle: "Handle | None") -> None:
    """Called from `HabanaGraphModule.__call__` once `graph_compile` returned.

    This is the first moment at which a recipe id exists for a partition, and it
    is always after the frame's partitions were created, so it is also the
    moment the numbering can be sealed.
    """
    if not enabled() or handle is None or recipe_id is None:
        return
    d = out_dir()
    if d is None:
        return
    with _lock:
        reg = _regions.get(handle.graph_name)
        if reg is None:
            return
        if not reg.sealed:
            _seal(reg)
        rec = reg.parts[handle.index][0] if handle.index < len(reg.parts) else None
        label = getattr(rec, "label", None) if rec is not None else None
        min_line = getattr(getattr(rec, "shape", None), "min_line", None)
        max_line = getattr(getattr(rec, "shape", None), "max_line", None)
        n_nodes = getattr(getattr(rec, "shape", None), "n_nodes", None)
        notes = " | ".join(reg.notes)
        row = "\t".join(
            str(v)
            for v in (
                int(recipe_id),
                label if label is not None else "",
                reg.region_id,
                reg.name,
                min_line if min_line is not None else "",
                max_line if max_line is not None else "",
                n_nodes if n_nodes is not None else "",
                reg.n_recipes,
                handle.index,
                handle.graph_name,
                notes,
            )
        )
    path = os.path.join(d, f"regions.{os.getpid()}.tsv")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (row + "\n").encode())
        finally:
            os.close(fd)
    except Exception as exc:  # noqa: BLE001
        _warn_once("cannot write the region table", f"{path}: {exc!r}")
