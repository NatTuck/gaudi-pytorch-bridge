# SPDX-License-Identifier: Apache-2.0
"""Explicit inputs to compilation. One snapshot; no ambient reads downstream.

The principle: **compilation must be a real function of explicit inputs.** An
environment variable that changes compilation without being an explicit function
argument is a defect, not something to accommodate with a bigger hash. This module
is the single place ambient state is read; everything on the compile path reads
the frozen values captured here.

## What was implicit, and is now explicit

The audit found six categories. Two were already explicit (the graph, and the
tensor metadata). This module makes four of the rest explicit:

    context  grad_enabled, inference_mode, num_threads, default_dtype
    config   the flags that change the lowered graph, plus a digest of the WHOLE
             environment so an unenumerated variable still shows up
    instance whatever Python values the caller declares it depends on
    region   a bytecode hash of the region's function

## What this does NOT do

The graph hash remains the real cache key. This record is built alongside it and
logged, so the two can be compared. Flipping the key is stage 2 and needs a bridge
change -- `libs_env_hash`, `PT_RECIPE_CACHE_IGNORE_VERSION`, the shape-inference
flags, `unique_graph_cntr` and `op_strs` are all inside the C++ bridge and cannot
be supplied from here.

## The collision detector is the point

We will have missed something. The detector is the mechanism that finds it: if one
computed record ever maps to two different graph hashes, then the record is missing
an input, and it names the record and both hashes rather than failing quietly.
"""
from __future__ import annotations

import hashlib
import logging
import os
import types
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Flags that change the lowered graph. Read ONCE, at snapshot time.
_CONFIG_FLAGS = (
    "PT_HPU_USE_FUSE_SDPA_PASS",
    "PT_HPU_ENABLE_FAST_SHAPE_INFERENCE",
    "PT_HPU_RUN_HYBRID_SIF",
    "PT_HPU_USE_SYN_TENSOR_IDS",
    "PT_HPU_ENABLE_DYNAMIC_OUTPUT_PREALLOCATE",
    "PT_HPU_USE_JIT_FORK",
    "PT_HPU_ENABLE_H2D_SCALES",
)

_TRUTHY = frozenset(("on", "1", "yes", "true", "y", "t"))
_DISABLED_PREFIX = "PT_HPU_DISABLE_"

_snapshot: Mapping[str, str] | None = None


def _freeze() -> Mapping[str, str]:
    """The ONE place the environment is read.

    Built on first use rather than at import: `compile_backend` is imported from
    several directions and reading `os.environ` at import time would make module
    import order part of the record. First-use is still exactly once, and it is
    before any compile can observe it.

    The returned mapping is a plain dict and callers must not mutate it; it is
    treated as immutable by everything below.
    """
    global _snapshot
    if _snapshot is None:
        _snapshot = dict(os.environ)
        logger.info(
            "compile-inputs snapshot frozen: %d env entries, digest=%s",
            len(_snapshot), _env_digest(_snapshot))
    return _snapshot


def env() -> Mapping[str, str]:
    """The frozen environment. The compile path must use this, never `os.environ`."""
    return _freeze()


def flag(name: str, default: str = "") -> bool:
    """A boolean config flag, read from the frozen snapshot."""
    return env().get(name, default).strip().lower() in _TRUTHY


def pass_disabled(pass_name: str) -> bool:
    """Whether `PT_HPU_DISABLE_<pass_name>` was set AT SNAPSHOT TIME.

    This replaces a per-compile `os.getenv` in `passes.py`. The ruling is that the
    disabled-pass set is fixed at boot: nobody needs to flip it between compiles,
    and if they ever do, it becomes an explicit argument rather than an ambient
    read.
    """
    return env().get(_DISABLED_PREFIX + pass_name, "").strip().upper() in (
        "ON", "1", "YES", "TRUE", "Y")


def _env_digest(snap: Mapping[str, str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for key in sorted(snap):
        h.update(key.encode("utf-8", "replace"))
        h.update(b"=")
        h.update(snap[key].encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def exec_context() -> tuple[bool, bool]:
    """`(grad_enabled, inference_mode)`, the ONE place these are read.

    This is a legitimate input rather than a defect -- the context genuinely
    varies -- but it must be an EXPLICIT input, not ambient state the compiler
    reads behind the caller's back. Measured cost of getting this wrong: the
    `GdnSnapshotStore` region re-minted a recipe for three boots because the warm
    ran with `grad=True, inference_mode=False` and serving ran
    `grad=False, inference_mode=True`.
    """
    import torch

    return (torch.is_grad_enabled(), torch.is_inference_mode_enabled())


def _thread_and_dtype() -> tuple[int, str]:
    import torch

    return (int(torch.get_num_threads()), str(torch.get_default_dtype()))


def code_hash(fn: Any) -> str:
    """A bytecode hash of `fn`, NOT a hash of its source text.

    Per PROTOCOL §8 a comment must not change the record, and comments never reach
    bytecode. Note the corollary: a DOCSTRING lives in `co_consts`, so editing one
    does change this hash. That is deliberate -- a docstring is part of the module
    object -- but it is worth knowing before reading a change as a code change.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return "none"
    h = hashlib.blake2b(digest_size=8)
    h.update(code.co_code)
    h.update(repr(code.co_names).encode("utf-8", "replace"))
    h.update(repr(code.co_consts).encode("utf-8", "replace"))
    return h.hexdigest()


@dataclass(frozen=True)
class CompileInputs:
    """The explicit record. Frozen, so it cannot drift after construction."""

    context: tuple = ()
    config: tuple = ()
    instance: tuple = ()
    region: tuple = ()
    shapes: tuple = ()

    def key(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for field in (self.context, self.config, self.instance, self.region, self.shapes):
            h.update(repr(field).encode("utf-8", "replace"))
            h.update(b"|")
        return h.hexdigest()

    def describe(self) -> str:
        return (f"context={self.context} config={self.config} "
                f"instance={self.instance} region={self.region} shapes={self.shapes}")


def build(*, instance: tuple = (), region: tuple = (), shapes: tuple = ()) -> CompileInputs:
    """Construct the record from the frozen snapshot plus explicit arguments.

    `instance`, `region` and `shapes` are supplied by the caller because only the
    caller knows them. Everything else is read here, once.
    """
    snap = env()
    threads, dtype = _thread_and_dtype()
    context = (exec_context()[0], exec_context()[1], threads, dtype)
    config = tuple((name, flag(name)) for name in _CONFIG_FLAGS) + (
        ("env_digest", _env_digest(snap)),)
    return CompileInputs(
        context=context,
        config=config,
        instance=tuple(instance),
        region=tuple(region),
        shapes=tuple(shapes),
    )


# ---------------------------------------------------------------------------
# Collision detector
# ---------------------------------------------------------------------------

_seen: dict[str, str] = {}
_reported: set[tuple[str, str]] = set()


def record(inputs: CompileInputs, graph_hash: Any) -> None:
    """Compare one computed record against the real graph hash.

    THE DISCOVERY MECHANISM. If a single record maps to two different graph
    hashes, the record is missing an input -- the real compilation distinguished
    something we did not enumerate. That is reported by name, with both hashes,
    because the alternative is a cache that silently serves one graph for another.

    This is a log, not a raise: it must be able to run on a boot that is already
    instrumented, and an abort here would hide the very evidence it is collecting.
    """
    key = inputs.key()
    gh = str(graph_hash)
    prev = _seen.get(key)
    if prev is None:
        _seen[key] = gh
        if (key, gh) not in _reported:
            _reported.add((key, gh))
            logger.debug("compile-inputs %s -> graph_hash=%s", key, gh)
        return
    if prev != gh and (key, gh) not in _reported:
        _reported.add((key, gh))
        logger.error(
            "COMPILE-INPUTS COLLISION: one record maps to two graph hashes "
            "(%s vs %s). The record is missing an input -- compilation "
            "distinguished something not enumerated here. Record: %s",
            prev, gh, inputs.describe())


def reset() -> None:
    """Drop the detector's memory. Test seam; not used on the serving path."""
    _seen.clear()
    _reported.clear()
