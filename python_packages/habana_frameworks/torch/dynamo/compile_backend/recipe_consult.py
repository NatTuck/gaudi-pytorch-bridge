"""Consult the recipe cache BEFORE anything is traced.

THE INVARIANT (docs/PROTOCOL.md): no tracing may happen until a cache consult
has recorded a genuine MISS. This module is the only place that consult happens,
and therefore the only place that is allowed to mint a `trace_gate` permit.

Why this file exists at all: the recipe key used to be computed per FX frame,
inside the dynamo backend -- that is, AFTER tracing. So the cache was consulted
after the trace had already been paid for, and a hit saved only the Synapse
compile. The trace is the expensive part (measured: ~3.5 s per frame on this
model), so a hit that still traces is not the thing this work is for. The key
has to be a function of what the CALLER already holds.

WHAT IS COMPUTABLE WITHOUT A TRACE, and therefore what the key may contain:
  * the region's identity -- module:qualname, and a normalised-AST hash of its
    source closure (`compile_inputs.code_identity`)
  * the caller's real arguments -- tensor shapes/dtypes/layouts, and every
    Python int, via `compile_inputs._signature_of`
  * the guarded module attributes -- `compile_inputs._instance_attributes`
  * exec context (grad/inference mode, thread count, default dtype)
  * enumerated config flags, torch version, synapse version, bridge source hash
That set is exactly `compile_inputs.make_record`. Nothing graph-derived is in it,
because a graph only exists after tracing.

WHAT THE ARTIFACT LOOKS LIKE ON DISK. The store side is unchanged and already
correct: `DiskCache::Add` -> `RecipeCache::store(compile_key, ...)` ->
`<cache_dir>/<compile_key>.recipe` and `.metadata` (cache_file_handler.cpp:43,
`cache_id_suffix_ == ""` at hpu_recipe_cache.cpp:349). So one key, one file pair,
and `PrecompiledRegion::load(dir, key)` reads precisely that pair.

ONE REGION CAN BE MANY FRAMES. A single `torch.compile` target produces one FX
frame per inner function dynamo compiles, each with its own frame key and its own
artifact. A region-granularity key therefore names a BUNDLE, not one file. The
bundle is recorded in `<cache_dir>/<region_key>.manifest` -- the ordered list of
frame keys that one trace of this region produced -- written as a side effect of
the trace that produced them.

Serving a bundle trace-free means replaying dynamo's frame chain, which is only
trivial when the bundle holds ONE frame. This module serves N == 1 from disk and
reports N > 1 as a MISS with the reason named, rather than pretending. The MISS
is genuine in the sense that matters to the gate: we did consult, we found that
we cannot serve this region from disk today, and so we are permitted to trace.
A MISS that is really a "cannot serve" must be distinguishable in the log, which
is why the reason is recorded.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

#: Returned by `consult` when the caller must trace. The permit has already been
#: minted, so the trace is legal.
MISS = "MISS"

_MANIFEST_SUFFIX = ".manifest"


def recipe_cache_dir() -> str | None:
    """The recipe cache directory from PT_HPU_RECIPE_CACHE_CONFIG, or None.

    The config is `<path>,<skip>,<size>`. Only the path is needed here. This is
    the same source `DiskCache` is constructed from, so the consult and the store
    cannot disagree about where artifacts live.
    """
    cfg = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "")
    path = cfg.split(",", 1)[0].strip()
    return path or None


def artifact_path(cache_dir: str, key: str) -> str:
    """The recipe file for `key`. Exact, not a glob.

    `<key>.recipe` -- see the module docstring. A glob here would be wrong twice
    over: it would miss the real name, and it would match an unrelated key that
    happens to share this one as a prefix.
    """
    return os.path.join(cache_dir, f"{key}.recipe")


def has_recipe(cache_dir: str, key: str) -> bool:
    return os.path.isfile(artifact_path(cache_dir, key))


def manifest_path(cache_dir: str, region_key: str) -> str:
    return os.path.join(cache_dir, region_key + _MANIFEST_SUFFIX)


def read_manifest(cache_dir: str, region_key: str) -> list[str] | None:
    """The ordered frame keys for this region, or None if never traced.

    A manifest that exists but cannot be read is a hard error: it is our own
    artifact, so a corrupt one means the cache was damaged, and silently
    retracing would hide that.
    """
    path = manifest_path(cache_dir, region_key)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"recipe consult: the manifest for region key {region_key} at {path} "
            f"cannot be read ({type(exc).__name__}: {exc}). It is this project's "
            f"own artifact, so a damaged one means the cache is damaged; "
            f"retracing silently would hide that.") from exc
    frames = data.get("frames")
    if not isinstance(frames, list) or not all(isinstance(f, str) for f in frames):
        raise RuntimeError(
            f"recipe consult: the manifest for region key {region_key} at {path} "
            f"has no valid 'frames' list. Refusing to guess.")
    return frames


def write_manifest(cache_dir: str, region_key: str, frames: list[str]) -> None:
    """Record the ordered frame keys one trace of this region produced.

    Written after the trace, because the frame keys only exist then. This is the
    one thing that is legitimately a product of tracing, and it is metadata
    ABOUT the bundle, not an input to the key.
    """
    path = manifest_path(cache_dir, region_key)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"region_key": region_key, "frames": frames}, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def _call_inputs(args: tuple, kwargs: dict) -> tuple:
    """The positional argument tuple a recipe launch expects.

    The recipe was compiled from a frame whose inputs are the region's own
    parameters, in order. Keyword arguments are refused rather than reordered:
    the mapping from keyword names to placeholder order is a property of the
    trace, and guessing it is exactly the class of bug this work removes.
    """
    if kwargs:
        raise RuntimeError(
            "recipe consult: a region was called with keyword arguments "
            f"({sorted(kwargs)}). The recipe's input order comes from the traced "
            "placeholder order, which cannot be reconstructed from keyword "
            "names without tracing. Call the region positionally.")
    return tuple(args)


def consult(*, fn, args: tuple, kwargs: dict, module, region_name: str):
    """Consult the cache for this region call.

    Returns `(key, result)`. `result` is the region's value if a single-frame
    bundle was loaded and launched from disk, with NO tracing having happened.
    `result is MISS` means the caller must trace, and a `trace_gate` permit has
    already been minted, so that trace is legal.

    The key is returned even on a miss because the caller must publish it as the
    declared identity: that is how the frames this trace produces get attributed
    to this region and the bundle gets written for the next boot.

    Ordering is the whole point: this runs before `torch.compile` is entered, so
    on a hit `torch.compile` is never entered at all.
    """
    from . import compile_inputs
    from .trace_gate import TraceGate

    cache_dir = recipe_cache_dir()
    if not cache_dir:
        # No cache directory configured: there is genuinely nothing to consult,
        # so this is a real miss. Mint and say why.
        TraceGate.mint(region_name, "")
        logger.info("region consult: region=%s key=NONE -> MISS (no recipe cache "
                    "directory configured)", region_name)
        return "", MISS

    key = compile_inputs.make_record(fn, args, kwargs, module).key()
    frames = read_manifest(cache_dir, key)

    if frames is None:
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (never traced)",
                    region_name, key)
        return key, MISS

    if len(frames) != 1:
        # A bundle of N frames needs dynamo's chain replayed outside dynamo. Not
        # implemented, and pretending otherwise would be a false hit.
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (bundle has %d "
                    "frames; only a single-frame bundle can be served from disk "
                    "today)", region_name, key, len(frames))
        return key, MISS

    frame_key = frames[0]
    if not has_recipe(cache_dir, frame_key):
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (manifest names "
                    "frame %s but its artifact is absent)",
                    region_name, key, frame_key)
        return key, MISS

    from ._recipe_compiler_C import graph_launch, precompiled_load  # noqa: PLC0415

    inputs = _call_inputs(args, kwargs)
    recipe_id = precompiled_load(cache_dir, frame_key)
    # A hit does NOT mint a permit: nothing may be traced on this path, and if
    # anything tries, the gate must fire. That is the enforcement, not an
    # oversight.
    outputs: list = []
    logger.info("region consult: region=%s key=%s -> HIT (frame %s, recipe %d); "
                "no trace", region_name, key, frame_key, recipe_id)
    return key, graph_launch(recipe_id, inputs, outputs, region_name)


def record_bundle(region_name: str, region_key: str, frames: list[str]) -> None:
    """Called after a trace, to make the next boot's consult a hit."""
    cache_dir = recipe_cache_dir()
    if not cache_dir or not region_key:
        return
    write_manifest(cache_dir, region_key, frames)
    logger.info("region consult: recorded bundle region=%s key=%s frames=%d",
                region_name, region_key, len(frames))
