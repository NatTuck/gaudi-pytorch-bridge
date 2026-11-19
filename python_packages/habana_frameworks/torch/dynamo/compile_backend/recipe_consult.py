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

WHAT THE ARTIFACT LOOKS LIKE ON DISK. `DiskCache::Add` ->
`RecipeCache::store(compile_key, ...)` -> `<cache_dir>/<compile_key>.recipe` and
`.metadata` (cache_file_handler.cpp:43, `cache_id_suffix_ == ""` at
hpu_recipe_cache.cpp:349). So one key, one file pair, and
`PrecompiledRegion::load(dir, key)` reads precisely that pair.

THE STORE IS ASYNCHRONOUS, AND THAT IS THE ONE THING TO GET RIGHT HERE.
`RecipeCache::store` (recipe_cache.cpp:298) does not write anything: it hands a
job to `cache_thread_->addJob`. The pair is written later, by that background
thread, recipe payload first and metadata second (recipe_cache.cpp:247-281).
`os.path.isfile("<key>.recipe")` is therefore true from the moment the payload
*starts*, not when it is finished -- and `<key>.metadata` exists from the moment
`store_task` opens it, empty, before serialising anything.

So file EXISTENCE says nothing about completeness. The only sound test is the
one the C++ reader itself uses (`RecipeCache::lookup`, recipe_cache.cpp:337):
metadata present AND non-empty. `artifact_complete` below is that test, and a
manifest is never written until it passes for every frame the bundle names --
see `record_bundle`. Writing the manifest first is what handed
`synRecipeDeSerialize` a half-written file in boot-g2.log (36 manifests, 34
recipes, `status 19`, `Handle doesn't exist`).

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

# Consult outcomes, so the explicit warm can report what actually happened.
#
# A warm CANNOT pre-check the cache. A region's key is a function of the
# arguments it is called with, so "is this region cached?" has no answer without
# calling it -- which is why the old `key_for_region(name)` pre-check could not
# work (it requires shapes/ints/instance and raised TypeError when called with
# just a name). The consult IS the skip logic: a warm calls each region, and a
# cached one returns from disk without tracing. These counters are the report.
# `load_failure` is a subset of `miss`, counted separately: a miss that came
# from an artifact we could not read is a different event from a miss that found
# nothing, and only the first one means the cache on disk is damaged.
STATS = {"hit": 0, "miss": 0, "load_failure": 0}


def stats_snapshot() -> tuple[int, int]:
    """(hits, misses) so far. Used to report a warm's effect.

    A failed load counts as a miss -- it is one -- so this stays a partition of
    the consults. The damage is counted separately by `load_failures`.
    """
    return STATS["hit"], STATS["miss"]


def load_failures() -> int:
    """How many artifacts were on disk but could not be loaded.

    Non-zero means the cache directory holds damage. It is never cleared, so a
    reader can tell "this boot saw none" from "this boot has not looked yet" by
    taking a snapshot first.
    """
    return STATS["load_failure"]


_MANIFEST_SUFFIX = ".manifest"

#: The metadata half of a recipe pair (`METADATA_SUFFIX`,
#: cache_file_handler.h:30). Written AFTER the recipe payload, and empty while
#: the payload is still being serialised -- which is exactly why the C++ reader
#: treats an empty one as "no entry".
_METADATA_SUFFIX = ".metadata"


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


def artifact_complete(cache_dir: str, key: str) -> bool:
    """True when `key`'s recipe pair is on disk AND finished being written.

    This is the C++ reader's own validity rule (`RecipeCache::lookup`:
    recipe_cache.cpp:337 -- an empty metadata file is "no entry", and it is
    removed). It is not a rule invented here; it is the vendor's, and using it
    means the consult and the loader cannot disagree about what "cached" means.

    The metadata is written after the payload by the same store task
    (recipe_cache.cpp:247-281) and closed, so a non-empty metadata file means
    `synRecipeSerialize` returned for that key. That is the whole completeness
    argument, and it is why the check is on the pair and not the payload alone.

    STRICTER THAN THE VENDOR IN ONE DIRECTION, deliberately. `lookup` checks the
    metadata only, and would hand a zero-byte payload to `synRecipeDeSerialize`
    and fail there -- `status 19`, `Handle doesn't exist`, the boot-g2.log crash.
    A non-empty metadata file proves a serialize ran, and a successful serialize
    never writes zero bytes, so an empty payload beside non-empty metadata is a
    truncated file, not a legitimate one. Declining it here turns that crash
    into a re-trace, which is the same outcome the load-failure path produces
    for free, reached earlier and without the failed syscall.
    """
    metadata = os.path.join(cache_dir, key + _METADATA_SUFFIX)
    try:
        return (os.path.getsize(metadata) > 0
                and os.path.getsize(artifact_path(cache_dir, key)) > 0)
    except OSError:
        return False


def artifact_state(cache_dir: str, key: str) -> str:
    """Why `key` is not loadable, as a short phrase for the log.

    Exists so that the two ways a manifest can fail to be useful are separable
    at a glance in a log: "metadata empty" / "payload missing" (knowable in
    advance, caught before any load) versus "cannot be loaded: <exception>"
    (the artifact looked complete and the deserialiser still rejected it). The
    first is a MISS on the incomplete branch; the second is a MISS on the
    load-failure branch. Different causes, different counters.
    """
    payload = artifact_path(cache_dir, key)
    if not os.path.isfile(payload):
        return "payload missing"
    try:
        if os.path.getsize(payload) == 0:
            return "payload empty"
    except OSError as exc:
        return f"payload unreadable ({type(exc).__name__})"
    metadata = os.path.join(cache_dir, key + _METADATA_SUFFIX)
    if not os.path.isfile(metadata):
        return "metadata missing"
    try:
        if os.path.getsize(metadata) == 0:
            return "metadata empty"
    except OSError as exc:
        return f"metadata unreadable ({type(exc).__name__})"
    return "complete"


def has_recipe(cache_dir: str, key: str) -> bool:
    """True when `key` is LOADABLE, not merely present.

    This used to be `os.path.isfile("<key>.recipe")`, which is the same fake
    detector this file already removed one of: it reports "loadable" without
    being able to see whether the artifact is loadable. `<key>.recipe` is
    created before a single byte of the payload is written -- `RecipeCache::store`
    is asynchronous -- so isfile() is true for a file that
    `synRecipeDeSerialize` cannot read. That is the boot-g2.log failure, one
    layer up: a check that answers a different question than the one being
    asked, and therefore always says yes.

    Delegating to `artifact_complete` puts the vendor's own validity rule here,
    so `consult` takes the clean "the manifest names a frame whose artifact is
    incomplete" MISS for something knowable in advance, instead of finding out
    by handing the file to the deserialiser. The load-failure path stays as the
    backstop for the genuinely unknowable case: an artifact that looks complete
    and still fails to load.
    """
    return artifact_complete(cache_dir, key)


def flush_recipe_cache() -> bool:
    """Drain the recipe cache's async store queue. True if it is drained.

    `RecipeCache::store` is asynchronous, so "the file exists" is not "the file
    is done". `RecipeCache::flush` (recipe_cache.cpp:128) is the vendor's own
    barrier: it spins until `cache_thread_->jobCounter()` reaches zero, and
    `job_thread.h:68` decrements that only after the store task -- payload plus
    metadata -- has returned. So when this returns True, every store queued
    before the call has finished.

    `habana_frameworks.torch.utils.debug._flush_disk_cache()` is that flush,
    reachable from Python (csrc/bindings.cpp:210 -> HPUDeviceContext::
    flush_disk_cache -> RecipeCacheLRU::FlushDiskCache -> DiskCache::flush).

    False means "cannot prove the queue drained", and the caller must treat it
    as "do not write a manifest". A missing manifest costs a re-trace on the
    next boot; a manifest over a half-written artifact cost the boot itself.
    """
    try:
        from habana_frameworks.torch.utils.debug import _flush_disk_cache
    except Exception as exc:  # noqa: BLE001 - no binding means no proof
        logger.error(
            "recipe consult: cannot flush the recipe cache (%s: %s), so no "
            "bundle manifest will be written this boot. The recipes may still "
            "be stored correctly; the next boot will simply re-trace instead of "
            "loading them, because nothing may claim a frame is loadable when "
            "its artifact is not yet complete.", type(exc).__name__, exc)
        return False
    try:
        _flush_disk_cache()
    except Exception as exc:  # noqa: BLE001 - a failed flush is not a proof
        logger.error(
            "recipe consult: flushing the recipe cache failed (%s: %s), so no "
            "bundle manifest will be written this boot.", type(exc).__name__, exc)
        return False
    return True


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

    PRECONDITION: every frame key passed here must already have a COMPLETE
    artifact pair on disk. A manifest is an assertion that the bundle is
    loadable, so writing one over an artifact the async store has not finished
    is a false assertion -- and the next boot believes it and hands the
    half-written file to `synRecipeDeSerialize`. `record_bundle` establishes
    that precondition; do not call this directly from anywhere else.

    The write itself is atomic (tmp + `os.replace`), so a reader never sees a
    partial manifest -- only a complete one, or none.
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

    The key is computed UNCONDITIONALLY, before the cache directory is even
    read, and it is never empty -- a miss with no cache directory configured
    still carries the real key. That ordering is what keeps every mint below a
    genuine "we consulted and recorded a miss" instead of a caller minting on
    its own behalf, and it is what makes the empty-key case impossible rather
    than merely unlikely.

    Ordering is the whole point: this runs before `torch.compile` is entered, so
    on a hit `torch.compile` is never entered at all.
    """
    from . import compile_inputs
    from .trace_gate import TraceGate

    # THE KEY COMES FIRST, BEFORE ANY BRANCH ON THE CACHE DIRECTORY.
    #
    # The key does not depend on the cache directory. It is a function of the
    # caller's arguments, the region's source identity and the frozen
    # environment snapshot, all of which exist whether or not a cache is
    # configured. Branching on the directory first is what made the old code
    # mint with `""` on the no-cache path: a caller minting on a branch that had
    # performed no lookup, carrying a key that names nothing. Both are
    # forbidden. The fix is ordering, not a special case.
    key = compile_inputs.make_record(fn, args, kwargs, module).key()

    # AN EMPTY KEY MUST BE IMPOSSIBLE BY CONSTRUCTION, NOT MERELY UNLIKELY.
    # `make_record` requires every field and hashes them, so this can only fire
    # if a future change lets one default away -- which is exactly the silent
    # collision that once produced one key for hundreds of distinct graphs. Fail
    # here, where the region can be named, rather than at a launch with an arity
    # mismatch that points nowhere.
    if not key:
        raise RuntimeError(
            f"recipe consult: region {region_name!r} produced an EMPTY cache key. "
            "An empty key names nothing: it can never be looked up, can never be "
            "stored, and a permit minted with it would authorise a trace without "
            "recording which compile it was for. This is a programming error in "
            "the record builder, not a cache problem -- "
            "compile_inputs.make_record must produce a non-empty key for every "
            "call.")

    cache_dir = recipe_cache_dir()
    if not cache_dir:
        # A REAL MISS WITH THE REAL KEY. LOUD, BUT NOT FATAL -- AND DO NOT
        # "FIX" THIS INTO A CRASH LATER.
        #
        # The invariant is "no trace without a recorded MISS", and a miss
        # carrying a valid key satisfies it exactly. A missing cache directory
        # is a SEPARATE operational misconfiguration: nothing can be looked up
        # and nothing can be stored, so every boot re-traces everything. That
        # must be loud, because it silently discards the entire point of the
        # cache, but it must not be fatal, because a no-cache run is a
        # legitimate configuration (first boot, cache disabled for a
        # measurement, a read-only filesystem). Crashing here would break those
        # runs while enforcing nothing the invariant does not already require.
        TraceGate.mint(region_name, key)
        logger.warning(
            "region consult: region=%s key=%s -> MISS (no recipe cache directory "
            "configured). This is a genuine miss, so the trace is permitted, but "
            "PT_HPU_RECIPE_CACHE_CONFIG names no directory: nothing will be "
            "stored, so every boot will re-trace this region from scratch.",
            region_name, key)
        STATS["miss"] += 1
        return key, MISS

    frames = read_manifest(cache_dir, key)

    if frames is None:
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (never traced)",
                    region_name, key)
        STATS["miss"] += 1
        return key, MISS

    if len(frames) != 1:
        # A bundle of N frames needs dynamo's chain replayed outside dynamo. Not
        # implemented, and pretending otherwise would be a false hit.
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (bundle has %d "
                    "frames; only a single-frame bundle can be served from disk "
                    "today)", region_name, key, len(frames))
        STATS["miss"] += 1
        return key, MISS

    frame_key = frames[0]
    if not has_recipe(cache_dir, frame_key):
        # INCOMPLETE, AND KNOWABLE IN ADVANCE. The manifest names a frame whose
        # artifact pair is not finished, so there is nothing to load and nothing
        # to repair here: this is an ordinary miss and the trace that follows
        # stores the artifact again. Deliberately worded differently from the
        # load-failure MISS below ("cannot be loaded: <exception>") so the two
        # are separable in a log: this one means the artifact was never
        # finished, that one means it looked finished and was rejected.
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (manifest names "
                    "frame %s but its artifact is incomplete: %s)",
                    region_name, key, frame_key,
                    artifact_state(cache_dir, frame_key))
        STATS["miss"] += 1
        return key, MISS

    from ._recipe_compiler_C import graph_launch, precompiled_load  # noqa: PLC0415

    inputs = _call_inputs(args, kwargs)
    try:
        recipe_id = precompiled_load(cache_dir, frame_key)
    except Exception as exc:  # noqa: BLE001 - a damaged artifact is a miss
        # A CACHE MISS, NOT A BOOT FAILURE.
        #
        # The artifact is on disk but cannot be deserialised: truncated by the
        # async store, or damaged. Both are recoverable by exactly the mechanism
        # this module exists for -- trace it and store it again -- so aborting
        # the process here trades a re-trace for a dead boot. That is what
        # boot-g2.log did: `synRecipeDeSerialize failed ... status 19` followed
        # by `Handle doesn't exist`, from an artifact the store had not finished
        # writing.
        #
        # THIS IS NOT A SILENT FALLBACK. It is logged at ERROR with the region,
        # the key and the exception; it is counted in `STATS["load_failure"]`,
        # separately from an ordinary miss, so a damaged cache is a number a
        # report can read rather than a line someone has to notice; and it mints
        # the permit the trace that follows needs, so the trace is legal. The
        # alternative -- propagating -- is not "safer", it is a boot that cannot
        # survive a cache it could have repaired by itself.
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        STATS["load_failure"] += 1
        logger.error(
            "region consult: region=%s key=%s -> MISS (artifact for frame %s is "
            "on disk but cannot be loaded: %s: %s). Treating it as a miss and "
            "re-tracing; the artifact will be stored again. This means the cache "
            "on disk is damaged or was written by an interrupted store.",
            region_name, key, frame_key, type(exc).__name__, exc)
        return key, MISS
    # A hit does NOT mint a permit: nothing may be traced on this path, and if
    # anything tries, the gate must fire. That is the enforcement, not an
    # oversight.
    outputs: list = []
    logger.info("region consult: region=%s key=%s -> HIT (frame %s, recipe %d); "
                "no trace", region_name, key, frame_key, recipe_id)
    STATS["hit"] += 1
    return key, graph_launch(recipe_id, inputs, outputs, region_name)


def record_bundle(region_name: str, region_key: str, frames: list[str]) -> None:
    """Called after a trace, to make the next boot's consult a hit.

    A MANIFEST MUST NEVER EXIST FOR AN ARTIFACT THAT IS NOT COMPLETELY WRITTEN.

    That is the whole contract, and it is why this function does not simply
    write: the store that produced these frames is asynchronous, so at the
    moment the trace returns the artifacts are at best queued. Three steps, in
    this order, and a manifest only at the end:

      1. FLUSH. `flush_recipe_cache()` waits for the vendor's own store queue to
         drain, so every artifact named below is finished when it returns. This
         is a synchronous barrier, not a sleep and not a guess.
      2. VERIFY. Each frame must satisfy `artifact_complete` -- the C++ reader's
         own rule. The flush proves the *current* queue drained; the check is
         what catches an artifact that was never stored at all (a store that
         failed, or a frame served from a previous boot's disk entry), and a
         pre-existing partial pair left by an older boot.
      3. WRITE. Only now. Atomically.

    If step 1 or step 2 cannot be satisfied the manifest is NOT written, and the
    reason is logged at ERROR. The cost is a re-trace next boot -- the recipe
    payloads themselves may well be fine. The alternative is a manifest that
    asserts a frame is loadable when it is not, which is what killed boot-g2.log
    (`synRecipeDeSerialize failed ... status 19`, `Handle doesn't exist`) and is
    not recoverable at the point of use.
    """
    cache_dir = recipe_cache_dir()
    if not cache_dir or not region_key:
        return
    if not frames:
        return

    if not flush_recipe_cache():
        # Already logged by the flush. Refusing to write is the safe half of the
        # choice: a missing manifest is a slow boot, a lying one is a dead boot.
        return

    incomplete = [frame for frame in frames
                  if not artifact_complete(cache_dir, frame)]
    if incomplete:
        logger.error(
            "region consult: NOT recording the bundle for region=%s key=%s: %d of "
            "%d frame artifact(s) are missing or incomplete (%s). A manifest is a "
            "claim that the bundle is loadable, so none is written; this region "
            "will re-trace next boot. The frames were flushed, so this means the "
            "store failed or the artifact pair is damaged, not that it is slow.",
            region_name, region_key, len(incomplete), len(frames),
            ", ".join(incomplete[:5]) + (" ..." if len(incomplete) > 5 else ""))
        return

    write_manifest(cache_dir, region_key, frames)
    logger.info("region consult: recorded bundle region=%s key=%s frames=%d "
                "(artifacts flushed and complete)", region_name, region_key,
                len(frames))
