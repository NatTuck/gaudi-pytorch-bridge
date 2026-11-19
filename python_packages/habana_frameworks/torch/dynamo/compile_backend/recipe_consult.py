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

import atexit
import json
import logging
import os
import struct
import threading

logger = logging.getLogger(__name__)

# DIAGNOSTIC -- KEEP UNCONDITIONAL FOR NOW, REVISIT AFTER THE GATE EXPERIMENT.
#
# THIS MODULE'S `-> HIT` AND ORDINARY `-> MISS` LINES ARE INVISIBLE WITHOUT THIS.
# `__name__` here is `habana_frameworks.torch.dynamo.compile_backend.
# recipe_consult`, so this logger is a CHILD of `habana_frameworks`, and it
# inherits that package's level rather than vLLM's root level. vLLM sets the root
# from VLLM_LOGGING_LEVEL; `habana_frameworks` sets its own, higher. So on a real
# boot every `logger.info` in this file is filtered while vLLM's own INFO lines
# still appear -- which is why boot-p9.log contains exactly three `region
# consult` lines, all of them from the `logger.error` launch-failure path, and
# none of the hits or ordinary misses.
#
# THAT IS A MEASUREMENT BLOCKER, NOT A COSMETIC ISSUE. Without these lines no
# boot can say WHICH regions hit, so a failure that depends on which region hit
# is indistinguishable from one that depends on any hit at all -- which is
# precisely the question the gate below exists to answer.
#
# Set explicitly rather than via the parent, so it survives any later change to
# the `habana_frameworks` level. `logger.setLevel` on a child overrides the
# inherited level without touching the parent or any sibling.
#
# THE FIX IS A HANDLER, NOT A LEVEL. `logger.setLevel(INFO)` alone was tried and
# did nothing, because a record that passes this logger's own level still has to
# pass an ANCESTOR's handler, and the `habana_frameworks` handler is above INFO.
# So this logger gets its own StreamHandler (stderr, which launch-dflash.sh
# redirects into the boot log) and stops propagating, so nothing is emitted twice.
_diag_handler = logging.StreamHandler()
_diag_handler.setLevel(logging.INFO)
_diag_handler.setFormatter(logging.Formatter("[recipe-consult] %(message)s"))
logger.addHandler(_diag_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

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
#
# `launch_failure` is the same idea one step later: the artifact loaded and the
# launch itself raised. That is a residual defect in the loaded recipe (or in the
# arguments we handed it), it is recoverable by tracing, and it must be a MISS
# rather than a dead boot -- but it must also be a NUMBER, so that "the cache is
# being used" and "the cache is being used and failing" are distinguishable.
STATS = {"hit": 0, "miss": 0, "load_failure": 0, "launch_failure": 0}


def stats_snapshot() -> tuple[int, int]:
    """(hits, misses) so far. Used to report a warm's effect.

    A failed load counts as a miss -- it is one -- so this stays a partition of
    the consults. The damage is counted separately by `load_failures`.
    """
    return STATS["hit"], STATS["miss"]


def launch_failures() -> int:
    """How many loaded recipes failed to LAUNCH.

    Non-zero means a trace-free hit reached `graph_launch` and the launch raised.
    Like `load_failures`, it is never cleared and is a subset of `miss`.
    """
    return STATS["launch_failure"]


def load_failures() -> int:
    """How many artifacts were on disk but could not be loaded.

    Non-zero means the cache directory holds damage. It is never cleared, so a
    reader can tell "this boot saw none" from "this boot has not looked yet" by
    taking a snapshot first.
    """
    return STATS["load_failure"]


_MANIFEST_SUFFIX = ".manifest"

# BUNDLES TRACED BUT NOT YET RECORDED AS MANIFESTS.
#
# WHY DEFERRED, AND NOT WRITTEN WHERE THEY ARE PRODUCED. `record_bundle` is
# called from `Region.__call__`'s `finally`, on the CALLING thread -- but these
# regions compile on the LOWERING thread, and `HabanaLaunchOpPipeline::
# LoweringTask` is ASYNCHRONOUS. When the region call returns, that thread has
# not yet reached `graph_compile` -> `DiskCache::Add` -> `RecipeCache::store`, so
# the store job is not merely unfinished, it is not QUEUED. `RecipeCache::flush`
# spins on a job counter that is vacuously zero at that instant, and a
# verification that follows it sees no artifact at all.
#
# That is boot-p1.log: 105 bundles refused, 1 manifest written, and the refused
# frame keys complete on disk minutes later. The earlier flush-and-verify was
# structurally too early -- a flush cannot prove anything about a job that has
# not been created yet.
#
# So the write moves to a point where the lowering thread is idle:
# `flush_bundles` drains the pipeline and only then verifies and writes. Bundles
# accumulate here, keyed by (cache_dir, region_key) so a re-trace of the same
# region REPLACES rather than duplicating -- the pending set is bounded by the
# number of distinct regions, not by the number of calls.
#
# CONSEQUENCE, STATED PLAINLY. A manifest may not exist during the boot that
# produced it, so a second call with identical arguments IN THE SAME PROCESS
# consults, finds no manifest, and reports a MISS. That is correct for a
# cross-boot cache: the compiled callable is already built and `Region` reuses
# `self._compiled`, so nothing re-traces -- the cost is a consult MISS instead of
# a disk HIT. Do not "fix" that by writing the manifest early; that is the bug.
#
# The lock is required, not decorative: `record_bundle` runs on whichever thread
# called the region, and `flush_bundles` runs at a quiescent point, so the two
# can differ.
_PENDING_LOCK = threading.Lock()
_PENDING: "dict[tuple[str, str], tuple[str, list[str], list]]" = {}

#: How many NEW bundles may accumulate before the boot path wakes the writer.
#:
#: 0 means DISABLED, and 0 is the shipped value: nothing wakes the writer from
#: the boot path, so a boot that never crosses a threshold records only at the
#: post-warmup call and at exit.
#:
#: WHY A THRESHOLD AT ALL, when exit-time recording already exists: the boots we
#: used to run were KILLED. `timeout` sends SIGTERM, SIGTERM does not run
#: `atexit` handlers, and the workflow depends on the killed boot having done
#: useful work -- "if the first run times out, it did some of the compiles, so
#: the second run should finish". Under exit-only, run 1 left 104 recipes on disk
#: and ZERO manifests, so run 2 hit nothing and the cache could never bootstrap.
#: That is the property this knob restores when it is turned on.
#:
#: WHY IT IS DISABLED, AND THIS IS MEASURED RATHER THAN CAUTIOUS. Writing
#: manifests DURING the boot, from the thread that happens to call a region,
#: breaks weight loading:
#:
#:   p6  threshold 8, binding check removed (so no refusal storm)
#:       -> `AssertionError: Attempted to load weight (torch.Size([5120])) into
#:          parameter (torch.Size([0]))` at device_weight_regions.py:508
#:          _hpu_default_weight_loader. No startup.
#:   p7  threshold effectively off, same build
#:       -> RESULT=STARTUP_OK elapsed=125s, and 201 manifests + 201 recipes were
#:          still recorded -- by ATEXIT, because a boot that reaches startup
#:          exits normally.
#:
#: So the refusal storm was NOT the trigger; ANY flush executing on the
#: weight-loading thread is. `Region.__call__`'s `finally` runs `record_bundle` on
#: the thread that called the region -- for weight loading that is the loader
#: thread -- so an in-boot flush puts file I/O and logging between two weight
#: copies.
#:
#: AND THE TRIGGER IS A PERTURBATION, NOT THE MECHANISM. p2 (drain ON) and p5
#: (drain OFF, no barrier at all) produced the SAME assertion from completely
#: different perturbations. A shared symptom from unrelated perturbations is the
#: signature of a pre-existing race in weight loading that our flush merely
#: exposes. The flush cannot make a parameter have size [0]; it can only let
#: another thread run at a point where the loader thread previously did not
#: yield.
#:
#: THE TRADEOFF, STATED PLAINLY: with this at 0, a boot KILLED by `timeout`
#: (SIGTERM, which does not run atexit handlers) records NOTHING. That is
#: acceptable only while boots complete, which p7 does. IF BOOTS START TIMING OUT
#: AGAIN THIS MUST BE RE-ENABLED -- and re-enabling it means either fixing the
#: underlying race in weight loading, or moving the I/O off the calling thread
#: first (which is what the writer thread below exists to do).
#:
#: Re-enabling is a one-line change: set this above 0 and the boot path wakes the
#: writer every N new bundles. It is deliberately NOT re-enabled yet: the writer
#: must be proven first, one variable at a time.
#:
#: WHEN IT IS RE-ENABLED, 8 is the value to use, and the reasoning is the one
#: that produced it: boot-p1.log enqueued 105 bundles, so 8 caps what a killed
#: boot loses at 7 bundles (~7% of one boot's traced regions).
#:   * 1 records everything but wakes the writer after EVERY region call.
#:   * 32+ loses up to 31 bundles, a third of a boot's work if killed early.
#: 8 keeps the loss to a handful and is small enough that a boot killed after its
#: first few regions has already recorded something -- the property the bootstrap
#: needs. With the writer thread doing the work, the cost of a wake is an
#: Event.set() on the calling thread, so the only cost of a small N is the
#: writer's own scheduling.
_BUNDLE_FLUSH_THRESHOLD = 0

#: New bundles enqueued since the last flush. Guarded by `_PENDING_LOCK`.
_SINCE_FLUSH = 0

# THE MANIFEST WRITER THREAD.
#
# WHY A THREAD AND NOT A CALL. Every syscall and every log line that a manifest
# write performs releases the GIL. Done on the weight-loading thread -- which is
# where `record_bundle` runs -- that yields at points where the loader previously
# ran uninterrupted, and p2 (barrier) and p5/p6 (no barrier) both ended in
# `Attempted to load weight ... into parameter (torch.Size([0]))`. The evidence
# says the flush is a perturbation that exposes a pre-existing race in weight
# loading, so the fix is to stop perturbing: the loader thread only ever appends
# to a dict and sets an Event, and ALL filesystem and logging work happens here.
#
# DAEMON: it must never keep the process alive. A boot that would otherwise exit
# is not held open by a thread whose only job is to write a cache hint.
#
# LAZY: the thread is created on the first wake, so a boot that never crosses the
# threshold -- which is every boot today, since `_BUNDLE_FLUSH_THRESHOLD` is 0 --
# never starts a thread at all.
_WRITER_LOCK = threading.Lock()
_WRITER: "threading.Thread | None" = None
_WAKE = threading.Event()
_STOP = threading.Event()

#: Serialises flush passes so a synchronous caller (post-warmup, exit) can wait
#: for an in-progress writer pass and then process whatever is left. Without it
#: the two could swap the queue out from under each other and "the post-warmup
#: call drained everything" would not be true.
_FLUSH_LOCK = threading.Lock()


def _writer_loop() -> None:
    """Write queued bundles until told to stop. Runs OFF the calling thread."""
    while True:
        _WAKE.wait()
        _WAKE.clear()
        if _STOP.is_set():
            # Whatever is still queued is written by the joiner in
            # `_flush_bundles_at_exit`, so nothing is lost by leaving now.
            return
        try:
            # drain=False: this thread must never take a global barrier. It is
            # not the main thread, and the barrier's value is only in making a
            # first attempt likely to succeed -- never worth a wrong-thread join.
            _flush_pending(drain=False)
        except Exception:  # noqa: BLE001 - a writer fault must not kill anything
            logger.exception(
                "region consult: the manifest writer failed a pass. The bundles "
                "it had taken from the queue may not have been recorded; they "
                "will be re-traced next boot.")


def _wake_writer() -> None:
    """Start the writer if needed and signal it. No I/O; safe on any thread."""
    global _WRITER
    with _WRITER_LOCK:
        if _WRITER is None or not _WRITER.is_alive():
            _WRITER = threading.Thread(target=_writer_loop,
                                       name="recipe-manifest-writer",
                                       daemon=True)
            _WRITER.start()
    _WAKE.set()

# The two barrier callables, resolved ONCE during the boot and reused at exit.
#
# NOT resolved lazily at exit, and that is a measured requirement, not tidiness:
# an import performed from an `atexit` handler can fail with "can't register
# atexit after shutdown", which is how the exit-path test lost every bundle in
# the boot while reporting only that it could not flush. Resolving here, on the
# calling thread while the interpreter is healthy, makes the exit handler a pure
# call. `record_bundle` warms these; `_resolve_hooks` is idempotent, so a boot
# that never calls a region simply never resolves them.
_SETTLE_HOOK = None
_FLUSH_HOOK = None


def _resolve_hooks() -> None:
    """Resolve the barrier callables if not already resolved. Never raises.

    Deliberately lazy w.r.t. module scope: importing `torch`/habana at import
    time is what this module has always avoided. It is NOT lazy w.r.t. the
    exit path -- see the comment on `_SETTLE_HOOK`.
    """
    global _SETTLE_HOOK, _FLUSH_HOOK
    if _SETTLE_HOOK is None:
        try:
            import torch  # noqa: PLC0415
            _SETTLE_HOOK = torch.hpu.synchronize
        except Exception:  # noqa: BLE001 - absence is reported where it matters
            pass
    if _FLUSH_HOOK is None:
        try:
            from habana_frameworks.torch.utils.debug import _flush_disk_cache  # noqa: PLC0415
            _FLUSH_HOOK = _flush_disk_cache
        except Exception:  # noqa: BLE001
            pass

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

    False means "cannot prove the queue drained". It does NOT mean "write
    nothing": `flush_bundles` still verifies every frame, and a bundle whose
    frames are all verifiably complete is recorded whether or not this succeeded.
    A drain failure only lowers coverage -- it can never admit an incomplete
    artifact, because the check, not the drain, decides that.
    """
    _resolve_hooks()
    if _FLUSH_HOOK is None:
        logger.warning(
            "recipe consult: the recipe-cache flush binding is unavailable, so "
            "the store queue cannot be drained. Bundles will be recorded only "
            "where every frame is already verifiably complete; the rest are "
            "refused and left for the next quiescent point.")
        return False
    try:
        _FLUSH_HOOK()
    except Exception as exc:  # noqa: BLE001 - a failed flush is not a proof
        logger.warning(
            "recipe consult: flushing the recipe cache failed (%s: %s). Bundles "
            "will be recorded only where every frame is already verifiably "
            "complete.", type(exc).__name__, exc)
        return False
    return True


def manifest_path(cache_dir: str, region_key: str) -> str:
    return os.path.join(cache_dir, region_key + _MANIFEST_SUFFIX)


def read_manifest(cache_dir: str, region_key: str):
    """`(frames, bindings)` for this region, or None if never traced.

    `bindings` is aligned with `frames` and holds each frame's ordered
    placeholder signatures, or None for a frame whose binding was not recorded.
    IT IS NONE FOR A WHOLE MANIFEST WRITTEN BEFORE BINDINGS EXISTED, and that is
    deliberate: an old manifest cannot say which argument belongs in which input,
    so the consult must treat it as "binding unverifiable" and MISS. Reading such
    a manifest as a hit is exactly the silent-corruption path this field closes.

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
    # ADDITIVE, AND ABSENCE IS NOT AN ERROR. A manifest written before this field
    # existed parses fine and yields None -- "binding unverifiable" -- which the
    # consult turns into a MISS. A manifest whose field is present but malformed
    # is treated the same way for the same reason: refusing to launch is always
    # available and never wrong, while launching a guess corrupts silently.
    bindings = data.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(frames):
        bindings = None
    return frames, bindings


def write_manifest(cache_dir: str, region_key: str, frames: list[str],
                   bindings: list | None = None) -> None:
    """Record the ordered frame keys one trace of this region produced.

    Written after the trace, because the frame keys only exist then. This is the
    one thing that is legitimately a product of tracing, and it is metadata
    ABOUT the bundle, not an input to the key.

    PRECONDITION: every frame key passed here must already have a COMPLETE
    artifact pair on disk. A manifest is an assertion that the bundle is
    loadable, so writing one over an artifact the async store has not finished
    is a false assertion -- and the next boot believes it and hands the
    half-written file to `synRecipeDeSerialize`. `flush_bundles` establishes
    that precondition; do not call this directly from anywhere else.

    The write itself is atomic (tmp + `os.replace`), so a reader never sees a
    partial manifest -- only a complete one, or none.
    """
    path = manifest_path(cache_dir, region_key)
    tmp = path + ".tmp"
    # ADDITIVE SCHEMA CHANGE. `region_key` and `frames` keep their exact meaning
    # and position, so every manifest already on disk still parses; `bindings` is
    # the new field, aligned with `frames`, each entry the ordered placeholder
    # signatures of that frame (or null when this process could not obtain them).
    payload = {"region_key": region_key, "frames": frames}
    if bindings is not None:
        payload["bindings"] = [_plain(binding) for binding in bindings]
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
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


def _plain(value):
    """Tuples to lists, recursively.

    The manifest goes through JSON, which has no tuples, so a recorded binding
    comes back as lists while `_signature_of` produces tuples. Comparing the two
    without this would refuse every binding -- a check that cannot pass is as
    useless as one that cannot fail.
    """
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _call_signature(args: tuple) -> list:
    """The caller's arguments in the SAME normalisation the key uses.

    Deliberately `compile_inputs._signature_of` rather than a second signature
    function: the key was built from that function, so a binding check built from
    anything else would eventually disagree with the key about the same call.
    """
    from . import compile_inputs  # noqa: PLC0415

    out = []
    for index, arg in enumerate(args):
        out.extend(compile_inputs._signature_of(arg, f"arg[{index}]"))
    return out


def _binding_key(entries) -> list:
    """Reduce signature entries to what decides a BINDING: kind and shape.

    WHY DTYPE AND STRIDE ARE DROPPED, and this is not laxity:

      * The recorded entry is the FRAME's placeholder -- the graph's own input
        type AFTER the hoisted `aten._to_copy`. For exactly the regions this
        feature exists for, that is an HPU bf16 tensor.
      * The caller's argument is the PRE-hoist value: a CPU fp32 tensor.
      * So the dtype differs BY CONSTRUCTION, because the binding is right, not
        because it is wrong. A dtype-sensitive comparison would therefore refuse
        every hit we are trying to earn -- safe on paper, useless in practice,
        and it would make the feature look broken.
      * Stride likewise: `to(device, dtype)` may re-lay-out, so stride is a
        COERCION dimension, not a binding discriminator.

    The question this check exists to answer is "is argument i the same LOGICAL
    input as recipe input i". Kind and shape answer that. Dtype and stride answer
    "does this need coercion", which the C++ side now handles deliberately.

    SCALARS KEEP THEIR VALUE. An `int` has no shape and no coercion dimension --
    the value IS its identity, and offsets are the discriminator this project
    cares about -- so reducing it to bare ("int",) would weaken the check for no
    reason at all.
    """
    out = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry and entry[0] == "tensor":
            # (kind, shape) only. Shape is entry[1]; dtype/stride are entries 2/3.
            out.append(("tensor", _plain(entry[1]) if len(entry) > 1 else None))
        elif isinstance(entry, (list, tuple)):
            out.append(tuple(_plain(entry)))
        else:
            # A MALFORMED entry from a damaged manifest. Kept verbatim so it
            # cannot accidentally equal a well-formed one; it will fail the
            # comparison and the call will be refused, not crash.
            out.append(entry)
    return out


def _binding_mismatch(recorded, args: tuple) -> str | None:
    """NOT CONSULTED. Kept for offline analysis of the manifest's `bindings` field.

    DO NOT CALL THIS FROM THE LAUNCH PATH. It was tried there and removed: it
    compares the FRAME's placeholder list against the REGION's argument list, and
    those differ by construction (`_setitem_scalar`: 1 placeholder, 2 arguments),
    so it refused every call. The C++ side owns the binding check now. See the
    comment in `consult`.

    None if the recorded binding matches this call, else the reason it does not.

    THE POINT OF THE CHECK. `consult` passes `tuple(args)` positionally and the
    C++ `update_patching_table` binds strictly positionally. If recipe input i is
    not the caller's argument i, the launch puts the WRONG tensor in the right
    place -- silent corruption, not an error. An artifact already exists whose
    input is named `%arg1_1` where the region's first parameter is `dest`, so
    "the binding is the obvious one" is not an assumption we are entitled to.

    WHAT THIS CANNOT CATCH, STATED RATHER THAN GLOSSED: two arguments that are
    identical in kind and shape but differ in dtype, device or identity, passed
    in SWAPPED order, are indistinguishable to this check. That is a residual
    risk of the same class as two same-shaped tensors swapped, and it is not
    closable without knowing each placeholder's identity -- which is exactly what
    this manifest is approximating. What is closed is the case that actually
    occurred: a binding shifted by one, where the shapes differ.
    """
    if recorded is None:
        return ("unverifiable: no binding was recorded for this frame, so the "
                "recipe's input order cannot be checked against this call "
                "(a manifest written before bindings were recorded)")
    try:
        want = _plain(recorded)
        got = _plain(_call_signature(args))
    except Exception as exc:  # noqa: BLE001 - no signature means no verification
        # A refusal, not a crash. This runs on the hit path inside `consult`,
        # which must never raise into a region call: if the normalisation cannot
        # be computed, the binding is unverifiable and the safe direction is the
        # same as for an old manifest.
        return (f"unverifiable: cannot compute this call's signature "
                f"({type(exc).__name__}: {exc})")
    try:
        matched = _binding_key(want) == _binding_key(got)
    except Exception as exc:  # noqa: BLE001 - a malformed binding is unverifiable
        return (f"unverifiable: the recorded binding cannot be compared with "
                f"this call ({type(exc).__name__}: {exc})")
    if matched:
        return None
    for index in range(min(len(want), len(got))):
        if _binding_key([want[index]]) != _binding_key([got[index]]):
            # The FULL entries are printed, including dtype and stride: they are
            # not what was compared, but they are what makes the line diagnosable
            # when the difference turns out to be a real binding shift.
            return (f"argument index {index} differs (compared on kind and shape "
                    f"only): manifest {want[index]!r} vs call {got[index]!r}")
    return (f"arity differs: the manifest records {len(want)} signature "
            f"entr(ies) and this call supplies {len(got)}")


# ---------------------------------------------------------------------------
# THE PHASE GATE -- PRODUCT BEHAVIOUR, not a diagnostic.
#
# Trace-free hits are DISABLED until the harness says the weights are on the
# device. That is not caution; it is the one thing measured to work.
#
# THE EVIDENCE, AND WHY IT IS NOT ABOUT THE LOAD.
#
#   b1  VLLM_RECIPE_CONSULT_GATE=none -> RESULT=STARTUP_OK elapsed=119s
#       The consult returns MISS BEFORE reading the manifest, checking the
#       artifact or parsing any metadata. Zero filesystem access.
#   b0  gate unset -> dies in weight loading at line 1393. read_manifest +
#       artifact_complete + a metadata parse.
#   b2  gate unset, every hit refused before the load -> dies at the SAME line.
#       precompiled_load: 0. TWO filesystem touches and no deserialise at all.
#
# b0 vs b2 is the decisive pair: b2 removed the load entirely and the failure did
# not move, so `precompiled_load` is NOT the harmful step. The only difference
# left between the boot that works and both boots that fail is whether the
# consult touches the FILESYSTEM at all.
#
# A CONSULT CANNOT MAKE A PARAMETER `torch.Size([0])`. What it can do is yield --
# every Python file read releases the GIL -- at a point during model construction
# where weight loading has a latent ordering assumption. That makes it a
# PERTURBATION rather than a mechanism, and it is why "fewer reads" is not a fix:
# b2 did it twice and still died.
#
# WHAT THIS COSTS IS NOT CORRECTNESS. A MISS mints the permit and re-traces, which
# is what a cold boot does and what b1 proves is safe. The cost is the cross-boot
# saving for the weight-loading regions -- real, and quantified in the report
# that accompanied this change rather than hidden here.
#
# An explicit `VLLM_RECIPE_CONSULT_GATE` value OVERRIDES this gate, so b0 and b2
# stay reproducible for diagnosis. With the variable unset -- the product default
# -- the phase gate governs.
_PHASE_PRE_WEIGHT_LOAD = "pre-weight-load"
_PHASE_SERVING = "serving"

_phase = _PHASE_PRE_WEIGHT_LOAD
_phase_gated_count = 0
_phase_first_gated_logged = False


def phase() -> str:
    """`"pre-weight-load"` or `"serving"`. Read-only, for tests and reports."""
    return _phase


def phase_gated_count() -> int:
    """How many consults the phase gate turned into a MISS. Reported at the flip."""
    return _phase_gated_count


def enable_hits_after_weight_load(reason: str = "") -> None:
    """Open the phase gate. Idempotent.

    Call once, after the model's weights are on the device, before anything that
    should be allowed to hit. The flip is logged with the number of consults it
    cost, so a boot that never flips is visible in the log rather than merely
    silent -- an armed guard that looks armed when it is not is the failure mode
    this campaign keeps finding.
    """
    global _phase
    if _phase == _PHASE_SERVING:
        return
    _phase = _PHASE_SERVING
    logger.info(
        "region consult: PHASE -> serving%s. %d consult(s) were MISSed by the "
        "phase gate while the model was loading; trace-free hits are enabled from "
        "here on.", (f" ({reason})" if reason else ""), _phase_gated_count)


def _phase_gate_refuses(region_name: str, key: str) -> bool:
    """True if the phase gate turns this consult into a MISS.

    DOES NO FILESYSTEM ACCESS ITSELF, and that is the whole point: while this
    returns True the caller returns before `read_manifest`, before
    `artifact_complete` and before any metadata parse, so the pre-weight-load
    phase touches nothing beyond the key -- which is b1, the boot that works.

    Logs the FIRST refusal of the boot only. Logging every one would bury the log
    under a few hundred identical lines during model init; the total is reported
    when the gate opens.
    """
    global _phase_gated_count, _phase_first_gated_logged
    if _phase == _PHASE_SERVING:
        return False
    _phase_gated_count += 1
    if not _phase_first_gated_logged:
        _phase_first_gated_logged = True
        logger.info(
            "region consult: region=%s key=%s -> MISS (PHASE GATE: weights are not "
            "loaded yet, so the cache is not consulted at all -- no manifest read, "
            "no artifact check, no metadata parse). First of the boot; the total is "
            "reported when the gate opens.", region_name, key)
    return True


# ---------------------------------------------------------------------------
# THE HIT GATE -- a DIAGNOSTIC, default off.
#
# WHY IT EXISTS. With manifests on disk a boot dies during weight loading at
# `_hpu_default_weight_loader`'s size assertion (`param.size() == torch.Size([0])`);
# with the cache cleared it reaches startup. The correlation is measured and
# repeatable, and it does not say WHERE the harm is.
#
# THERE ARE THREE POSSIBILITIES, NOT TWO, and the middle one is why a simple
# on/off switch is not enough:
#
#   1. a SPECIFIC hit is wrong   -> fix that region's binding or body;
#   2. ANY hit during weight loading is harmful -> keep hits off during loading;
#   3. THE LOAD IS HARMFUL AND NO LAUNCH MATTERS.
#
# (3) was the hypothesis behind `load_only`, and it is now REFUTED: b2 refused
# every hit before the load (`precompiled_load: 0`) and the boot died at the same
# line. The mode is kept because it is the cheapest way to re-test that
# conclusion if the phase gate ever stops being enough.
#
# VALUES
#   ""              the phase gate governs (product behaviour).
#   "none"          no load, no launch -- bypasses the phase gate.
#   "load_only"     `precompiled_load` runs, the launch does not.
#   "<substr>,..."  allow hits only for regions whose name contains one of the
#                   substrings, e.g. "_copy_weight_flat,_copy_shard_into_shard".
#                   This is the (1) vs (2) discriminator.
#
# WHY A GATED BOOT IS A CORRECT BOOT, AND THEREFORE A SOUND COMPARISON. The gate
# does not skip the consult: the key is still computed, a MISS still mints a
# `TraceGate` permit, and the region re-traces exactly as it would on a cold
# cache. So a gated boot produces the same model as an ungated one that misses;
# the only thing removed is the hit. Without that property the experiment would
# be comparing two different failures rather than one failure and its absence.
_GATE = os.environ.get("VLLM_RECIPE_CONSULT_GATE", "")


def _gate_decision(region_name: str) -> tuple[bool, bool] | None:
    """(run_load, run_launch) from `VLLM_RECIPE_CONSULT_GATE`, or None if unset.

    `None` means "no diagnostic override in force, let the phase gate decide",
    which is the product default. A non-None answer BYPASSES the phase gate, so
    b0 and b2 stay reproducible.
    """
    gate = _GATE.strip()
    if not gate:
        return None
    if gate == "none":
        return False, False
    if gate == "load_only":
        return True, False
    tokens = [tok.strip() for tok in gate.split(",") if tok.strip()]
    if any(tok in region_name for tok in tokens):
        return True, True
    return False, False


# ---------------------------------------------------------------------------
# VALIDATE WHAT THE METADATA ALONE CAN PROVE, BEFORE THE LOAD.
# ---------------------------------------------------------------------------
#
# WHY BEFORE: `precompiled_load` is expensive and, on the evidence below, is not
# something to pay for a recipe that will then be refused for its binding. The
# ordering is still worth keeping -- it makes a refusal cost a metadata parse
# instead of a deserialise -- but the REASON ORIGINALLY RECORDED HERE WAS WRONG
# and is corrected rather than deleted:
#
#   WHAT WAS CLAIMED: "b0 refused both hits after `precompiled_load` and still
#   died, so THE LOAD ALONE IS HARMFUL."
#
#   WHAT b2 SHOWED: with every hit refused BEFORE the load
#   (`precompiled_load: 0`), the boot died at the SAME line, 1393, with the same
#   assertion. So removing the load entirely does not move the failure. The load
#   is not the harmful step, and b0-vs-b1 conflated the load with the whole read
#   path. What actually separates the boot that works (b1) from both boots that
#   fail (b0, b2) is whether the consult touches the FILESYSTEM at all -- which
#   is the PHASE GATE's business, not this validation's.
#
# So this pre-load validation is kept for cost and for message quality, NOT
# because it fixes the boot. Nothing here should be read as the reason the phase
# gate exists.
#
# The C++ checks stay exactly as they are: they are the backstop for anything
# only the recipe handle can see (device, dtype, the real patching table). This
# is the part that needs no handle, moved earlier.

#: Field codes from `RecipeValueSpec`, so a shape list is readable in a log.
_META_NONE = 2 ** 64 - 1

#: Parsed `(num_inputs, [input shapes])` per (cache_dir, frame_key). The artifact
#: is immutable once written, so this cannot go stale; without it every trace-free
#: HIT would re-read and re-parse the metadata, and hits happen per token.
_METADATA_CACHE: "dict[tuple[str, str], tuple[int, list]]" = {}
_METADATA_CACHE_LIMIT = 4096


class _MetaReader:
    """Just enough of `RecipeHolder`'s binary format to reach the input shapes.

    Field order taken from tensor_info.cpp:189-247 (PtTensorInfo) and
    hpu_habana_cache.cpp:539-573 (RecipeValueSpec). Self-contained on purpose:
    this must not depend on a tool outside the tree.
    """

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0:
            raise EOFError(f"negative length {n} at byte {self.pos}")
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise EOFError(f"recipe metadata truncated at byte {self.pos}")
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.take(1))[0]

    def boolean(self) -> bool:
        return self.u8() != 0

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.i32()).decode("utf-8", "replace")

    def vec_i64(self) -> list:
        n = self.i32()
        return list(struct.unpack("<%dq" % n, self.take(8 * n)))

    def vec_u8(self) -> list:
        return list(self.take(self.i32()))

    def topts(self) -> None:
        if self.boolean():
            self.take(2)                 # device (i8, i8)
        if self.boolean():
            self.take(1)                 # dtype
        if self.boolean():
            self.take(1)                 # layout
        if self.boolean():
            self.boolean()               # requires_grad
        if self.boolean():
            self.boolean()               # pinned
        if self.boolean():
            self.take(1)                 # memory_format


def read_recipe_inputs(path: str) -> tuple:
    """`(num_inputs, num_outputs, [shape, ...])` from a recipe `.metadata` file.

    Raises on anything it does not understand. The caller turns that into a
    refusal rather than a guess: a format this reader cannot follow is a reason
    not to load, because the load is the harmful step (b0 vs b1).
    """
    with open(path, "rb") as handle:
        reader = _MetaReader(handle.read())
    shapes = []
    for _ in range(reader.i32()):
        # PtTensorInfo, in field order.
        reader.boolean()                 # is_ZST
        reader.boolean()                 # is_view
        reader.boolean()                 # is_restrided
        reader.boolean()                 # is_allow_permutation
        reader.vec_u8()                  # perm
        reader.boolean()                 # dont_permute
        reader.u64()                     # offset
        reader.u64()                     # external_offset
        reader.string()                  # ir_name
        reader.string()                  # syn_name
        reader.u64()                     # numel
        reader.u64()                     # external_numel
        reader.u64()                     # size
        reader.boolean()                 # is_duplicate
        reader.u64()                     # parent_index
        reader.u64()                     # output_index
        shapes.append(tuple(reader.vec_i64()))
        reader.vec_i64()                 # strides
        reader.i8()                      # memory format
        reader.topts()
        reader.i32()                     # tensor_type
        reader.u64()                     # dma_tensor_idx
        reader.u64()                     # tensor_id
        reader.i32()                     # dma_gen_id
    reader.u64()                         # id
    reader.u64()                         # iter_idx
    num_inputs = reader.u64()
    # THE HEADER CONTINUES, and num_outputs is what says whether the launch
    # returns anything. Field order is RecipeArgumentSpec's, verified against the
    # serializer: id, iter_idx, num_inputs, num_induplicates, num_dma_inputs,
    # num_shape_tensors, num_intermediates, num_outputs, ...
    reader.u64()                         # num_induplicates
    reader.u64()                         # num_dma_inputs
    reader.u64()                         # num_shape_tensors
    reader.u64()                         # num_intermediates
    num_outputs = reader.u64()
    return num_inputs, num_outputs, shapes


def _recipe_inputs(cache_dir: str, frame_key: str) -> tuple:
    """Cached `read_recipe_inputs`."""
    cache_key = (cache_dir, frame_key)
    cached = _METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parsed = read_recipe_inputs(
        os.path.join(cache_dir, frame_key + _METADATA_SUFFIX))
    if len(_METADATA_CACHE) >= _METADATA_CACHE_LIMIT:
        _METADATA_CACHE.clear()
    _METADATA_CACHE[cache_key] = parsed
    return parsed


def _in_place_result(cache_dir: str, frame_key: str, inputs: tuple):
    """The tensor an IN-PLACE region returns, or None.

    A recipe with `num_outputs == 0` is an in-place op: the graph writes into its
    argument and produces no new tensor, so `graph_launch` hands back an empty
    stack. The eager function still returns the mutated tensor and callers write
    `t = _zero_attention_kv(t)`, so returning the empty stack makes `t` None.

    CONSERVATIVE ON PURPOSE: only when the recipe has no outputs AND the call has
    exactly ONE tensor slot, which is then unambiguously the in-place target. With
    two or more slots there is no way to know which one the region returns, and
    returning the wrong one would be silent corruption instead of a loud failure,
    so it returns None and the caller keeps the launch's own result.
    """
    try:
        _n_in, num_outputs, _shapes = _recipe_inputs(cache_dir, frame_key)
    except Exception:  # noqa: BLE001 - unreadable metadata: keep the launch result
        return None
    if num_outputs != 0:
        return None
    slots = _call_tensor_slots(inputs)
    return slots[0] if len(slots) == 1 else None


def _is_tensor_like(value) -> bool:
    """True for anything shaped like a torch tensor.

    DUCK-TYPED, not `isinstance(value, torch.Tensor)`, for two reasons: this runs
    on every trace-free HIT and importing torch here would put an import on that
    path, and the C++ side's `isTensor()` is about the IValue's kind rather than a
    Python class anyway. A false positive would need an object with a shape, a
    dtype and a stride() method that is not a tensor, which would be miscounted
    by `bind_region_inputs` too.
    """
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and callable(getattr(value, "stride", None)))


def _call_tensor_slots(inputs: tuple) -> list:
    """The caller's tensor slots, counted EXACTLY as `bind_region_inputs` does.

    A tensor consumes one slot; a list of tensors consumes `len`; anything else
    consumes none (hpu_habana_cache.cpp:2646). Counted the same way so the two
    cannot disagree about what "arity" means.
    """
    slots = []
    for value in inputs:
        if _is_tensor_like(value):
            slots.append(value)
        elif (isinstance(value, (list, tuple)) and value
              and all(_is_tensor_like(item) for item in value)):
            slots.extend(value)
    return slots


def _preload_refusal(cache_dir: str, frame_key: str, inputs: tuple):
    """Why this artifact cannot be launched with these arguments, or None.

    Only what the metadata can prove: arity, and each tensor's shape. Anything
    that needs the recipe handle is left to the C++ checks, which remain the
    backstop.
    """
    try:
        num_inputs, _num_outputs, shapes = _recipe_inputs(cache_dir, frame_key)
    except Exception as exc:  # noqa: BLE001 - unreadable is a reason to refuse
        return (f"its metadata could not be read ({type(exc).__name__}: {exc}), "
                f"so the input binding cannot be validated before loading")
    try:
        slots = _call_tensor_slots(inputs)
    except Exception as exc:  # noqa: BLE001
        return (f"the call's arguments could not be inspected "
                f"({type(exc).__name__}: {exc})")
    if len(slots) != num_inputs:
        return (f"arity: the artifact records {num_inputs} input tensor slot(s) "
                f"and this call supplies {len(slots)}")
    for index, slot in enumerate(slots):
        if index >= len(shapes):
            return (f"argument {index}: the artifact records no shape for it")
        recorded = tuple(shapes[index])
        supplied = tuple(int(dim) for dim in slot.shape)
        if recorded != supplied:
            return (f"argument {index}: the artifact records shape "
                    f"{list(recorded)} and this call supplies {list(supplied)}")
    return None


# ---------------------------------------------------------------------------
# CAPTURE -- record what a region call was ACTUALLY called with.
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS, MEASURED. The explicit warm mints the regions in
# `vllm_gaudi/v1/spec_decode/spec_launch_regions.py` at shapes it READS FROM THE
# CONFIGURATION, and boot-b8.log shows that this covers 21 of them and leaves the
# rest to compile at serving:
#
#   hpu_dflash2:_combine_aux_hidden_states   64 misses (recipe records 6 input
#                                            slots, the call supplies 7)
#   prefix_state_cache:_index_copy_rows      40 misses (recipe records one shape
#                                            list, the call supplies
#                                            [162,48,128,128] / [162,10,10240])
#   prefix_state_cache:_zeros_rows           10 misses (recipe records 0 slots,
#                                            the call supplies 1)
#   extension.ops:_dynamic_quant_naive        4 misses, never warmed
#   device_regions:_zeros_like                1 miss,  never warmed
#   spec_launch_regions:_scatter_draft_rows   2 misses at arities 7 and 14
#   model.forward                             1 miss,  never warmed
#   Qwen3NextDecoderLayer.forward             9 compiles at serving shapes
#
# Every one of those is a call the WARM does not make, at arguments the warm
# cannot know. The arity mismatches above are the same defect seen from the other
# side: the warm traced a recipe with the wrong number of inputs, so even the
# region it did warm could not be loaded at serving.
#
# The cure is not a better guess. It is to let a real serving run say what it
# called, and let the next boot's warm call exactly that.
#
# THE RECORD IS THE SIGNATURE THE KEY USES. `_signature_of` normalises one
# argument into the entries the key is built from, and that is what is written
# here -- so a call rebuilt from this file is INDISTINGUISHABLE from the recorded
# one: `make_record` over the rebuilt arguments produces the same `shapes` field
# and therefore the same key. A second normaliser here would eventually disagree
# with the key about the same call, and the disagreement would be a warm that
# looks like it worked.
#
# ONE LINE PER CALL, APPENDED ATOMICALLY:
#
#   {"region": "<name as the consult knows it>",
#    "args":   [<signature of arg 0>, <signature of arg 1>, ...],
#    "kinds":  ["scalar" | "list" | "dict" | "nested", ...],
#    "kwargs": {"<name>": <signature>},      # only when the call had keywords
#    "kwkinds": {"<name>": "<kind>"}}
#
# `args` is PER ARGUMENT, not the flat concatenation `make_record` builds: the
# argument BOUNDARY has to survive, or the replay cannot know whether a run of
# six tensor entries was one list of six or six separate arguments -- which is
# precisely `_combine_aux_hidden_states` (a list of six aux states plus one index
# tensor) versus `_index_copy_rows` (four separate tensors).
#
# `kinds` is what the flattening loses about the CONTAINER, recorded so the replay
# does not have to guess it back:
#   scalar -- not a container; exactly one signature entry
#   list   -- a flat list/tuple; one entry per element, rebuilt as a list
#   dict   -- a dict; `_signature_of` sorts by `repr(key)` and DROPS the keys, so
#             it cannot be rebuilt and the replay refuses it by name
#   nested -- a container inside a container; the flattening erases the grouping
#             and the replay refuses it by name rather than guessing
# The last two do not occur in this region set (every container argument is a flat
# list of tensors, or a flat tuple of ints for `torch.full`'s shape). They are
# recorded anyway, because "does not occur today" is not a property a reader can
# check, and a refusal that names its reason is worth more than a silent guess.
#
# COST WHEN OFF: one `is None` test on the consult path. Nothing is computed, no
# descriptor is opened, and the file this module writes is never created.

#: Where captured calls are appended, from `VLLM_WARM_CAPTURE`. Read ONCE, at
#: import, like every other ambient flag this module reads (`_GATE`).
_CAPTURE_PATH = os.environ.get("VLLM_WARM_CAPTURE", "").strip() or None

#: The append-only descriptor. Opened lazily on the first captured call and never
#: closed: a capture file is written by whichever thread calls a region, and for
#: weight loading that thread is not the one that would own a `with` block.
_CAPTURE_FD: "int | None" = None

#: Set when the capture file cannot be written. Capture is a diagnostic, so a
#: write fault must not kill the boot -- but it must not be silent either, or a
#: boot that recorded nothing looks exactly like one that recorded everything.
_CAPTURE_BROKEN = False

#: Serialises the lazy open and the single `write` per line. The lowering thread
#: and the calling thread differ, so this is required rather than decorative;
#: `O_APPEND` covers the cross-PROCESS case, which this lock cannot.
_CAPTURE_LOCK = threading.Lock()


def capture_path() -> "str | None":
    """The capture file this process appends to, or None. Read-only."""
    return _CAPTURE_PATH


def set_capture_path(path: "str | None") -> "str | None":
    """Point capture at `path` (None to stop capturing). Returns the previous path.

    The supported seam. The warm REPLAY uses it to switch capture off for the
    duration of a replay: a replayed call is not a serving call, so recording it
    would put the warm's own guesses into the file a later warm replays -- and
    would make the replay read the lines it was writing.
    """
    global _CAPTURE_PATH
    previous = _CAPTURE_PATH
    _CAPTURE_PATH = (path or None)
    return previous


def _container_kind(value) -> str:
    """What `_signature_of` did to this argument's container, as a word.

    Only what the REPLAY needs to know and cannot recover from the flattened
    entries: whether the argument was not a container, a flat one, or one whose
    structure the flattening destroyed. See the block comment above.
    """
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple, dict)):
                return "nested"
        return "list"
    return "scalar"


def _capture_off(reason: str) -> None:
    """Stop capturing, and say why. Never raises."""
    global _CAPTURE_BROKEN
    _CAPTURE_BROKEN = True
    logger.error(
        "recipe consult: CAPTURE DISABLED for this process (%s). The capture file "
        "%r will not be written again, so any warm replayed from it is INCOMPLETE. "
        "The region calls themselves are unaffected.", reason, _CAPTURE_PATH)


def _capture_call(region_name: str, args: tuple, kwargs: dict) -> None:
    """Append one line describing this call. Called ONLY when capturing.

    A fault here is logged and turns capture off; it never reaches the region
    call. That is the same rule the bundle recorder follows, for the same reason:
    a diagnostic must not be able to kill a boot.
    """
    global _CAPTURE_FD
    if _CAPTURE_BROKEN:
        return
    try:
        from . import compile_inputs  # noqa: PLC0415 - same lazy import the key uses

        record = {
            "region": region_name,
            # PER ARGUMENT. `_signature_of` returns a LIST of entries for one
            # value, so this is one normalised argument per element.
            "args": [compile_inputs._signature_of(value, f"args[{index}]")
                     for index, value in enumerate(args)],
            "kinds": [_container_kind(value) for value in args],
        }
        if kwargs:
            # Sorted by `repr`, which is the order `make_record` folds them in,
            # so the file says what the key did.
            order = sorted(kwargs, key=repr)
            record["kwargs"] = {
                key: compile_inputs._signature_of(kwargs[key], f"kwargs[{key!r}]")
                for key in order}
            record["kwkinds"] = {key: _container_kind(kwargs[key]) for key in order}
        line = json.dumps(record, separators=(",", ":")) + "\n"
    except Exception as exc:  # noqa: BLE001 - a capture fault must not be a region fault
        _capture_off(f"could not build the record for {region_name!r}: "
                     f"{type(exc).__name__}: {exc}")
        return

    data = line.encode("utf-8")
    try:
        with _CAPTURE_LOCK:
            descriptor = _CAPTURE_FD
            if descriptor is None:
                descriptor = os.open(_CAPTURE_PATH,
                                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                _CAPTURE_FD = descriptor
            # ONE write() FOR THE WHOLE LINE. With `O_APPEND` the offset update
            # and the write are a single operation, so concurrent writers -- the
            # lowering thread here, another process elsewhere -- cannot interleave
            # inside a line. A SHORT write would mean the line is already torn, so
            # it is an error rather than something to retry in pieces.
            written = os.write(descriptor, data)
    except Exception as exc:  # noqa: BLE001
        _capture_off(f"could not append to {_CAPTURE_PATH!r}: "
                     f"{type(exc).__name__}: {exc}")
        return
    if written != len(data):
        _capture_off(f"a short write to {_CAPTURE_PATH!r} ({written} of {len(data)} "
                     f"bytes) left a partial line in the file")


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

    # CAPTURE, AFTER THE KEY AND BEFORE ANYTHING ELSE. The key comes first so
    # that turning capture on cannot change what the key is; the capture comes
    # before every branch below so that a MISS the phase gate swallows is still
    # recorded -- during a real serving run almost every call is MISSed by the
    # phase gate while the model loads, and those are exactly the calls the warm
    # needs to replay.
    #
    # THE OFF PATH IS THIS ONE TEST. Nothing is imported, computed or opened when
    # `VLLM_WARM_CAPTURE` is unset.
    if _CAPTURE_PATH is not None:
        _capture_call(region_name, args, kwargs)

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

    # ---- THE PHASE GATE, BEFORE ANY FILESYSTEM ACCESS. --------------------
    #
    # This is the first branch after the key, and it must stay first: while the
    # phase gate is closed this consult touches NOTHING on disk. No
    # `recipe_cache_dir()` (harmless, but it belongs after the gate so the
    # "no cache directory configured" warning is not emitted a few hundred times
    # during model init), no `read_manifest`, no `artifact_complete`, no metadata
    # parse. That is b1, the boot that reaches startup.
    #
    # AN EXPLICIT `VLLM_RECIPE_CONSULT_GATE` OVERRIDES THE PHASE GATE. It has to:
    # b0 and b2 were produced with an unset variable and must stay reproducible,
    # and the allowlist experiments are the only way to localise a specific bad
    # hit. So the override is read first and, when present, wins.
    _override = _gate_decision(region_name)
    if _override is not None:
        _run_load, _run_launch = _override
    elif _phase_gate_refuses(region_name, key):
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        return key, MISS
    else:
        _run_load, _run_launch = True, True

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

    manifest = read_manifest(cache_dir, key)

    if manifest is None:
        TraceGate.mint(region_name, key)
        logger.info("region consult: region=%s key=%s -> MISS (never traced)",
                    region_name, key)
        STATS["miss"] += 1
        return key, MISS

    frames, bindings = manifest

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

    # THERE IS DELIBERATELY NO BINDING CHECK HERE. DO NOT ADD ONE BACK.
    #
    # A Python-side comparison of the manifest's recorded binding against the
    # call's arguments was tried and REMOVED: it compared two things that are not
    # comparable. The manifest records the FRAME's placeholder signatures; the
    # region's CALL ARGUMENTS legitimately include values that were specialised
    # into the graph and are therefore not placeholders at all. `_setitem_scalar`
    # genuinely has ONE placeholder and TWO call arguments, so the lists differ
    # in length BY CONSTRUCTION and the check refused every call it saw --
    # boot-p5.log: 760 "REFUSED TO LAUNCH" lines, 472 of them arity mismatches
    # ("the manifest records 1 signature entry(ies) and this call supplies 2")
    # and ~288 unverifiable. Not one of them was a binding defect.
    #
    # There is no correct comparison available at this layer: the recipe's real
    # input list is neither the frame's placeholders nor the region's arguments.
    # It is the graph's inputs after AOTAutograd's projections, which include the
    # hoisted `aten._to_copy` that appears in neither list. Only the C++ side
    # holds the real binding (`rvs.dtensorinfos`), and it now checks shape, arity
    # and post-coercion device there, raising the named
    # `PrecompiledRegionBindingError`. THAT is the binding check. It arrives at
    # the `graph_launch` handler below as an ordinary exception, where it is
    # counted in `launch_failure` and logged by name.
    #
    # The manifest's `bindings` field and the helpers that read it are kept
    # (`_binding_key`, `_plain`, `_call_signature`, `_binding_mismatch`) because
    # the field costs nothing and may be useful for offline analysis. NOTHING on
    # the launch path consults them, and nothing may refuse because of them.
    #
    # RESIDUAL RISK, still open: a same-shape wrong-tensor binding is caught by
    # neither layer. The C++ check catches arity, shape and device; it cannot
    # tell two same-shaped tensors apart, and neither could the Python check.

    # THE DECISION WAS TAKEN ABOVE, BEFORE THE FIRST FILESYSTEM ACCESS, so that
    # the phase gate can return without touching anything. This block only acts
    # on a diagnostic override that said "do not load"; the phase gate has
    # already returned by now. Both are outside the try/except below, because a
    # fault in a gate must not be caught by the handler that exists to treat a
    # DAMAGED ARTIFACT as a miss.
    if not _run_load:
        # GATED OFF by `VLLM_RECIPE_CONSULT_GATE`. Everything above this line
        # still happened: the key was computed, the manifest was read, and the
        # artifact was verified complete. The permit is minted and the call is
        # counted as a MISS, so the region re-traces exactly as it would on a
        # cold cache and the boot stays correct.
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        logger.info(
            "region consult: region=%s key=%s -> MISS (hit gated off by "
            "VLLM_RECIPE_CONSULT_GATE=%r; frame %s neither loaded nor launched)",
            region_name, key, _GATE, frame_key)
        return key, MISS

    # VALIDATE BEFORE LOADING. See the block above `_preload_refusal` for the b0
    # vs b1 evidence: the LOAD is the harmful step, and `_init_normal` is refused
    # for arity on every boot, so it must never be loaded at all. Everything the
    # metadata can prove is checked here; anything that needs the recipe handle is
    # left to the C++ checks, which are unchanged and remain the backstop.
    _refusal = _preload_refusal(cache_dir, frame_key, inputs)
    if _refusal is not None:
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        logger.error(
            "region consult: region=%s key=%s -> MISS (REFUSED BEFORE LOAD: %s). "
            "Frame %s. The artifact was NOT loaded -- the load is the step that "
            "breaks weight loading (b0 vs b1), so a recipe that cannot be launched "
            "is refused from its metadata first, and the region re-traces.",
            region_name, key, _refusal, frame_key)
        return key, MISS

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
        # The same treatment applies to any NAMED exception the C++ side raises
        # from the load path: caught here like any other, counted in
        # `load_failure`, type name logged. Nothing is swallowed for being known.
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

    if not _run_launch:
        # load_only: the artifact WAS deserialized and added to GraphStorage; the
        # launch does not run. If a boot fails here but passes with the gate set
        # to "none", the harm is in the LOAD (deserialize + the unbounded growth
        # of `GraphStorage::m_storage_vec`, which retains one recipe per hit) and
        # not in any launch -- possibility (3) in the gate's comment.
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        logger.info(
            "region consult: region=%s key=%s -> MISS (recipe %d loaded for frame "
            "%s but the launch was skipped by VLLM_RECIPE_CONSULT_GATE=load_only)",
            region_name, key, recipe_id, frame_key)
        return key, MISS

    # A hit does NOT mint a permit: nothing may be traced on this path, and if
    # anything tries, the gate must fire. That is the enforcement, not an
    # oversight.
    outputs: list = []
    try:
        _result = graph_launch(recipe_id, inputs, outputs, region_name)
    except Exception as exc:  # noqa: BLE001 - a launch defect is a miss
        # THE LAUNCH IS THE LAST PLACE A CACHE CAN TURN INTO A DEAD BOOT.
        #
        # A loaded recipe can still fail to launch: a residual defect in the
        # artifact, or in the arguments we handed it. That is recoverable by
        # exactly the mechanism this module exists for -- trace it -- so it is a
        # MISS, not an abort. It is logged at ERROR with the region, key and
        # frame key, counted separately in `launch_failure` so a report can tell
        # "the cache is used" from "the cache is used and failing", and the
        # permit is minted so the trace that follows is legal.
        #
        # NOTE THE ACCOUNTING: `hit` is incremented only AFTER a launch returns,
        # so a failed launch counts as a miss and never as both.
        #
        # THIS INCLUDES THE C++ SIDE'S NAMED FAILURES. The coercion path raises a
        # specific exception for a post-coercion shape mismatch or a non-HPU
        # input; it is NOT special-cased into silence here. It arrives as an
        # ordinary exception, is counted in `launch_failure`, and its type name
        # and message are logged -- so a coercion defect is a number in the
        # report and a named line in the log, not a quiet fallback.
        TraceGate.mint(region_name, key)
        STATS["miss"] += 1
        STATS["launch_failure"] += 1
        logger.error(
            "region consult: region=%s key=%s -> MISS (recipe %d loaded for frame "
            "%s but the launch failed: %s: %s). Re-tracing; the artifact will be "
            "stored again. This is a launch defect, not a missing cache entry.",
            region_name, key, recipe_id, frame_key, type(exc).__name__, exc)
        return key, MISS
    logger.info("region consult: region=%s key=%s -> HIT (frame %s, recipe %d); "
                "no trace", region_name, key, frame_key, recipe_id)
    STATS["hit"] += 1
    # AN IN-PLACE REGION HAS NO RECIPE OUTPUTS, so the launch returns an empty
    # stack while the eager function returns the tensor it mutated. Callers write
    # `t = _zero_attention_kv(t)`, so handing back the empty stack made `t` None
    # and the next `.view()` failed -- that was the KV-cache crash, and the log
    # shows it: `_zero_attention_kv -> HIT`, then
    # `AttributeError: 'NoneType' object has no attribute 'view'` in
    # `_attention_kv_tensors`.
    _in_place = _in_place_result(cache_dir, frame_key, inputs)
    if _in_place is not None and not _result:
        _result = _in_place
    return key, _result


def _frame_bindings(frames: list[str]) -> list:
    """The recorded placeholder signatures for each frame, None where unknown.

    Read from `compile_inputs`, which captured them in `frame_record` -- the one
    place that holds both a frame's placeholders and its key. A frame with no
    recorded binding yields None, and its manifest will say so, which makes every
    later consult for it a MISS rather than an unverified launch.
    """
    try:
        from . import compile_inputs  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no helper means no binding, not a crash
        return [None] * len(frames)
    out = []
    for frame in frames:
        try:
            out.append(compile_inputs.frame_binding(frame))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def record_bundle(region_name: str, region_key: str, frames: list[str]) -> None:
    """Queue a traced bundle for recording. DOES NOT WRITE A MANIFEST.

    Writing here is what produced boot-p1.log. This runs on the calling thread,
    in `Region.__call__`'s `finally`, while the recipes these frames name are
    still being compiled on the asynchronous LOWERING thread -- the store job has
    not been created yet, so no flush can see it and no verification can pass.
    105 bundles were refused for artifacts that were complete minutes later.

    So the write is deferred to `flush_bundles`, which drains the pipeline first.
    The contract is unchanged and non-negotiable: a manifest must never exist for
    an artifact that is not completely written. Only the *moment* moves.

    Same-process consequence: until `flush_bundles` runs, a repeated call with
    identical arguments finds no manifest and takes a MISS. Nothing re-traces
    (the compiled callable is already built), so this costs a consult, not a
    compile. See `_PENDING`.
    """
    cache_dir = recipe_cache_dir()
    if not cache_dir or not region_key:
        return
    if not frames:
        return

    # NOTHING BELOW TOUCHES THE FILESYSTEM OR THE LOGGER. That is a hard
    # requirement, not a style preference: this runs in `Region.__call__`'s
    # `finally` on the thread that called the region, and for weight loading that
    # is the loader thread. File I/O and logging there perturb the loader (p6:
    # `Attempted to load weight ... into parameter (torch.Size([0]))`), and no
    # manifest is worth that. So: a dict insert, a counter, and at most an
    # Event.set() to wake the writer thread. Everything else belongs to the
    # writer.
    #
    # `_resolve_hooks()` used to be called here to warm the barrier callables
    # before exit; it is now called from `_flush_pending`, which never runs on
    # this thread.

    global _SINCE_FLUSH

    # THE BINDING TRAVELS WITH THE BUNDLE. It is captured now, while this
    # process still has the frame's placeholders; a later boot reading the
    # manifest has only what is written here. A pure dict pop -- no I/O.
    bindings = _frame_bindings(frames)

    pending_key = (cache_dir, region_key)
    wake = False
    with _PENDING_LOCK:
        # Only a NEW bundle counts toward the threshold: re-recording a region
        # that is already queued replaces it and adds no work to record.
        if pending_key not in _PENDING:
            _SINCE_FLUSH += 1
        _PENDING[pending_key] = (region_name, list(frames), bindings)
        # Reset under the lock, before the writer runs, so exactly one thread
        # crosses each threshold and a failed write cannot cause a retry storm.
        # A threshold of 0 means DISABLED: nothing is signalled and the writer
        # stays asleep for the whole boot.
        if (_BUNDLE_FLUSH_THRESHOLD > 0
                and _SINCE_FLUSH >= _BUNDLE_FLUSH_THRESHOLD):
            _SINCE_FLUSH = 0
            wake = True

    if wake:
        # The WHOLE of the boot-path work. `_wake_writer` starts the daemon if
        # needed and sets an Event; it does no I/O and cannot raise in a way that
        # matters (it is wrapped). The write itself happens on the writer thread.
        try:
            _wake_writer()
        except Exception:  # noqa: BLE001 - never kill a boot over a manifest
            logger.exception(
                "region consult: could not wake the manifest writer. The bundle "
                "stays queued and will be written at the next flush; recording is "
                "best-effort during a boot, never at the cost of the boot.")


def _settle_lowering_thread() -> bool:
    """Wait for the lowering thread to go idle. False if that cannot be proven.

    These regions compile on the LOWERING thread (`HabanaLaunchOpPipeline::
    LoweringTask` is asynchronous), and that is the thread which eventually calls
    `graph_compile` -> `DiskCache::Add` -> `RecipeCache::store`. Draining the
    store queue before that thread has run proves nothing about a job that does
    not exist yet, so this barrier has to come first.

    `torch.hpu.synchronize()` is the Python-reachable primitive.
    `_hpu_C.synchronize_device` (hpu/csrc/bindings.cpp:189) calls
    `HbLazyTensor::StepMarker()` and then
    `HPUDeviceContext::synchronize_host_multistage_pipeline()`, which in eager
    mode (`PT_HPU_LAZY_MODE=0` -- the mode these regions run in) is
    `JoinAllThreads()` -> `JoinPipelineThreads()` ->
    `lowering_thread()->waitWorkComplete()`, plus the compile pool, the execute
    thread and the garbage-collection thread. `waitWorkComplete` releases the
    GIL while it waits (`GILReleaseOnWaitPool`, HPUDevice.h:34), so this cannot
    deadlock against a lowering thread that needs the interpreter.

    Note the mode dependence: in lazy mode `synchronize_host_multistage_pipeline`
    is a no-op, so this returns True without having joined anything. That is not
    a correctness hole -- `flush_bundles` verifies every frame independently and
    refuses anything incomplete -- it only means fewer bundles are recordable.

    Returns False rather than raising. A failed drain does not make a complete
    artifact incomplete; the per-frame check, not this call, decides what may be
    written.
    """
    _resolve_hooks()
    if _SETTLE_HOOK is None:
        logger.warning(
            "region consult: torch.hpu.synchronize is unavailable, so the "
            "lowering thread cannot be settled. Every frame is still verified "
            "before any manifest is written; fewer bundles may be recordable.")
        return False
    try:
        _SETTLE_HOOK()
    except Exception as exc:  # noqa: BLE001 - a failed drain is not a proof
        logger.warning(
            "region consult: could not settle the lowering thread (%s: %s). "
            "Every frame will still be verified before any manifest is written, "
            "so nothing incomplete can be recorded; fewer bundles may be "
            "recordable at this point.", type(exc).__name__, exc)
        return False
    return True


def drain_pipeline() -> bool:
    """Settle the lowering thread, THEN the recipe-cache store queue.

    Two barriers for two stages, in this order:
      1. the lowering thread, which has not even QUEUED the store when a region
         call returns (see `_PENDING`);
      2. the recipe cache's async store thread, which writes the artifact.

    Flushing first and settling second would verify against artifacts whose
    store job had not been created -- the boot-p1.log bug, exactly. Both must
    hold for the answer to mean anything, so this is an AND, not an OR.
    """
    settled = _settle_lowering_thread()
    flushed = flush_recipe_cache()
    return settled and flushed


def flush_bundles(drain: bool = True) -> dict:
    """Record every queued bundle whose artifacts are complete. Idempotent.

    SYNCHRONOUS, and meant for the two quiescent callers only: the post-warmup
    call in `hpu_worker.py` and the `atexit` handler. Because it holds
    `_FLUSH_LOCK`, it waits for any in-progress writer pass and then processes
    whatever is left, so when it returns, every bundle queued before the call has
    been written or refused. That is the "drains deterministically" property the
    post-warmup hook depends on.

    NOT for the region-call path. `record_bundle` wakes the writer thread
    instead, precisely so that no filesystem work happens on the loader thread.
    """
    return _flush_pending(drain)


def _flush_pending(drain: bool = True) -> dict:
    """The actual work: swap the queue, verify every frame, write or refuse.

    Runs either on the writer thread (`drain=False`) or synchronously from
    `flush_bundles` (`drain` as given). `_FLUSH_LOCK` serialises the two.

    For each queued bundle this verifies EVERY frame with `artifact_complete` and
    writes the manifest atomically if and only if all of them are complete. A
    bundle with one incomplete frame is refused, nothing is written, and it goes
    back on the queue for a later retry.

    `drain` selects whether to settle the pipeline first, and the default is
    deliberately the safe-for-quiescence one:

      * `drain=True` (default) -- the two callers where a global barrier is safe
        and useful: the post-warmup call in `hpu_worker.py` and the `atexit`
        handler. The pipeline is quiescent there, so draining makes the first
        attempt likely to succeed.
      * `drain=False` -- the WRITER THREAD. NO barrier, ever: the writer is not
        the main thread, and `torch.hpu.synchronize()` there would be a
        wrong-thread global join. The barrier's value is only in making a first
        attempt likely to succeed, which is never worth that.

    Verification needs no barrier -- `artifact_complete` reads the filesystem, so
    a frame whose store has not landed is refused and retried, not mis-recorded.
    The barrier only improves how many bundles succeed on the FIRST attempt, and
    that is never worth breaking the boot for.

    Idempotent: the queue is swapped out under the lock before any work, so a
    second call with nothing new queued does nothing and returns zeroes.

    Returns `{"bundles", "written", "refused", "frames"}`.
    """
    with _FLUSH_LOCK:
        return _flush_pending_locked(drain)


def _flush_pending_locked(drain: bool) -> dict:
    """`_flush_pending`'s body, with `_FLUSH_LOCK` already held."""
    global _SINCE_FLUSH

    with _PENDING_LOCK:
        pending = list(_PENDING.items())
        _PENDING.clear()
        _SINCE_FLUSH = 0

    report = {"bundles": len(pending), "written": 0, "refused": 0, "frames": 0}
    if not pending:
        return report

    if drain:
        drained = drain_pipeline()
        drain_note = "drained" if drained else "drain attempted, unproven"
        incomplete_note = "were still incomplete after draining the pipeline"
    else:
        # No barrier on this path, so an incomplete frame means only "the store
        # has not landed yet" -- it must not be reported as a drained-then-
        # missing artifact, which would assert a cause this path cannot know.
        drain_note = "no drain on this path"
        incomplete_note = ("were still incomplete (no drain on this path; the "
                           "store may simply not have landed yet)")

    refused: list[tuple[tuple[str, str], tuple[str, list[str], list]]] = []
    for (cache_dir, region_key), (region_name, frames, bindings) in pending:
        incomplete = [frame for frame in frames
                      if not artifact_complete(cache_dir, frame)]
        if incomplete:
            report["refused"] += 1
            refused.append(((cache_dir, region_key),
                            (region_name, frames, bindings)))
            # WORDED TO ASSERT NOTHING IT CANNOT ESTABLISH. The previous message
            # said "the store failed or the artifact pair is damaged", which was
            # a guess dressed as a diagnosis -- in boot-p1.log every one of those
            # artifacts was complete shortly afterwards, because the barrier ran
            # before the lowering thread had queued the store. Even after a drain
            # that inference is not available (the drain may not have joined
            # anything, e.g. in lazy mode), so this reports the observation and
            # stops there.
            logger.error(
                "region consult: NOT recording the bundle for region=%s key=%s: "
                "%d of %d frame artifact(s) %s (%s). No manifest was written, so "
                "this bundle is not recorded yet and is retried at the next "
                "flush. That is the observation; it does not establish why the "
                "artifact was incomplete.",
                region_name, region_key, len(incomplete), len(frames),
                incomplete_note,
                ", ".join(incomplete[:5]) + (" ..." if len(incomplete) > 5 else ""))
            continue
        write_manifest(cache_dir, region_key, frames, bindings)
        report["written"] += 1
        report["frames"] += len(frames)

    if refused:
        # Retry at the next quiescent point (the exit handler is one). A refusal
        # is not a verdict on the bundle, only on this instant.
        with _PENDING_LOCK:
            for pending_key, value in refused:
                _PENDING.setdefault(pending_key, value)

    logger.info("region consult: bundle flush: %d considered, %d written, %d "
                "refused, %d frame(s) recorded (%s)",
                report["bundles"], report["written"], report["refused"],
                report["frames"], drain_note)
    return report


def _flush_bundles_at_exit() -> None:
    """Record whatever is still queued when the interpreter shuts down.

    Why exit is enough for a cross-boot cache: the manifests are only read by the
    NEXT process, so they need to be on disk before this one exits -- not before
    serving. A boot that exits normally therefore records everything it traced,
    even though nothing was recorded during the boot itself. A boot that is
    killed records nothing, which costs a re-trace next time and never a bad
    manifest; `flush_bundles` refuses rather than guesses.
    """
    try:
        # STOP AND JOIN THE WRITER FIRST. A pass in progress must finish before
        # the synchronous flush below, or the two would interleave on the same
        # queue; and a writer still running after this handler returns would be
        # killed mid-write by interpreter shutdown.
        _STOP.set()
        _WAKE.set()
        with _WRITER_LOCK:
            writer = _WRITER
        if writer is not None and writer.is_alive():
            writer.join()
        # Anything the writer left behind (a pass it skipped after seeing _STOP,
        # or work queued while it was finishing) is written here, synchronously,
        # so the exit path cannot lose a queued bundle.
        flush_bundles()
    except Exception:  # noqa: BLE001 - never raise out of an exit handler
        logger.exception("region consult: the bundle flush at exit failed")


atexit.register(_flush_bundles_at_exit)


# ---------------------------------------------------------------------------
# MAKE THE CONSULT OPAQUE TO DYNAMO. DO NOT REMOVE THIS.
# ---------------------------------------------------------------------------
#
# THE BUG THIS FIXES. `consult` runs INSIDE dynamo traces: a region called from
# a traced model frame calls straight into this function, and its Python -- arg
# hashing, the key-string comparison, the manifest read, the thread-local
# permit bookkeeping in `trace_gate` -- is visible to the tracer. Dynamo cannot
# trace that, so it GRAPH-BREAKS the enclosing frame, splits it, and the
# remainder becomes a new frame with its own `graph_<id>`. That new graph is
# compiled for the first time on the SERVING thread (warmup is over), where it
# can be rejected -- synStatus 26 -- and the boot dies.
#
# MEASURED, and the shape of the evidence is unmistakable: a boot that served
# 6/6 had ZERO dynamo frames inside our own modules; the failing boot had 15,
# named `torch_dynamo_resume_in_consult_at_771`, `..._at_729`, `consult` (:695),
# `_consult_and_dispatch` (compile_gate.py:189) and `__call__`
# (compile_gate.py:275). `torch_dynamo_resume_in_*` is dynamo's name for the
# second half of a frame it split. Its callers in `extension/ops.py` became
# frames of their own at the same time, which they were not before.
#
# THE FIX. `torch._dynamo.disable` makes the tracer treat the whole call as ONE
# opaque call: no frame is created for it, nothing inside it is traced, and the
# value it returns flows into the surrounding graph as a graph input. Measured
# on this stack with a body shaped like this function (dict work, a thread-local
# mutation, a file-existence check):
#
#     undecorated: graph_breaks=1 graphs=2 frames_converted=5
#     decorated:   graph_breaks=0 graphs=1 frames_converted=2
#
# `recursive=False` IS DELIBERATE AND LOAD-BEARING. The default is recursive,
# which would suppress tracing for everything called from here too. `consult`
# calls no compiled callable itself, but `_consult_and_dispatch` does, and a
# recursive disable there would stop the region being traced at all -- Rule 12:
# hook into `@torch.compile`, do not eliminate it. Measured both ways: the
# nested compiled callable still traced with recursive=True on this stack, so
# recursive=False is belt-and-braces rather than a workaround.
#
# The permit is unaffected: it is minted by the miss branch below, before the
# caller traces anything, so the gate still holds. Being opaque to dynamo
# changes what the TRACER sees, not what the GATE requires.


#: Set once `consult` has been made opaque.
_CONSULT_OPAQUE = False


def make_consult_opaque() -> bool:
    """Make `consult` opaque to dynamo. Idempotent; safe to call repeatedly.

    CALLED FROM `compile_gate._apply_dynamo_opacity`, NOT AT IMPORT TIME, and
    that is deliberate: this module deliberately imports no torch, and it is
    imported by `explicit_warm` and `device_regions` on paths where dragging
    torch in at module scope would be a new import edge (and, on a Habana build,
    would load the device stack). Applying it from the module that already owns
    `torch.compile` keeps that property.

    It is also defence in depth: every caller of `consult` is already inside an
    opaque entry point (`Region.__call__`, `_consult_and_dispatch`), so this
    covers only a future caller that is not.

    Returns False when torch._dynamo is unavailable; the caller logs.
    """
    global consult, _CONSULT_OPAQUE
    if _CONSULT_OPAQUE:
        return True
    wrapped = _make_opaque(consult)
    if wrapped is consult:
        return False
    consult = wrapped
    _CONSULT_OPAQUE = True
    return True


def _make_opaque(fn):
    """Wrap `fn` so dynamo treats it as one opaque call instead of tracing it.

    Returns `fn` unchanged if `torch._dynamo` cannot be imported, and says so at
    ERROR: a silent no-op here would restore the graph-break bug with nothing in
    the log to explain the synStatus 26 that follows.
    """
    try:
        import torch._dynamo as _dynamo  # noqa: PLC0415 - torch is up by now
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "recipe consult: torch._dynamo is unavailable (%s: %s), so the "
            "consult CANNOT be made opaque to dynamo. A region called from "
            "inside a traced frame will graph-break it, and the split frame can "
            "be compiled for the first time on the serving thread. This is the "
            "synStatus 26 path.", type(exc).__name__, exc)
        return fn
    try:
        return _dynamo.disable(fn, recursive=False)
    except TypeError:
        # An older signature without `recursive`; the default is recursive, which
        # is still correct for THIS function (it calls nothing compiled).
        return _dynamo.disable(fn)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "recipe consult: torch._dynamo.disable failed on consult (%s: %s); "
            "the consult stays traceable.", type(exc).__name__, exc)
        return fn


# NOT applied here on purpose -- see `make_consult_opaque`.
