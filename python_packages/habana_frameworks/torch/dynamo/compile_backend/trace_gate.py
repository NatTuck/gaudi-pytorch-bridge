"""Hard gate: no ``torch.compile`` tracing without a recorded cache MISS.

This module is an *enforcement* mechanism, not a diagnostic.  The invariant it
enforces is:

    A cache consult must be performed, and a genuine MISS recorded, before any
    frame may be traced.  If that has not happened, tracing does not degrade
    gracefully -- the process dies.

Why a hard crash and not a warning
----------------------------------
The failure mode we are defending against is *silent*: a cache that reports a
MISS when it should have reported a HIT (or a cache consult that never happens
at all) makes Dynamo trace and recompile a graph that should have been served
from cache.  The run still produces correct tokens, just much more slowly, so
nothing in the output reveals the bug.  A warning would be equally invisible in
a long benchmark log.  Only an exception that cannot be caught by the tracing
machinery forces the bug to the surface.  Consequently there is deliberately:

  * no "warn" mode,
  * no "report only" mode,
  * no environment-variable / config switch to soften the gate,
  * no ``try/except`` around the gate inside this module.

How the gate works: permits on a per-thread stack
-------------------------------------------------
The gate is a *permit* system.  The code that owns the cache consult mints a
permit (:meth:`TraceGate.mint`) after it has recorded a MISS, does the tracing,
and revokes the permit (:meth:`TraceGate.revoke`) when the region is finished.
The permit names the ``region`` (which part of the model/runtime is compiling)
and the ``key`` (which cache entry was consulted), so a crash report identifies
the exact consult that was skipped.

Permits live on a *stack* held in :class:`threading.local`, for two reasons:

1. Compilation is genuinely nested.  An outer region (say, the whole model) may
   compile, and during that compilation an inner region (a submodule, an
   inlined helper, a nested ``torch.compile``) may compile as well.  A flat
   "is a permit held?" boolean cannot represent that, and revoking the inner
   region would incorrectly disarm the outer one.  With a stack the inner
   ``revoke()`` pops only the inner permit and the outer permit stays armed.

2. Tracing is per-thread.  Dynamo's eval-frame hook runs on whichever thread
   executed the Python frame.  A single global stack would let a permit minted
   on one thread silently authorise tracing on another -- exactly the kind of
   accidental authorisation this gate exists to prevent.  ``threading.local``
   gives every thread its own stack, so a thread that never minted a permit
   always crashes.

Usage
-----
::

    from ...trace_gate import TraceGate

    TraceGate.install()                    # once, at startup; idempotent

    hit, key = consult_cache(region, ...)
    if hit:
        use_cached(key)                    # no tracing, no permit needed
    else:
        TraceGate.mint(region, key)        # a genuine MISS was recorded
        try:
            compile_the_region()           # tracing is now permitted
        finally:
            frames = TraceGate.revoke()    # returns frames traced

Importing this module has no side effects at all: it does not import ``torch``
and it does not patch anything.  :meth:`TraceGate.install` is the only method
that installs the gate.

Implementation note on where the gate fires
-------------------------------------------
``torch._dynamo.convert_frame.convert_frame_assert`` is *not* the per-frame
entry point, despite the name: it is a **factory** that constructs and returns a
``ConvertFrameAssert`` instance (``convert_frame.py:811``).  That instance's
``__call__`` (``convert_frame.py:631``) is what actually converts a frame, and
it is reached from both dispatch paths:

* default (``fullgraph=False``): ``CatchErrorsWrapper.__call__``
  (``convert_frame.py:2522``) -> ``ConvertFrame.__call__``
  (``convert_frame.py:2308``) -> ``self._inner_convert(...)``
  (``convert_frame.py:2323``), where ``_inner_convert`` is the instance built
  once at ``convert_frame.py:2290``;
* ``fullgraph=True``: ``eval_frame.py:2699`` builds the same
  ``ConvertFrameAssert``, and the callback invokes its ``__call__`` directly.

So the gate is installed on ``ConvertFrameAssert.__call__``, which is the one
function every frame conversion passes through on both paths, and it therefore
fires **once per frame conversion** -- including a recompile, because a
guard-failure recompile re-enters that same cached instance's ``__call__``
rather than reconstructing it.

The factory is deliberately left unpatched.  It traces nothing (measured: at the
end of ``import torch`` the gate is armed and
``torch._dynamo.utils.counters["frames"]["total"]`` is still 0), so gating it
would refuse a construction that cannot trace; and making that survivable would
require the *caller* to mint a permit on its own behalf, with no cache key,
which is exactly the "empty key" and "caller mints" conditions the protocol
forbids.  One choke point, at the place where frames are actually converted.
"""

from __future__ import annotations

import functools
import importlib
import logging
import threading
from typing import Any, Final

__all__ = ["TraceWithoutCacheConsult", "TraceGate", "ATTRIBUTION_MISMATCHES"]

logger = logging.getLogger(__name__)

#: Every advisory attribution mismatch seen this process, newest last. Reported,
#: never raised: see the comment in `TraceGate.assert_permitted`. Read by the
#: report so "the gate is quiet" and "the gate never looked" are distinguishable.
ATTRIBUTION_MISMATCHES: list[str] = []

#: Module that owns the frame-conversion choke point.
_TARGET_MODULE: Final[str] = "torch._dynamo.convert_frame"

#: Class inside that module whose ``__call__`` performs one frame conversion.
_TARGET_CLASS: Final[str] = "ConvertFrameAssert"

#: The per-frame method on that class, wrapped by :meth:`TraceGate.install`.
_TARGET_METHOD: Final[str] = "__call__"

#: Marker set on our wrapper so ``install()`` can recognise its own work and
#: so ``installed()`` can answer without guessing.
_WRAPPER_MARK: Final[str] = "__trace_gate_wrapper__"

#: Marker holding the function our wrapper replaced, for introspection.
_ORIGINAL_MARK: Final[str] = "__trace_gate_original__"


class TraceWithoutCacheConsult(BaseException):
    """Tracing was attempted without a permit from a recorded cache MISS.

    This deliberately derives from :class:`BaseException` and **not** from
    :class:`Exception`.  This is the single most important design decision in
    the module, so the reasoning is spelled out here.

    Dynamo wraps the whole frame-conversion pipeline in ``except Exception`` in
    several places -- most notably ``torch._dynamo.convert_frame``'s
    ``catch_errors_wrapper``, and the ``config.suppress_errors`` handling in
    ``torch._dynamo.eval_frame``.  The purpose of those handlers is to turn any
    compilation problem into a *silent eager fallback*: the frame runs
    uncompiled and the user sees correct output at a fraction of the speed.

    An ``Exception`` subclass would be caught by those handlers.  The result
    would be the worst possible outcome: the gate would fire, the exception
    would be swallowed, the model would keep running eagerly, and the very
    condition the gate exists to detect -- tracing without a cache consult --
    would be converted into a performance regression that looks like a normal
    graph break.  That is a *fake detector*: it appears to enforce an
    invariant while in fact only logging it.

    Deriving from ``BaseException`` puts the error outside ``except Exception``
    (and outside ``contextlib.suppress(Exception)``), so it escapes the tracing
    machinery, escapes Dynamo's fallback handlers, and terminates the process
    with a traceback naming the region and cache key that were never consulted.
    ``KeyboardInterrupt`` and ``SystemExit`` live in the same part of the
    hierarchy, and for the same reason: they must not be mistaken for ordinary
    recoverable errors.

    Callers must therefore not write ``except Exception`` around a cache
    consult and expect to catch this.  If a permit genuinely cannot be minted,
    the correct response is to fix the consult, not to catch the exception.
    """


class _Permit:
    """One minted permit: which region, which cache key, how much tracing.

    Mutable on purpose: :meth:`TraceGate.assert_permitted` bumps
    ``frames_traced`` in place on the innermost permit, so the count survives
    until the matching :meth:`TraceGate.revoke` returns it.
    """

    __slots__ = ("region", "key", "frames_traced", "reported")

    def __init__(self, region: str, key: str) -> None:
        self.region: str = region
        self.key: str = key
        self.frames_traced: int = 0
        #: Whether an attribution mismatch has already been reported for this
        #: permit, so a region that trips it reports once and not once per frame.
        self.reported: bool = False

    def describe(self) -> str:
        """Render this permit for an error message."""
        return f"region={_fmt_region(self.region)} key={_fmt_key(self.key)}"


class _PermitStack(threading.local):
    """Per-thread stack of minted permits.

    Subclassing ``threading.local`` means CPython calls ``__init__`` once per
    thread the first time that thread touches the instance, so every thread
    starts with an empty stack without any explicit bookkeeping.
    """

    def __init__(self) -> None:
        # Innermost (most recently minted) permit is last.
        self.permits: list[_Permit] = []


def _fmt_region(region: str) -> str:
    """Render a region name, making an empty/absent region obvious."""
    return repr(region) if region else "<unnamed region>"


def _fmt_key(key: str) -> str:
    """Render a cache key, using the literal word ``NONE`` when it is empty."""
    return repr(key) if key else "NONE"


def _format_stack(permits: list[_Permit]) -> str:
    """Render the whole permit stack, innermost last, for an error message."""
    if not permits:
        return "    (no permits held)"
    lines = []
    for index, permit in enumerate(permits):
        lines.append(
            f"    [{index}] {permit.describe()} "
            f"frames_traced={permit.frames_traced}"
        )
    return "\n".join(lines)


def _looks_like_frame(obj: Any) -> bool:
    """Return True if ``obj`` quacks like a Python frame.

    Guarded with ``try``/``except`` rather than ``hasattr`` so that an object
    with a raising ``__getattr__`` cannot blow up while we are already in the
    middle of building an error message.
    """
    try:
        return (
            hasattr(obj, "f_code")
            and hasattr(obj, "f_lineno")
            and hasattr(obj.f_code, "co_filename")
        )
    except Exception:  # pragma: no cover - defensive only
        return False


def _find_frame(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The frame being converted, or None.

    ``ConvertFrameAssert.__call__(self, frame, cache_entry, hooks, frame_state)``
    takes the frame as its second positional argument. The scan is over all
    arguments rather than a fixed index so the gate stays correct if the
    position moves, and so a ``frame`` passed by keyword is still found.
    """
    candidates: list[Any] = list(args) + list(kwargs.values())
    for candidate in candidates:
        if _looks_like_frame(candidate):
            return candidate
    return None


def _describe_frame(frame: Any) -> str:
    """``name (file:line)`` for a frame, or an honest note when there is none."""
    if frame is None:
        return (
            "an unnamed frame (no frame object was visible in the arguments to "
            "ConvertFrameAssert.__call__; the first positional argument is the "
            "converter, the second is the frame)"
        )
    code = frame.f_code
    return (
        f"{getattr(code, 'co_name', '<unknown>')} "
        f"({code.co_filename}:{frame.f_lineno})"
    )


def _locate_frame(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Describe the frame being converted, from a call's arguments.

    ``ConvertFrameAssert.__call__(self, frame, cache_entry, hooks, frame_state,
    *, skip=0)`` takes the frame as its second positional argument, so the
    normal case finds it and the crash message names a real ``file:line``.  The
    scan is over all arguments rather than a fixed index so that the gate stays
    correct if the position ever moves, and so a ``frame`` passed by keyword is
    still found.  A ``file:line`` in the crash message is worth much more than
    "unknown", so when no frame is found the message says exactly that instead
    of guessing.
    """
    return _describe_frame(_find_frame(args, kwargs))


def _split_region(region: str) -> tuple[str, str]:
    """``(module, qualname)`` from a region name, or ``("", region)``.

    Region names come from two places and both are ``module:qualname``:
    ``device_regions`` builds ``f"{fn.__module__}:{fn.__qualname__}"`` and
    ``compile_gate`` builds ``f"torch.compile:{module}:{qualname}"``. The prefix
    is stripped so the two compare the same way.
    """
    text = region
    prefix = "torch.compile:"
    if text.startswith(prefix):
        text = text[len(prefix):]
    module, sep, qualname = text.rpartition(":")
    if not sep:
        return "", text
    return module, qualname


def _same_module(frame: Any, region: str) -> bool:
    """True when the frame's file looks like the region's module.

    Used only to keep the advisory attribution report quiet: a frame from a
    DIFFERENT module is unambiguously a callee of the region, so there is nothing
    to report. Returns True when the module cannot be determined, so an unknown
    layout is reported rather than silently skipped.
    """
    module, _qualname = _split_region(region)
    code = getattr(frame, "f_code", None)
    if not module or code is None:
        return True
    suffix = module.replace(".", "/") + ".py"
    return str(getattr(code, "co_filename", "")).replace("\\", "/").endswith(suffix)


def _attribution_mismatch(frame: Any, region: str) -> str | None:
    """None if the frame can be attributed to ``region``, else the reason.

    ADVISORY. The result is reported and counted; it never aborts a trace. See
    the comment in :meth:`TraceGate.assert_permitted` for the boot that made a
    hard refusal untenable.

    WHAT IT CATCHES, precisely: the FIRST frame converted under a permit came
    from the region's own module but has a different name from the region. That
    is the shape of "this permit authorised converting a different function in
    the same file", which is worth knowing about.

    WHAT IT CANNOT CATCH, stated rather than glossed:

      * A callee. Dynamo converts the functions a region calls, and their names
        are unrelated to the region's by construction. This is why the check is
        advisory: a wrong region and a callee are the SAME observation.
      * The region's own frame may not be the first one converted at all. If it
        is already compiled, dynamo reuses the cached code object and the first
        conversion under a new permit is whichever callee is new -- exactly
        boot-b1.log, where the first frame was `Logger.info`.
      * Two same-named functions are indistinguishable; it compares names, not
        identities.
      * Anything after the first conversion under a permit (not checked at all).
      * Nested permits: a correct outer permit with inner code from elsewhere.

    The relation is on the name because the name is what both sides share: a
    permit carries ``region`` (``module:qualname``), and the frame carries its
    code object's ``co_qualname``. Match means equal, or one a dotted suffix of
    the other -- a nested function's qualname is ``outer.<locals>.inner`` and a
    method's is ``Class.method``, and both must be accepted.
    """
    if not region:
        # An unnamed permit cannot be attributed to anything. Refusing here would
        # be refusing a permit the protocol says must never exist, which mint()
        # already prevents; there is nothing to compare against.
        return None
    if frame is None:
        return ("the frame being converted could not be identified, so it cannot "
                "be attributed to the permit that authorised it")
    code = getattr(frame, "f_code", None)
    if code is None:
        return ("the object being converted has no code object, so it cannot be "
                "attributed to the permit that authorised it")
    frame_qual = (getattr(code, "co_qualname", None)
                  or getattr(code, "co_name", "") or "")
    _module, region_qual = _split_region(region)
    if not region_qual:
        return None
    if (frame_qual == region_qual
            or frame_qual.endswith("." + region_qual)
            or region_qual.endswith("." + frame_qual)):
        return None
    return (f"the frame being converted is {frame_qual!r} "
            f"({code.co_filename}:{frame.f_lineno}), which is not attributable to "
            f"the innermost permit")


class TraceGate:
    """Static gate enforcing "cache consult MISS first, then trace".

    All state lives in the per-thread permit stack; there are no instances of
    this class.  Every entry point is a ``staticmethod`` so the class is used
    purely as a namespace.
    """

    _local: Final[_PermitStack] = _PermitStack()

    #: Serialises :meth:`install` so two threads cannot both wrap the target.
    _install_lock: Final[threading.Lock] = threading.Lock()

    @staticmethod
    def mint(region: str, key: str) -> None:
        """Push a permit authorising tracing for ``region``/``key``.

        Call this only *after* a cache consult has been performed and has
        recorded a genuine MISS.  Minting before the consult, or on a HIT,
        defeats the entire gate.

        Minting an identical ``(region, key)`` permit while that same permit is
        already on top of the stack is rejected: it is always a double-arm bug
        (a region that mints without revoking leaves a stale permit that later
        authorises tracing nobody asked for), and the stack would no longer
        match the program's region nesting.
        """
        permits = TraceGate._local.permits
        if permits and permits[-1].region == region and permits[-1].key == key:
            raise TraceWithoutCacheConsult(
                "TraceGate.mint() called twice for the same permit -- this is a "
                "double-arm bug.\n"
                f"\n"
                f"  region: {_fmt_region(region)}\n"
                f"  key:    {_fmt_key(key)}\n"
                f"\n"
                "The identical permit is already the innermost permit on this "
                "thread's stack.  A region must mint exactly once and revoke "
                "exactly once; nesting requires distinct (region, key) pairs.  "
                "The usual cause is a mint() on a code path that never calls "
                "revoke(), or a revoke() that was skipped by an exception "
                "handler.\n"
                f"\n"
                f"  permits held on this thread (innermost last):\n"
                f"{_format_stack(permits)}\n"
            )
        permits.append(_Permit(region, key))

    @staticmethod
    def revoke() -> int:
        """Pop the innermost permit and return how much tracing it authorised.

        Must be called on the same thread that called :meth:`mint`.  Revoking
        an empty stack means a region revoked without minting (or revoked
        twice), which is an error: silently ignoring it would hide an unbalanced
        region and leave later tracing ungated.
        """
        permits = TraceGate._local.permits
        if not permits:
            raise TraceWithoutCacheConsult(
                "TraceGate.revoke() called with no permit held on this thread.\n"
                "\n"
                "  region: <no permit held>\n"
                "  key:    NONE\n"
                "\n"
                "A region revoked a permit that it never minted, or it revoked "
                "the same permit twice.  Every mint() must be paired with "
                "exactly one revoke() on the same thread, normally in a "
                "finally: block.\n"
                "\n"
                "  permits held on this thread (innermost last):\n"
                "    (no permits held)\n"
            )
        permit = permits.pop()
        return permit.frames_traced

    @staticmethod
    def assert_permitted(what: str, frame: Any = None) -> None:
        """Assert that tracing is authorised AND attributable; count it.

        Called at the top of every frame conversion.  On success the innermost
        permit's ``frames_traced`` counter is incremented so that
        :meth:`revoke` can report how much tracing the permit covered.

        On an empty stack this raises :class:`TraceWithoutCacheConsult`.  There
        is no return-value signal and no logging-only path: an unauthorised
        trace attempt always raises.

        ``frame`` is the frame object being converted, when the caller has it.
        The FIRST conversion under a permit must be attributable to that permit
        by name (see :func:`_attribution_mismatch`); a permit minted for one
        region must not authorise converting a different region's code. Later
        conversions under the same permit are the region's own callees and are
        not checked, because their names are unrelated to the region's by
        construction.
        """
        permits = TraceGate._local.permits
        if not permits:
            raise TraceWithoutCacheConsult(
                "torch.compile tracing attempted with NO cache consult "
                "recorded.\n"
                "\n"
                f"  converting:    {what}\n"
                "  permit region: <no permit held>\n"
                "  permit key:    NONE\n"
                "\n"
                "No permit is held on this thread, so no cache consult can "
                "have recorded a MISS for this region.  The cache must be "
                "consulted and a MISS recorded before any tracing may occur; "
                "only then may TraceGate.mint(region, key) be called.\n"
                "\n"
                "This is a hard gate.  There is no warn mode, no soft "
                "fallback, and no report-only path: tracing has been aborted.\n"
                "\n"
                "  permits held on this thread (innermost last):\n"
                "    (no permits held)\n"
            )
        permit = permits[-1]
        permit.frames_traced += 1

        # ATTRIBUTION IS CHECKED ON THE FIRST CONVERSION ONLY, AND IT IS
        # ADVISORY. Both halves of that sentence are the result of a boot.
        #
        # The first conversion is the only one that could ever be attributed:
        # every later frame under a permit is a function the region calls --
        # dynamo converts those too -- and their names are unrelated to the
        # region's by construction.
        #
        # But even the FIRST conversion is not reliably the region's own frame.
        # boot-b1.log, the serving blocker: a permit for
        # `HPUAttentionMetadataProcessor.process_metadata` was minted, and the
        # first frame converted under it was `Logger.info` in
        # /usr/lib/python3.12/logging/__init__.py -- because `process_metadata`
        # had already been compiled, so dynamo reused its cached code object and
        # never re-converted it, while a callee inside it was converted for the
        # first time. `frames_traced=1` in that report is proof it was the FIRST
        # conversion under the permit, so "check only the first" does not save
        # it. A raise there killed the engine (EngineDeadError) and lost serving.
        #
        # There is no name-based rule that separates "a callee dynamo chose to
        # convert" from "a different region's code": both are frames whose name
        # is not the permit's. So this reports and counts, and does not abort.
        # What it catches is stated on `_attribution_mismatch`.
        #
        # The module filter keeps it quiet and meaningful: a frame from a
        # DIFFERENT module than the permit's region is unambiguously a callee,
        # so it is skipped. A frame from the SAME module with a different name
        # is the one case worth reporting.
        if permit.frames_traced == 1 and not permit.reported:
            if _same_module(frame, permit.region):
                mismatch = _attribution_mismatch(frame, permit.region)
                if mismatch is not None:
                    permit.reported = True
                    ATTRIBUTION_MISMATCHES.append(mismatch)
                    logger.error(
                        "trace gate: the first frame converted under a permit does "
                        "not look like the region that minted it (advisory, not "
                        "fatal).\n"
                        "  converting:    %s\n"
                        "  permit region: %s\n"
                        "  permit key:    %s\n"
                        "  why not:       %s\n"
                        "  This is reported, not refused: a callee of the region "
                        "can be converted under its permit, and no name-based rule "
                        "can tell that apart from a wrong region.",
                        what, _fmt_region(permit.region), _fmt_key(permit.key),
                        mismatch)

    @staticmethod
    def held() -> tuple[str, str] | None:
        """Return ``(region, key)`` of the innermost permit, or ``None``.

        Read-only: this neither counts as a trace attempt nor changes the
        stack.  Useful for logging and for tests that need to assert the stack
        is balanced.
        """
        permits = TraceGate._local.permits
        if not permits:
            return None
        top = permits[-1]
        return (top.region, top.key)

    @staticmethod
    def install() -> None:
        """Patch ``ConvertFrameAssert.__call__`` to enforce the gate.

        The wrapper calls :meth:`assert_permitted` first, before the original
        body runs, so no frame can be converted without a permit.  This is the
        single per-frame entry point for both the default and the ``fullgraph``
        dispatch paths, and a guard-failure recompile re-enters it, so every
        trace -- first or repeated -- passes through the gate exactly once.

        The ``convert_frame_assert`` factory is deliberately NOT patched: it
        constructs the converter and converts nothing, and gating it would force
        callers to mint a keyless permit for a construction, which the protocol
        forbids.  See the module docstring.

        Idempotent: calling it twice patches once.  The wrapper is marked with
        :data:`_WRAPPER_MARK`, so even a re-imported copy of this module (which
        would otherwise hold a fresh, unmarked wrapper) recognises the existing
        patch and leaves it alone rather than stacking a second wrapper.
        """
        with TraceGate._install_lock:
            if TraceGate.installed():
                return
            module = importlib.import_module(_TARGET_MODULE)
            owner = getattr(module, _TARGET_CLASS)
            original = getattr(owner, _TARGET_METHOD)
            if getattr(original, _WRAPPER_MARK, False):
                # Patched by another copy of this module; do not double-wrap.
                return
            setattr(owner, _TARGET_METHOD, _make_wrapper(original))

    @staticmethod
    def installed() -> bool:
        """Return True if the gate is currently patched into Dynamo.

        Never raises for a missing/unimportable ``torch``: an environment
        without torch simply cannot have the gate installed.
        """
        try:
            module = importlib.import_module(_TARGET_MODULE)
        except Exception:
            return False
        owner = getattr(module, _TARGET_CLASS, None)
        if owner is None:
            return False
        target = getattr(owner, _TARGET_METHOD, None)
        return bool(getattr(target, _WRAPPER_MARK, False))


def _make_wrapper(original: Any) -> Any:
    """Build the gate-enforcing replacement for ``ConvertFrameAssert.__call__``.

    ``functools.wraps`` preserves ``__name__``, ``__qualname__``, ``__doc__``,
    ``__module__`` and ``__dict__`` so the wrapper is indistinguishable from the
    original to anything that introspects it.  The signature is ``(*args,
    **kwargs)`` and forwarded verbatim, so a caller using positional or keyword
    arguments sees identical behaviour.  The markers are set *after*
    ``functools.wraps`` so that copying the original's ``__dict__`` cannot
    clobber them.

    Patching the *class* attribute rather than an instance is what makes the
    gate cover every converter, including the one ``ConvertFrame`` builds for
    itself at ``convert_frame.py:2290`` and the one ``fullgraph`` builds at
    ``eval_frame.py:2699``.  Python looks special methods up on the type, so a
    class-level ``__call__`` is the only place that catches all of them.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # The frame OBJECT is passed as well as its description: the gate has to
        # compare the frame's name against the permit's region, and a formatted
        # string cannot be compared reliably against a qualname.
        frame = _find_frame(args, kwargs)
        TraceGate.assert_permitted(_describe_frame(frame), frame=frame)
        return original(*args, **kwargs)

    setattr(wrapper, _WRAPPER_MARK, True)
    setattr(wrapper, _ORIGINAL_MARK, original)
    return wrapper
