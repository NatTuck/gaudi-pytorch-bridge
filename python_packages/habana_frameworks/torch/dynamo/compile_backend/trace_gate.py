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
``torch._dynamo.convert_frame.convert_frame_assert`` is the function named as
the choke point, and :meth:`TraceGate.install` wraps exactly that function, as
specified.  Be aware of its real shape: ``convert_frame_assert`` is a *factory*
that constructs and returns a ``ConvertFrameAssert`` instance, whose
``__call__`` performs the actual per-frame conversion.  So the gate fires at
each construction of the frame converter rather than once per converted frame.
Because the converter is constructed before any frame in its region can be
converted, tracing still cannot begin without a permit -- the gate is real --
but ``frames_traced`` counts gated converter constructions, which is one per
region plus any rebuilds, not one per traced frame.  If per-frame granularity
is wanted, the same ``assert_permitted`` call must additionally be wrapped
around ``ConvertFrameAssert.__call__``.
"""

from __future__ import annotations

import functools
import importlib
import threading
from typing import Any, Final

__all__ = ["TraceWithoutCacheConsult", "TraceGate"]

#: Module that owns the frame-conversion choke point.
_TARGET_MODULE: Final[str] = "torch._dynamo.convert_frame"

#: Attribute inside that module which is wrapped by :meth:`TraceGate.install`.
_TARGET_ATTR: Final[str] = "convert_frame_assert"

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

    __slots__ = ("region", "key", "frames_traced")

    def __init__(self, region: str, key: str) -> None:
        self.region: str = region
        self.key: str = key
        self.frames_traced: int = 0

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


def _locate_frame(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Describe the frame being converted, from a call's arguments.

    ``convert_frame_assert`` does not take a frame directly -- it takes a
    ``compiler_fn`` and returns a converter -- so in the common case there is no
    frame to name here.  We still scan the arguments (and the ``frame`` keyword,
    which other frame-conversion entry points use) because the caller may be
    wrapping a different, frame-taking choke point, and a real
    ``file:line`` in the crash message is far more useful than "unknown".
    """
    candidates: list[Any] = list(args) + list(kwargs.values())
    for candidate in candidates:
        if _looks_like_frame(candidate):
            code = candidate.f_code
            return (
                f"{getattr(code, 'co_name', '<unknown>')} "
                f"({code.co_filename}:{candidate.f_lineno})"
            )
    return (
        "an unnamed frame (no frame argument is visible in the call to "
        "convert_frame_assert; it takes a compiler_fn and returns a converter)"
    )


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
    def assert_permitted(what: str) -> None:
        """Assert that tracing is authorised; count it against the permit.

        Called at the top of every frame conversion.  On success the innermost
        permit's ``frames_traced`` counter is incremented so that
        :meth:`revoke` can report how much tracing the permit covered.

        On an empty stack this raises :class:`TraceWithoutCacheConsult`.  There
        is no return-value signal and no logging-only path: an unauthorised
        trace attempt always raises.
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
        permits[-1].frames_traced += 1

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
        """Patch ``convert_frame.convert_frame_assert`` to enforce the gate.

        The wrapper calls :meth:`assert_permitted` first, before the original
        body runs, so no frame converter can be constructed -- and therefore no
        frame can be traced -- without a permit.

        Idempotent: calling it twice patches once.  The wrapper is marked with
        :data:`_WRAPPER_MARK`, so even a re-imported copy of this module (which
        would otherwise hold a fresh, unmarked wrapper) recognises the existing
        patch and leaves it alone rather than stacking a second wrapper.
        """
        with TraceGate._install_lock:
            if TraceGate.installed():
                return
            module = importlib.import_module(_TARGET_MODULE)
            original = getattr(module, _TARGET_ATTR)
            if getattr(original, _WRAPPER_MARK, False):
                # Patched by another copy of this module; do not double-wrap.
                return
            setattr(module, _TARGET_ATTR, _make_wrapper(original))

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
        target = getattr(module, _TARGET_ATTR, None)
        return bool(getattr(target, _WRAPPER_MARK, False))


def _make_wrapper(original: Any) -> Any:
    """Build the gate-enforcing replacement for ``convert_frame_assert``.

    ``functools.wraps`` preserves ``__name__``, ``__qualname__``, ``__doc__``,
    ``__module__`` and ``__dict__`` so the wrapper is indistinguishable from the
    original to anything that introspects it.  The signature is ``(*args,
    **kwargs)`` and forwarded verbatim, so a caller using positional or keyword
    arguments sees identical behaviour.  The markers are set *after*
    ``functools.wraps`` so that copying the original's ``__dict__`` cannot
    clobber them.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        TraceGate.assert_permitted(_locate_frame(args, kwargs))
        return original(*args, **kwargs)

    setattr(wrapper, _WRAPPER_MARK, True)
    setattr(wrapper, _ORIGINAL_MARK, original)
    return wrapper
