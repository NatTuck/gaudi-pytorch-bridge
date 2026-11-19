"""Make the recipe consult the single choke point for every ``torch.compile``.

The problem this solves
-----------------------
The trace gate (``trace_gate.TraceGate``) refuses to let Dynamo convert a frame
unless a cache consult has just recorded a genuine MISS. The first attempt to
satisfy that invariant put the consult in the callers we own -- the ``Region``
wrapper in ``vllm_gaudi/device_regions.py`` and the model-forward call site in
``hpu_model_runner``. That works for code we own. It does not work for upstream
``vllm-core``, which calls ``torch.compile`` at module scope in places we cannot
turn into ``Region`` s one at a time::

    vllm-core/vllm/utils/deep_gemm.py:708
        @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)

reached from ``vllm-core/vllm/v1/attention/backends/mla/indexer.py:21``. There
are more of them, they are not ours, and patching them one by one is
whack-a-mole: the next ``vllm-core`` bump adds another.

So the consult moves to the one place every ``torch.compile`` in the process
must pass through: ``torch.compile`` itself. ``install()`` patches it, and the
object it now returns is a :class:`ConsultFirst`, which consults on **every
call**. On a hit the recipe is launched from disk and Dynamo is never entered at
all -- not merely not traced, but not reached. On a miss the consult has already
minted the permit, so the trace that follows is legal, and the permit is revoked
exactly once when the call finishes.

Why construction needs no permit
--------------------------------
``torch.compile(fn)`` does not trace anything. It builds a
``ConvertFrameAssert`` converter (``convert_frame.py:811``) and returns; no
frame is converted until the compiled callable is *called*. Measured, not
assumed: at the end of ``import torch`` the trace gate is armed and
``torch._dynamo.utils.counters["frames"]["total"]`` is still 0.

The trace gate is therefore installed on ``ConvertFrameAssert.__call__``
(``convert_frame.py:631``) -- the one function every frame conversion passes
through, on both the default and the ``fullgraph`` dispatch paths -- and not on
the ``convert_frame_assert`` factory. That means construction is simply not
gated, and this module needs no permit for it. An earlier draft minted a
keyless permit around the construction call; that was wrong twice over: an empty
key has no valid circumstance, and the caller was minting on its own behalf
rather than the miss branch of a consult that actually performed a lookup. Both
are forbidden. Nothing here mints; only ``recipe_consult.consult`` does.

Known boundaries (each fails loudly, none is silently absorbed)
--------------------------------------------------------------
* A target with no Python source -- ``torch.compile(torch.ops.foo.bar)``, a
  builtin, a ``functools.partial`` -- cannot be keyed. ``compile_inputs``
  raises ``UnnamedCompile`` from the consult and it propagates. That is the
  protocol's rule ("library code is identified by version"), not a defect of
  this module; such a call site has to be named or exempted deliberately.
* A compiled target re-entered with the *same* ``(region, key)`` while its own
  permit is still on the stack is rejected by ``TraceGate.mint`` as a
  double-arm bug. Recursive compiled functions with identical arguments are the
  way to hit it.
* ``torch._dynamo.optimize`` called directly, or ``from torch import compile``
  executed *before* ``install()``, bypasses the patch. ``torch.compiler.compile``
  does not: it delegates to ``torch.compile`` at call time
  (``torch/compiler/__init__.py:52-57``).
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Final

__all__ = ["ConsultFirst", "install", "installed"]

logger = logging.getLogger(__name__)

#: Same marker string ``trace_gate`` uses, deliberately: one predicate should
#: recognise a patched entry point regardless of which module patched it.
_WRAPPER_MARK: Final[str] = "__trace_gate_wrapper__"

#: Holds the function our patch replaced, for introspection and for tests.
_ORIGINAL_MARK: Final[str] = "__compile_gate_original__"

#: Serialises :func:`install` so two threads cannot both patch ``torch.compile``.
_install_lock: Final[threading.Lock] = threading.Lock()


def _qualified(fn: Any) -> str:
    """``module:qualname`` for ``fn``, degrading rather than raising.

    This is only used to build a human-readable region name for log lines and
    error messages, so a target without ``__qualname__`` (an ``OpOverload``, a
    builtin) gets its ``repr`` instead of a crash here. The crash, when there is
    one, comes later from the key builder, which is where it belongs.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not qualname:
        qualname = getattr(fn, "__name__", None)
    if not qualname:
        return repr(fn)
    return f"{module}:{qualname}" if module else str(qualname)


def _region_name(fn: Any) -> str:
    """The stable, human name of a compiled callable, used in logs and errors."""
    return f"torch.compile:{_qualified(fn)}"


def _resolve(target: Any) -> tuple[Any, Any]:
    """Return ``(fn, module)`` describing ``target`` for the cache record.

    ``fn`` is what the key hashes: the function's own source. ``module`` is what
    supplies the guarded instance attributes. For an ``nn.Module`` the function
    is the *class's* ``forward`` -- an unbound function, which has the
    ``__code__`` and ``__qualname__`` the key builder needs, and which is the
    same function for every instance of that class. For anything else the
    target is already the function.
    """
    import torch.nn

    if isinstance(target, torch.nn.Module):
        return type(target).forward, target
    return target, None


def _source_position(fn: Any) -> tuple[str | None, int | None]:
    """The definition site of ``fn``, or ``(None, None)`` when it has no code."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return None, None
    return getattr(code, "co_filename", None), getattr(code, "co_firstlineno", None)


class ConsultFirst:
    """The object ``torch.compile`` now returns: consult, then dispatch.

    Every call goes through the recipe consult first. A hit is served from disk
    and Dynamo is never entered; a miss has already minted a trace permit, so
    the call is dispatched to the real compiled callable inside that permit, and
    the permit is revoked exactly once when the call returns or raises.

    It is a drop-in for the object it wraps: unknown attributes are forwarded,
    ``__wrapped__`` points at the wrapped compiled callable, and the usual
    function metadata is copied when the wrapped object is a function.

    It is *not* an ``nn.Module`` even when the wrapped object is one. Upstream
    code that does ``isinstance(x, torch.nn.Module)`` on the result of
    ``torch.compile`` will see ``False`` here. Attribute forwarding covers the
    common uses (``.forward``, parameters, submodules); the ``isinstance`` test
    is the one thing forwarding cannot cover, and it is called out rather than
    papered over.
    """

    def __init__(self, compiled: Any, target: Any) -> None:
        # Set the private attributes FIRST: __getattr__ is live from the moment
        # the instance exists, and it must find _compiled rather than recurse.
        self._compiled = compiled
        self._target = target
        self._fn, self._module = _resolve(target)
        self._name = _region_name(self._fn)
        self._file, self._line = _source_position(self._fn)

        # Standard wrapper protocol: __wrapped__ is the thing this object wraps.
        # torch's own __wrapped__/_orig_mod on the compiled object stay reachable
        # through the chain (cf.__wrapped__.__wrapped__), not flattened away.
        self.__wrapped__ = compiled

        for attribute in ("__name__", "__qualname__", "__doc__", "__module__"):
            try:
                setattr(self, attribute, getattr(compiled, attribute))
            except AttributeError:
                pass

        orig_mod = getattr(compiled, "_orig_mod", None)
        if orig_mod is not None:
            self._orig_mod = orig_mod

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from . import launch_ledger, recipe_consult
        from .trace_gate import TraceGate

        # NO try/except here on purpose. A consult that cannot build a key, or a
        # cache that cannot be read, must stop the run: a fallback to the
        # uncompiled callable would hide exactly the class of bug this is for.
        key, served = recipe_consult.consult(
            fn=self._fn,
            args=args,
            kwargs=kwargs,
            module=self._module,
            region_name=self._name,
        )

        if served is not recipe_consult.MISS:
            # A hit: the recipe was launched from disk and nothing was traced.
            return served

        # A miss: consult() minted the permit. Publish the region key so the
        # frames this trace produces are attributed to this region, then
        # discharge the permit unconditionally.
        token = launch_ledger.set_declared_identity(self._name, self._file, self._line, key)
        try:
            return self._compiled(*args, **kwargs)
        finally:
            frames = launch_ledger.take_bundle(key)
            if frames:
                recipe_consult.record_bundle(self._name, key, frames)
            TraceGate.revoke()
            launch_ledger.clear_declared_identity(token)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped compiled callable.

        Guarded against the two ways this can go wrong: being asked for
        ``_compiled`` itself before ``__init__`` has set it (which would recurse
        forever), and being asked for anything at all on a half-built instance.
        """
        if name in {
            "_compiled",
            "_target",
            "_fn",
            "_module",
            "_name",
            "_file",
            "_line",
        }:
            raise AttributeError(name)
        try:
            compiled = self.__dict__["_compiled"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(compiled, name)

    def __repr__(self) -> str:
        return f"<ConsultFirst {self._name} -> {type(self._compiled).__name__}>"


def _wrap_decorator(decorator: Any) -> Any:
    """Wrap the decorator that bare ``torch.compile(...)`` returns.

    ``torch.compile`` with no model returns ``fn`` whose body calls the
    module-global ``compile`` again (``torch/__init__.py:2769-2788``). Because
    :func:`install` patches that very global, the recursive call already lands
    in our wrapper and the result is already a :class:`ConsultFirst`; the
    ``isinstance`` check keeps this from stacking a second one. If a future
    torch stops re-entering the module global, the wrap below still produces the
    right object.
    """

    @functools.wraps(decorator)
    def consult_decorator(target: Any) -> Any:
        compiled = decorator(target)
        if isinstance(compiled, ConsultFirst):
            return compiled
        return ConsultFirst(compiled, target)

    return consult_decorator


def _make_compile_wrapper(original: Any) -> Any:
    """Build the replacement for ``torch.compile``.

    ``functools.wraps`` preserves ``__name__``/``__doc__``/``__module__`` and
    sets ``__wrapped__``; the arguments are forwarded verbatim so positional and
    keyword forms both behave exactly as before. The markers are set *after*
    ``functools.wraps`` so copying the original's ``__dict__`` cannot clobber
    them.
    """

    @functools.wraps(original)
    def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
        target = kwargs.get("model", args[0] if args else None)

        # NO permit here. Construction builds the converter and converts no
        # frame, so the gate (installed on ConvertFrameAssert.__call__) does not
        # fire, and nothing here is allowed to mint: a permit comes only from
        # the miss branch of a consult that actually performed a lookup.
        compiled = original(*args, **kwargs)

        if target is None:
            # Decorator form: no model yet, so the original returned a decorator.
            return _wrap_decorator(compiled)

        return ConsultFirst(compiled, target)

    setattr(compile_wrapper, _WRAPPER_MARK, True)
    setattr(compile_wrapper, _ORIGINAL_MARK, original)
    return compile_wrapper


def install() -> None:
    """Patch ``torch.compile`` so it returns :class:`ConsultFirst`.

    Idempotent and lock-guarded. A second call is a no-op, and so is a call made
    after another copy of this module (a re-import under a different name) has
    already patched, because the wrapper carries :data:`_WRAPPER_MARK`.

    Safe to call from ``habana_frameworks/torch/dynamo/compile_backend/__init__.py``:
    that package is imported during ``import torch`` itself, via
    ``habana_frameworks/torch/__init__.py:81`` ->
    ``habana_frameworks/torch/core/__init__.py:53``, and by then ``torch.compile``
    already exists -- the device-backend autoload runs at
    ``torch/__init__.py:3076-3077``, long after ``def compile`` at
    ``torch/__init__.py:2580``.
    """
    with _install_lock:
        if installed():
            return
        import torch

        original = getattr(torch, "compile", None)
        if original is None:
            raise RuntimeError(
                "compile_gate.install(): torch.compile does not exist yet. This "
                "must run after torch's own module body has finished defining "
                "compile (torch/__init__.py:2580); the device-backend autoload at "
                "torch/__init__.py:3076 is late enough, a call from earlier in "
                "torch/__init__.py is not.")
        if getattr(original, _WRAPPER_MARK, False):
            # Already patched by another copy of this module; do not double-wrap.
            return
        setattr(torch, "compile", _make_compile_wrapper(original))


def installed() -> bool:
    """True when ``torch.compile`` is currently the patched wrapper.

    Never raises for an unimportable or partially-initialised ``torch``: during
    ``import torch`` the module exists in ``sys.modules`` but may not yet define
    ``compile``, and "not installed" is the honest answer there.
    """
    try:
        import torch
    except Exception:
        return False
    target = getattr(torch, "compile", None)
    return bool(getattr(target, _WRAPPER_MARK, False))
