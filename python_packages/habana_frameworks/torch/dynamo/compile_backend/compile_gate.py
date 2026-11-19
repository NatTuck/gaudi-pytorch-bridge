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
object it now returns consults on **every call**. On a hit the recipe is
launched from disk and Dynamo is never entered at all -- not merely not traced,
but not reached. On a miss the consult has already minted the permit, so the
trace that follows is legal, and the permit is revoked exactly once when the
call finishes.

Two wrapper classes, one consult path
-------------------------------------
``torch.compile`` has two kinds of target and they must be wrapped differently.

* A **plain function** or method gets :class:`ConsultFirst`: a callable that is
  deliberately *not* an ``nn.Module``, because turning a function into a module
  would change how torch treats it.

* An **``nn.Module``** gets :class:`ConsultFirstModule`, which *is* an
  ``nn.Module``. This is not cosmetic. ``torch.compile(module)`` returns an
  ``OptimizedModule``, which is an ``nn.Module``, and the engine assigns
  compiled modules into the model tree during weight loading:

      hpu_model_runner.py:8224  _compile_region
        -> vllm_gaudi/utils.py:151  setattr_nested -> setattr(obj, part, value)
          -> torch/nn/modules/module.py:2020  nn.Module.__setattr__

  which raised ``TypeError: cannot assign 'torch._dynamo.eval_frame.protect.
  <locals>.wrapper' as child module '0' (torch.nn.Module or None expected)``
  and killed the boot. A non-module wrapper cannot be assigned there, so the
  module case must present as a module.

The two classes differ ONLY in how they present themselves to torch. Both call
the same :func:`_consult_and_dispatch`, so there is exactly one consult path and
it cannot drift between them.

How the module wrapper presents itself (and why it is not simply nesting)
-------------------------------------------------------------------------
``OptimizedModule`` keeps its original module in ``_modules["_orig_mod"]`` and
nothing else, so its ``state_dict()`` keys are ``_orig_mod.<...>``. To keep
checkpoint compatibility, :class:`ConsultFirstModule` mirrors that exactly: it
registers the *original* module under ``_orig_mod`` and holds the compiled
object outside the module tree.

Registering the compiled object as the child instead would be one level too
deep -- keys would become ``_orig_mod._orig_mod.<...>`` -- which silently
renames every key in the checkpoint. So the child is the original module, and
the compiled callable is stashed with ``object.__setattr__``, which writes
``__dict__`` directly and therefore leaves ``_modules`` with exactly one entry.
The parameters are still the same objects the compiled callable uses (the
compiled module wraps that very module), so ``.parameters()``, ``.to()``,
``.train()``/``.eval()`` and ``state_dict()`` all behave as they do on the raw
``torch.compile`` result.

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
* A module wrapper forwards attribute *reads* to the compiled object, so
  ``wrapper.some_submodule`` works, but assigning ``wrapper.some_submodule = x``
  registers it in the wrapper's own ``_modules`` rather than in ``_orig_mod``.
  ``OptimizedModule`` has the same hazard, so this matches the behaviour it
  replaces rather than introducing a new one.
"""

from __future__ import annotations

import functools
import logging
import threading
import weakref
from typing import Any, Final

import torch.nn as nn

__all__ = ["ConsultFirst", "ConsultFirstModule", "install", "installed"]

logger = logging.getLogger(__name__)

#: Same marker string ``trace_gate`` uses, deliberately: one predicate should
#: recognise a patched entry point regardless of which module patched it.
_WRAPPER_MARK: Final[str] = "__trace_gate_wrapper__"

#: Holds the function our patch replaced, for introspection and for tests.
_ORIGINAL_MARK: Final[str] = "__compile_gate_original__"

#: Serialises :func:`install` so two threads cannot both patch ``torch.compile``.
_install_lock: Final[threading.Lock] = threading.Lock()

#: Attributes that ``__getattr__`` may be asked for before ``__init__`` has set
#: them. Answering those with a forwarded lookup would recurse, so they raise
#: AttributeError instead. ``_orig_mod`` is NOT in this set: on the module
#: wrapper it is a real registered child and must resolve normally.
_PRIVATE_ATTRS: Final[frozenset[str]] = frozenset(
    {"_compiled", "_target", "_fn", "_module", "_name", "_file", "_line"}
)


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


#: Region name -> the wrapper object the serving path calls.
#
# WHY A REGISTRY EXISTS AT ALL. The warm replay (`warm_replay.replay_capture`)
# has a NAME -- the one the capture file recorded, which is the name the consult
# logs -- and it must call the SAME object the serving path calls, or the consult
# it runs is not the consult serving runs and the warm is theatre. For a
# `Region` that is `vllm_gaudi/device_regions.py`'s registry; for everything
# `torch.compile` produced, this is it.
#
# IT MATTERS FOR THIS MODEL: boot-b8.log has 9 `Qwen3NextDecoderLayer.forward`
# compiles at serving shapes, plus `model.forward` and 4
# `extension.ops:_dynamic_quant_naive`, and none of those is a `Region`. They are
# compiled targets whose only handle is the wrapper built here.
#
# WEAK, so the registry cannot extend the life of a compiled module that the
# model tree has already dropped: a wrapper that is gone reports "not registered
# in this process" rather than being resurrected from a diagnostic table.
_REPLAY_REGISTRY: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


def compiled_regions() -> "weakref.WeakValueDictionary[str, Any]":
    """The `torch.compile` wrappers by region name, for the warm replay."""
    return _REPLAY_REGISTRY


def _resolve(target: Any) -> tuple[Any, Any]:
    """Return ``(fn, module)`` describing ``target`` for the cache record.

    ``fn`` is what the key hashes: the function's own source. ``module`` is what
    supplies the guarded instance attributes. For an ``nn.Module`` the function
    is the *class's* ``forward`` -- an unbound function, which has the
    ``__code__`` and ``__qualname__`` the key builder needs, and which is the
    same function for every instance of that class. For anything else the
    target is already the function.
    """
    if isinstance(target, nn.Module):
        return type(target).forward, target
    return target, None


def _source_position(fn: Any) -> tuple[str | None, int | None]:
    """The definition site of ``fn``, or ``(None, None)`` when it has no code."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return None, None
    return getattr(code, "co_filename", None), getattr(code, "co_firstlineno", None)


#: Marker set on a function already made opaque, so the application is idempotent
#: and can be retried (it is attempted at `install()` and again on the first
#: consult, because `install()` runs during `import torch`, where `torch._dynamo`
#: may not be importable yet).
_OPAQUE_MARK: Final[str] = "__compile_gate_opaque__"

#: Whether the opacity application has succeeded at least once.
_OPACITY_APPLIED: bool = False

#: How many self-heal attempts have been made. Bounded so an environment where
#: `torch._dynamo` never becomes importable gets three ERROR lines, not one per
#: region call.
_OPACITY_ATTEMPTS: int = 0
_OPACITY_MAX_ATTEMPTS: Final[int] = 3


def _make_opaque(fn: Any) -> Any:
    """Wrap `fn` so dynamo treats it as ONE opaque call instead of tracing it.

    WHY THIS IS NEEDED. The consult runs inside dynamo traces -- a compiled
    region called from a traced model frame calls straight into this module --
    and its Python is not traceable. Dynamo then GRAPH-BREAKS the enclosing
    frame, splits it, and the remainder becomes a new `graph_<id>` that is
    compiled for the first time on the SERVING thread (warmup is over), where it
    can be rejected with synStatus 26.

    Measured: a boot that served 6/6 had zero dynamo frames inside our modules;
    the failing boot had 15, named `torch_dynamo_resume_in_consult_at_771`,
    `_consult_and_dispatch` (this file, :189) and `__call__` (:275).
    `torch_dynamo_resume_in_*` is dynamo's name for the second half of a split
    frame.

    `recursive=False` IS LOAD-BEARING. The default is recursive, which would
    suppress tracing for everything called from here -- including
    `compiled(*args, **kwargs)`, the region itself. That would stop the region
    being traced at all, and Rule 12 is "hook into @torch.compile, do not
    eliminate it". With `recursive=False` only THIS function is opaque; the
    compiled callable inside it still traces normally (measured both ways on
    this stack).

    Returns `fn` unchanged when `torch._dynamo` is unavailable, and says so at
    ERROR: a silent no-op restores the graph-break bug with nothing in the log
    to explain the synStatus 26 that follows.
    """
    try:
        import torch._dynamo as _dynamo  # noqa: PLC0415 - not importable at module load
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "compile_gate: torch._dynamo is unavailable (%s: %s), so the consult "
            "path CANNOT be made opaque. A compiled region called from a traced "
            "frame will graph-break it and the split frame may be compiled for "
            "the first time on the serving thread (synStatus 26).",
            type(exc).__name__, exc)
        return fn
    try:
        return _dynamo.disable(fn, recursive=False)
    except TypeError:
        return _dynamo.disable(fn)
    except Exception as exc:  # noqa: BLE001
        logger.error("compile_gate: torch._dynamo.disable failed (%s: %s)",
                     type(exc).__name__, exc)
        return fn


def _apply_dynamo_opacity() -> bool:
    """Make the consult path opaque to dynamo. Idempotent; safe to retry.

    ONE place per entry point, and the entry points are exactly three:
    `_consult_and_dispatch` (the single consult path both wrappers share) and
    the two wrapper methods that reach it. Decorating the shared function covers
    the consult and the bundle bookkeeping for both wrappers at once, so the
    path cannot drift; the two methods are covered as well because they are the
    frames dynamo actually sees from model code, and a frame it can see is a
    frame it can split.

    Returns True when everything is opaque.
    """
    global _OPACITY_APPLIED, _consult_and_dispatch
    if getattr(_consult_and_dispatch, _OPAQUE_MARK, False):
        _OPACITY_APPLIED = True
        return True
    wrapped = _make_opaque(_consult_and_dispatch)
    if wrapped is _consult_and_dispatch:
        return False
    setattr(wrapped, _OPAQUE_MARK, True)
    _consult_and_dispatch = wrapped
    # `consult` itself too, so a future caller that is not behind one of the
    # entry points above is still opaque. It lives in recipe_consult, which
    # imports no torch, so the wrapping has to be requested from here.
    try:
        from . import recipe_consult  # noqa: PLC0415
        recipe_consult.make_consult_opaque()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error("compile_gate: could not make recipe_consult.consult opaque "
                     "(%s: %s)", type(exc).__name__, exc)
    for owner, attribute in ((ConsultFirst, "__call__"),
                             (ConsultFirstModule, "forward")):
        current = getattr(owner, attribute)
        if getattr(current, _OPAQUE_MARK, False):
            continue
        wrapped = _make_opaque(current)
        if wrapped is current:
            continue
        setattr(wrapped, _OPAQUE_MARK, True)
        setattr(owner, attribute, wrapped)
    _OPACITY_APPLIED = True
    return True


def _consult_and_dispatch(
    *,
    name: str,
    fn: Any,
    module: Any,
    file: str | None,
    line: int | None,
    compiled: Any,
    args: tuple,
    kwargs: dict,
) -> Any:
    """The ONE consult path, shared by both wrapper classes.

    Consult, serve on a hit, otherwise dispatch inside the permit the consult
    minted and discharge it exactly once. Factored out so the module wrapper and
    the function wrapper cannot drift apart: the two classes differ only in how
    they present themselves to torch, never in what they do here.
    """
    global _OPACITY_ATTEMPTS
    if not _OPACITY_APPLIED and _OPACITY_ATTEMPTS < _OPACITY_MAX_ATTEMPTS:
        # `install()` runs during `import torch`, where `torch._dynamo` may not
        # be importable yet; this is the retry, on a thread and at a time where
        # it is. Bounded, so a permanently unavailable torch._dynamo does not
        # produce one ERROR per region call.
        _OPACITY_ATTEMPTS += 1
        _apply_dynamo_opacity()

    from . import launch_ledger, recipe_consult
    from .trace_gate import TraceGate

    # NO try/except here on purpose. A consult that cannot build a key, or a
    # cache that cannot be read, must stop the run: a fallback to the
    # uncompiled callable would hide exactly the class of bug this is for.
    key, served = recipe_consult.consult(
        fn=fn, args=args, kwargs=kwargs, module=module, region_name=name
    )

    if served is not recipe_consult.MISS:
        # A hit: the recipe was launched from disk and nothing was traced.
        return served

    # A miss: consult() minted the permit. Publish the region key so the frames
    # this trace produces are attributed to this region, then discharge the
    # permit unconditionally.
    token = launch_ledger.set_declared_identity(name, file, line, key)
    try:
        return compiled(*args, **kwargs)
    finally:
        frames = launch_ledger.take_bundle(key)
        if frames:
            recipe_consult.record_bundle(name, key, frames)
        TraceGate.revoke()
        launch_ledger.clear_declared_identity(token)


class ConsultFirst:
    """The object ``torch.compile`` returns for a non-module target.

    Every call goes through the recipe consult first. A hit is served from disk
    and Dynamo is never entered; a miss has already minted a trace permit, so
    the call is dispatched to the real compiled callable inside that permit, and
    the permit is revoked exactly once when the call returns or raises.

    It is a drop-in for the object it wraps: unknown attributes are forwarded,
    ``__wrapped__`` points at the wrapped compiled callable, and the usual
    function metadata is copied when the wrapped object is a function.

    It is deliberately NOT an ``nn.Module``, because a function target must not
    become a module. Module targets get :class:`ConsultFirstModule` instead.
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

        # Published for the warm replay, which has the name and needs the object.
        _REPLAY_REGISTRY[self._name] = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _consult_and_dispatch(
            name=self._name,
            fn=self._fn,
            module=self._module,
            file=self._file,
            line=self._line,
            compiled=self._compiled,
            args=args,
            kwargs=kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped compiled callable.

        Guarded against the two ways this can go wrong: being asked for
        ``_compiled`` itself before ``__init__`` has set it (which would recurse
        forever), and being asked for anything at all on a half-built instance.
        """
        if name in _PRIVATE_ATTRS:
            raise AttributeError(name)
        try:
            compiled = self.__dict__["_compiled"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(compiled, name)

    def __repr__(self) -> str:
        return f"<ConsultFirst {self._name} -> {type(self._compiled).__name__}>"


class ConsultFirstModule(nn.Module):
    """The object ``torch.compile`` returns for an ``nn.Module`` target.

    Same consult-then-dispatch behaviour as :class:`ConsultFirst` -- both call
    :func:`_consult_and_dispatch` -- but shaped as an ``nn.Module``, because the
    engine assigns compiled modules into the model tree and
    ``nn.Module.__setattr__`` rejects a non-module child.

    The original module is registered as the single child ``_orig_mod``, which
    is exactly what ``OptimizedModule`` does, so ``state_dict()`` keys,
    ``.parameters()``, ``.to()``, ``.train()`` and ``.eval()`` all match the raw
    ``torch.compile`` result. The compiled callable itself is held out of the
    module tree with ``object.__setattr__`` so ``_modules`` keeps exactly one
    entry and no parameter is duplicated.

    ``forward`` carries the consult rather than ``__call__``: ``nn.Module``'s
    own ``__call__`` machinery then still runs, so forward hooks, ``_slow_forward``
    under tracing, and the rest of the module protocol keep working. Nothing is
    skipped, because ``_compiled_call_impl`` is a class attribute defaulting to
    ``None`` (``torch/nn/modules/module.py:480``) and therefore never falls
    through to the ``__getattr__`` forwarding below.
    """

    def __init__(self, compiled: Any, target: Any) -> None:
        # super().__init__() MUST come first: nn.Module.__setattr__ is live
        # during __init__ and requires _parameters/_modules/_buffers to exist,
        # so any attribute assignment before this line would be intercepted by
        # a half-built module.
        super().__init__()

        # What we present to torch: the underlying module, not the compiled
        # wrapper. OptimizedModule registers its original module as `_orig_mod`
        # and nothing else, so its state_dict keys are `_orig_mod.*`. Mirroring
        # that keeps checkpoint keys identical; registering the OptimizedModule
        # instead would nest one level deeper and rename every key.
        content = getattr(compiled, "_orig_mod", None)
        if not isinstance(content, nn.Module):
            content = compiled
        self._orig_mod = content

        # Held OUT of the module tree. `compiled` is an nn.Module, so a normal
        # assignment would register it in _modules, giving a second entry and
        # duplicating every parameter in .parameters(). object.__setattr__
        # writes __dict__ directly and leaves _modules with one entry.
        object.__setattr__(self, "_compiled", compiled)
        object.__setattr__(self, "__wrapped__", compiled)

        fn, module = _resolve(target)
        self._fn = fn
        # `module` is the target, which is the same object as `_orig_mod`; it
        # must not go through nn.Module.__setattr__ or it would be registered a
        # second time under a second name.
        object.__setattr__(self, "_module", module)
        self._name = _region_name(fn)
        self._file, self._line = _source_position(fn)

        # Published for the warm replay, which has the name and needs the object.
        _REPLAY_REGISTRY[self._name] = self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _consult_and_dispatch(
            name=self._name,
            fn=self._fn,
            module=self._module,
            file=self._file,
            line=self._line,
            compiled=self._compiled,
            args=args,
            kwargs=kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Let ``nn.Module`` resolve its registries first, then forward.

        ``nn.Module.__getattr__`` is what resolves anything registered in
        ``_parameters``/``_buffers``/``_modules`` -- including ``_orig_mod`` --
        so it must be tried first or the module protocol breaks. Only if it
        fails does the lookup fall through to the wrapped compiled object, which
        is how ``wrapper.some_submodule`` keeps working.

        A missing name therefore raises AttributeError, not something stranger:
        ``nn.Module`` raises it, the forward raises it, and the private-attribute
        guard raises it. That is asserted in the checks.
        """
        if name in _PRIVATE_ATTRS:
            raise AttributeError(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        compiled = self.__dict__.get("_compiled")
        if compiled is None:
            raise AttributeError(name)
        return getattr(compiled, name)

    def extra_repr(self) -> str:
        return f"region={self._name!r}"


def _wrap(compiled: Any, target: Any) -> Any:
    """Pick the wrapper class by what ``torch.compile`` actually returned.

    The test is on ``compiled``, not on ``target``: the object being wrapped is
    what must satisfy the caller's ``isinstance(..., nn.Module)``, and that is
    the compiled object, whatever the target was.
    """
    if isinstance(compiled, nn.Module):
        return ConsultFirstModule(compiled, target)
    return ConsultFirst(compiled, target)


def _wrap_decorator(decorator: Any) -> Any:
    """Wrap the decorator that bare ``torch.compile(...)`` returns.

    ``torch.compile`` with no model returns ``fn`` whose body calls the
    module-global ``compile`` again (``torch/__init__.py:2769-2788``). Because
    :func:`install` patches that very global, the recursive call already lands
    in our wrapper and the result is already wrapped; the ``isinstance`` check
    keeps this from stacking a second wrapper. If a future torch stops
    re-entering the module global, the wrap below still produces the right
    object.
    """

    @functools.wraps(decorator)
    def consult_decorator(target: Any) -> Any:
        compiled = decorator(target)
        if isinstance(compiled, (ConsultFirst, ConsultFirstModule)):
            return compiled
        return _wrap(compiled, target)

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

        return _wrap(compiled, target)

    setattr(compile_wrapper, _WRAPPER_MARK, True)
    setattr(compile_wrapper, _ORIGINAL_MARK, original)
    return compile_wrapper


def install() -> None:
    """Patch ``torch.compile`` so it returns a consult-first wrapper.

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
    # BEFORE the early return: a second install() must still be able to apply
    # the opacity that the first one could not (torch._dynamo not yet importable
    # during `import torch`).
    _apply_dynamo_opacity()

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
