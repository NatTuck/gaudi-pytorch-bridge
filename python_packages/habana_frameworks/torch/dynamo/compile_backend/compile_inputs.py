# SPDX-License-Identifier: Apache-2.0
"""Explicit inputs to compilation. One snapshot; no ambient reads downstream.

The principle: **compilation must be a real function of explicit inputs.** An
environment variable that changes compilation without being an explicit function
argument is a defect, not something to accommodate with a bigger hash. This module
is the single place ambient state is read; everything on the compile path reads
the frozen values captured here.

## The key

    hash( code_identity, shapes, ints, context, config,
          torch_version, synapse_version, bridge_source )

`code_identity` is a NORMALISED-AST hash of the region function plus every
in-file function it transitively calls -- see `code_identity` below for why it is
an AST and not bytecode.

Nothing graph-derived is in the key: `op_strs`, the graph topology and the graph
hash are all computed AFTER tracing, so keying on them would make the key
uncomputable without tracing, which is the thing we are trying to avoid.

## What there is NOT

There is no collision detector. One existed and was removed: it ran only after
`graph_compile`, so it covered the cold path and was blind on the warm path --
which is the case that matters, because a warm boot loads the recipe from disk and
never recompiles. A check that protects the rare path while appearing to protect
the common one is worse than none, because it invites trust.

THE CONTRACT (PROTOCOL §11): key = actual inputs. If you change code that alters
lowering, CLEAR THE CACHE -- or the affected entries -- BY HAND. Nothing here does
that automatically, deliberately: an automatic response would hide the signal that
the lowering changed.
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

    PROTOCOL §8: an invalidation is legitimate only if the recipe actually
    changed. A comment does not reach bytecode at all, so it cannot change this.
    A DOCSTRING does -- it lives in `co_consts` -- and a docstring does not change
    a lowered graph either, so including it would be a §8 violation rather than a
    caveat.

    So exactly ONE constant is dropped: `co_consts[0]` when it is a `str`, which is
    the docstring by definition. Every OTHER string constant is kept, because a
    string passed to an op genuinely can change the graph and dropping it would
    make the record unsound in the other direction.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return "none"
    consts = code.co_consts
    if consts and isinstance(consts[0], str):
        consts = consts[1:]
    h = hashlib.blake2b(digest_size=8)
    h.update(code.co_code)
    h.update(repr(code.co_names).encode("utf-8", "replace"))
    h.update(repr(consts).encode("utf-8", "replace"))
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
    # ENUMERATED CONFIG FLAGS ONLY. An earlier version folded in a digest of the
    # whole environment as deliberate over-coverage; that was wrong under the
    # principle that the key depends only on the actual inputs, because it made an
    # unrelated variable change produce a spurious miss. The detector covers the
    # unenumerated case better anyway -- it compares GRAPH HASHES rather than
    # hoping a digest moved.
    config = tuple((name, flag(name)) for name in _CONFIG_FLAGS) + (
        # Toolchain identity. Not traced into, so its identity has to come from a
        # version or a source hash -- otherwise a change that alters the lowered
        # graph is invisible. `bridge_source` is a SOURCE hash by ruling, not a
        # `.so` hash: the bridge is our code, so a source change invalidates and a
        # rebuild with no source change does not.
        ("torch_version", _torch_version()),
        ("synapse_version", _synapse_version()),
        ("bridge_source", _bridge_source()),
    )
    return CompileInputs(
        context=context,
        config=config,
        instance=tuple(instance),
        region=tuple(region),
        shapes=tuple(shapes),
    )





# ---------------------------------------------------------------------------
# Code identity: a normalised-AST hash of the region and its in-file closure
# ---------------------------------------------------------------------------

# Directories whose source participates in the identity. Anything else (torch,
# stdlib, Synapse) is opaque and is identified by version instead.
_PROJECT_ROOTS = ("vllm-gaudi", "vllm-core", "gaudi-pytorch-bridge")

# (path, mtime) -> parsed Module. A repeat call for the same file must NOT re-parse:
# that is what the counter assertion below is for.
_PARSE_CACHE: dict[tuple[str, float], object] = {}
_PARSE_COUNT = 0


def parse_count() -> int:
    """How many times `ast.parse` has actually run. The counter assertion reads it."""
    return _PARSE_COUNT


def reset_parse_cache() -> None:
    """Test seam: drop the parse cache and the counter."""
    global _PARSE_COUNT
    _PARSE_CACHE.clear()
    _PARSE_COUNT = 0


def _in_project(path: str) -> bool:
    return any(root in path for root in _PROJECT_ROOTS)


def _parse(path: str):
    """Parse `path`, cached by (path, mtime). Counts real parses."""
    global _PARSE_COUNT
    import ast
    import os

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime)
    hit = _PARSE_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            source = stream.read()
    except OSError:
        return None
    _PARSE_COUNT += 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    _PARSE_CACHE[key] = tree
    return tree


def _strip_docstrings(node):
    """Remove docstring Expr(Constant(str)) nodes, in place.

    A docstring is documentation, not logic: PROTOCOL §8 says an invalidation is
    legitimate only if the recipe actually changed, and a docstring does not change
    a lowered graph. Comments never reach an AST at all.
    """
    import ast

    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
    return node


def _module_functions(tree) -> dict:
    """Top-level functions in one module, by name."""
    import ast

    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _names_used(node) -> set:
    """Every bare name and attribute base referenced inside `node`."""
    import ast

    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)
        elif isinstance(child, ast.Attribute):
            base = child
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                out.add(base.id)
    return out


def _closure(path: str, fn_name: str):
    """The in-file transitive closure of `fn_name`, as a sorted list of (name, node).

    BOUNDED TO ONE FILE BY MEASUREMENT: every compiled region's in-project closure
    is exactly one file -- 56 regions, 19 files, zero closures spanning more than
    one file. So no cross-file walk is needed, and a cross-file call is treated as
    opaque rather than chased.
    """
    tree = _parse(path)
    if tree is None:
        return []
    funcs = _module_functions(tree)
    if fn_name not in funcs:
        return []
    seen = set()
    queue = [fn_name]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        node = funcs.get(name)
        if node is None:
            continue
        for used in _names_used(node):
            if used in funcs and used not in seen:
                queue.append(used)
    return [(name, funcs[name]) for name in sorted(seen)]


def code_identity(fn) -> str:
    """hash(normalised AST of fn + every in-file function it transitively calls).

    Normalised: docstrings stripped, no line numbers, no attributes, fields
    unannotated. Two files differing only in comments or whitespace produce the
    same digest, which is the PROTOCOL §8 requirement.

    A function whose source cannot be resolved raises `UnnamedCompile` rather than
    returning a placeholder: a placeholder would keep working while keyed on
    something we cannot see, and nobody would know which compiles were doing it.
    """
    import ast
    import hashlib

    path = getattr(getattr(fn, "__code__", None), "co_filename", None)
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", None))
    if not path or not qualname:
        raise UnnamedCompile(
            f"cannot resolve code identity for {fn!r}: no __code__.co_filename or "
            f"__qualname__. A region must wrap a module-level function in project "
            f"source; a lambda, a bound method of a builtin, or a functools.partial "
            f"has no source to hash.")
    if not _in_project(path):
        raise UnnamedCompile(
            f"cannot resolve code identity for {qualname} at {path}: the file is not "
            f"under any of {_PROJECT_ROOTS}, so its source is not part of the key. "
            f"Regions must wrap project code; library code is identified by version.")

    # Methods appear as "Class.method"; the closure walk works on module-level names.
    name = qualname.split(".")[-1]
    entries = _closure(path, name)
    if not entries:
        raise UnnamedCompile(
            f"cannot resolve code identity for {qualname} at {path}: no module-level "
            f"function named {name!r} was found. If the region wraps a method, wrap "
            f"the module-level function it calls instead.")

    h = hashlib.blake2b(digest_size=16)
    for entry_name, node in entries:
        normalised = _strip_docstrings(node)
        h.update(entry_name.encode("utf-8"))
        h.update(b"\0")
        h.update(ast.dump(normalised, annotate_fields=False,
                          include_attributes=False).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# Region name -> code identity, registered where the function object exists (the
# `Region` wrapper) and looked up here where only the name exists (the launcher).
_region_code: dict[str, str] = {}


def register_region_code(name: str, fn: Any) -> None:
    """Record a region's code identity. Called by `Region.__init__`.

    USES `code_identity`, NOT `code_hash`. The bytecode hash was interpreter-
    dependent (`co_code` changes with the Python version and with `-O`), which made
    the key implicitly depend on the interpreter.

    RAISES `UnnamedCompile` WHEN IT CANNOT RESOLVE. No try/except here: a swallowed
    failure is a compile keyed on something we cannot see, and nobody would know
    which ones. Failing at construction is also the completeness check -- the first
    boot reports the true population of compiled regions.
    """
    _region_code[name] = code_identity(fn)


def region_code(name: str) -> str:
    return _region_code.get(name, "unknown")


def key_for_region(name: str) -> str:
    """The recipe cache key for a region: hash of every DECLARED input.

        hash( code_identity, shapes, ints, context, config,
              torch_version, synapse_version, bridge_source )

    `code_identity` arrives through `region`, which is populated by
    `register_region_code` from the region's AST closure. Shapes and ints are the
    `shapes` field. Context and config come from the frozen snapshot.

    NOT in the key, deliberately: `libs_env_hash()`, `.so` content, and anything
    graph-derived (`op_strs`, topology, the graph hash) -- those are computed AFTER
    tracing, and keying on them would make the key uncomputable without tracing,
    which is the thing this work removes.
    """
    return build(
        instance=(("region", name),),
        region=(("name", name), ("code", region_code(name))),
    ).key()


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _synapse_version() -> str:
    """The Synapse/Habana toolchain version.

    IN THE KEY because we do not trace into Synapse: its identity has to come from
    its version, or a change that alters the lowered graph is invisible.

    RAISES when it cannot be determined. Recording "unknown" would mean two
    different toolchains produce the same key and nothing would notice -- the same
    silent fallback this module exists to remove.
    """
    try:
        import habana_frameworks.torch as _htorch

        # `habana_frameworks.torch.__version__` is the Synapse/Habana stack
        # version (e.g. "1.24.2.481"). The TOP-LEVEL `habana_frameworks` package
        # carries no `__version__` -- verified, not assumed.
        version = getattr(_htorch, "__version__", None)
        if version:
            return str(version)
    except Exception as exc:  # noqa: BLE001 - re-raised below with context
        raise RuntimeError(
            f"cannot determine the Synapse version for the recipe key: {exc!r}. "
            "It is an input to the key, so it must not be guessed.") from exc
    raise RuntimeError(
        "cannot determine the Synapse version for the recipe key: "
        "habana_frameworks.torch.__version__ is absent. It is an input to the key, "
        "so it must not be guessed.")


# Bridge Python whose source participates in the key (see `_bridge_source`).
_BRIDGE_LOWERING_SOURCES = ("passes.py", "compilers.py", "partition_fn.py",
                            "recipe_compiler.py")
_bridge_source_cache: str | None = None


def _bridge_source() -> str:
    """A hash of the lowering-path bridge SOURCE.

    RULING, FLAGGED FOR OVERRULE: the bridge contributes a hash of its source, not
    of its `.so` content. The bridge is our code, so it is traced like the rest: a
    source change invalidates, and a rebuild with no source change does not.

    Cached: the files do not change within a process, and this runs per compile.
    """
    global _bridge_source_cache
    if _bridge_source_cache is not None:
        return _bridge_source_cache
    h = hashlib.blake2b(digest_size=16)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _BRIDGE_LOWERING_SOURCES:
        path = os.path.join(here, name)
        try:
            with open(path, "rb") as stream:
                h.update(stream.read())
        except OSError as exc:
            raise RuntimeError(
                f"cannot hash the bridge lowering source {path!r}: {exc}. It is an "
                f"input to the recipe key, so it must not be skipped.") from exc
        h.update(b"\0")
    _bridge_source_cache = h.hexdigest()
    return _bridge_source_cache


def reset_bridge_source_cache() -> None:
    """Test seam."""
    global _bridge_source_cache
    _bridge_source_cache = None


class UnnamedCompile(RuntimeError):
    """A compile reached the key with no resolvable code identity. Fatal by design.

    The alternative -- falling back to the graph hash -- would keep working while
    keyed on something explicitly ruled out, and nobody would know which compiles
    were doing it. Raising also gives the true population of compiled regions on
    the first boot, which closes the "how many are there really" gap.
    """
