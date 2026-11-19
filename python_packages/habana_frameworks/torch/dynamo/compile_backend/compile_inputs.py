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

import ast
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

    # NO DEFAULTS ON instance/region/shapes. They were defaulted to `()`, so a
    # caller that forgot one produced a key that looked valid and silently omitted
    # the input -- which is how one region came to produce ONE key for hundreds of
    # distinct graphs. `context` and `config` are filled by `build()` itself and
    # cannot be forgotten.
    context: tuple
    config: tuple
    instance: tuple
    region: tuple
    shapes: tuple

    def key(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for field in (self.context, self.config, self.instance, self.region, self.shapes):
            h.update(repr(field).encode("utf-8", "replace"))
            h.update(b"|")
        return h.hexdigest()

    def describe(self) -> str:
        return (f"context={self.context} config={self.config} "
                f"instance={self.instance} region={self.region} shapes={self.shapes}")


def build(*, instance: tuple, region: tuple, shapes: tuple) -> CompileInputs:
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
# Code identity: a normalised-AST hash of the region and its INLINED source
# ---------------------------------------------------------------------------
#
# The identity covers the region and every PROJECT function that is inlined into
# it -- across files, because a region's graph is built from everything dynamo
# inlines, not from one file. It never covers library code: a callee whose source
# is outside `_PROJECT_ROOTS` is opaque by design and identified by version, which
# is the existing rule for library code.

# Directories whose source participates in the identity. Anything else (torch,
# stdlib, C extensions, installed packages, Synapse) is opaque.
_PROJECT_ROOTS = ("vllm-gaudi", "vllm-core", "gaudi-pytorch-bridge")

# (path, st_mtime_ns, st_size) -> parsed Module. A repeat call for an UNCHANGED
# file must not re-parse: that is what the counter assertion below is for.
#
# mtime AND SIZE, not mtime alone. Two writes to one path inside a single kernel
# timestamp tick can share an mtime, and an mtime-keyed cache then serves the
# previous tree -- which is how a test for "no memoisation" can pass or fail by
# luck. Size closes the common half of that hole (a rewrite that changes length);
# `st_mtime_ns` closes the rest as far as the filesystem reports it.
_PARSE_CACHE: dict[tuple[str, int, int], object] = {}
_PARSE_COUNT = 0


def parse_count() -> int:
    """How many times `ast.parse` has actually run. The counter assertion reads it."""
    return _PARSE_COUNT


def reset_parse_cache() -> None:
    """Test seam: drop the parse cache and the counter.

    There is no identity memo to drop: `code_identity` recomputes the digest from
    source on every call (see the note beside `_PARSE_CACHE`). Clearing this cache
    therefore forces a real re-parse, which is what a test wants after it rewrites
    a file.
    """
    global _PARSE_COUNT
    _PARSE_CACHE.clear()
    _PARSE_COUNT = 0


# THERE IS NO MEMO FOR THE DIGEST, DELIBERATELY. One existed -- `_IDENTITY_CACHE`
# plus a `reset_identity_cache()` test seam -- and was REMOVED. It was keyed on the
# FUNCTION OBJECT (`id(fn)` plus an identity check), not on the source. A function
# object cannot change, but its source FILE can, so the memo could return a digest
# for source that is no longer on disk: exactly the "accepting invalid data" this
# module exists to prevent. It also hid the cost of the work the key is supposed to
# do. Do not re-add it as an optimisation; the digest is computed from source on
# every call, including the cross-file walk.
#
# `_PARSE_CACHE` is NOT the same thing and stays: it is keyed on (path, mtime, size),
# so a changed file is re-parsed, and it caches PARSING rather than the identity.


def _in_project(path: str) -> bool:
    return any(root in path for root in _PROJECT_ROOTS)


def _project_relative(path: str):
    """The path from the matched project root onward, or None if external.

    This is the identity's file name. It is NOT the absolute path: a digest that
    embedded one would differ between checkouts, between machines, and between two
    identical trees in different directories, which would make the recipe cache
    useless and would make "same graph, same digest" untestable.
    """
    if not path:
        return None
    flat = path.replace(os.sep, "/")
    for root in _PROJECT_ROOTS:
        marker = "/" + root + "/"
        at = flat.find(marker)
        if at != -1:
            return flat[at + 1:]
        if flat.startswith(root + "/"):
            return flat
    # `_in_project` is a plain substring check; agree with it rather than
    # disagreeing about an unusual path.
    for root in _PROJECT_ROOTS:
        at = flat.find(root)
        if at != -1:
            return flat[at:]
    return None


def _parse(path: str):
    """Parse `path`, cached by (path, st_mtime_ns, st_size). Counts real parses."""
    global _PARSE_COUNT
    import ast

    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
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


# Names that exist in every module without a binding statement. Not callees.
_IMPLICIT_MODULE_NAMES = frozenset((
    "__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__path__", "__debug__", "__dict__", "__class__", "__all__",
    "__annotations__", "__module__", "__qualname__",
))

# ---------------------------------------------------------------------------
# Bindings: what a name MEANS, not what it is called
# ---------------------------------------------------------------------------
#
# Resolution is by BINDING. A bare name is classified, in this order:
#   * local to the function (parameter, assignment, inner def, inner import) -> skip
#   * defined at module scope in this file                                  -> inline
#   * bound by an import: in-project module/symbol                          -> inline
#                         external (torch, stdlib, C ext, installed)        -> skip
#   * a builtin or a module dunder                                          -> skip
#   * nothing anywhere                                                      -> CRASH
#
# THERE IS NO LAST-RESORT SEARCH. An earlier version, when a bare name had no
# binding, searched the region's own directory tree for a definition of that name.
# That was removed: it let a name with no real binding resolve to something, which
# is the "accept invalid data" this module exists to prevent. A name that is not
# local, not module-level, not imported and not a builtin has NO BINDING, and the
# answer is to crash naming the call site -- never to guess where it might live.
#
# Attributes are resolved only where the base is statically known: `self.x` /
# `cls.x` inside a class resolves in that class and its in-project bases, and
# `module.x` resolves when `module` is an in-project import. An attribute on any
# other value is opaque -- `self.model(...)` is an instance attribute assigned in
# `__init__`, which no AST walk can see -- and is skipped rather than guessed at.


class _Module:
    """One parsed file, indexed. Built per call; never cached across calls."""

    __slots__ = ("path", "relpath", "tree", "defs", "classes", "bindings", "stars",
                 "module_members", "_package")

    def __init__(self, path, tree):
        self.path = path
        self.relpath = _project_relative(path) or path
        self.tree = tree
        self.defs = {}          # qualpath tuple -> {member name: node}, built lazily
        self.classes = {}       # qualpath tuple -> ClassDef node
        self.bindings = {}      # name -> [(kind, payload)] ; kind in def/import/node
        self.stars = []         # dotted modules imported with `*`
        self.module_members = {}  # name -> node for module-level defs and classes
        self._package = False   # False = not computed yet

    def members(self, qualpath):
        """The members of the class at `qualpath`, or the module's own defs.

        LAZY, and only ever within one walk: a class body is indexed when it is
        looked up, not when the file is read. Indexing every class in a 2,000-line
        model file to resolve one `self.attr` was most of the walk's cost.
        """
        if qualpath == ():
            return self.module_members
        cached = self.defs.get(qualpath)
        if cached is not None:
            return cached
        members = {}
        node = self.classes.get(qualpath)
        if node is not None:
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.setdefault(stmt.name, stmt)
                elif isinstance(stmt, ast.ClassDef):
                    members.setdefault(stmt.name, stmt)
                    self.classes.setdefault(qualpath + (stmt.name,), stmt)
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    # A CLASS-LEVEL ASSIGNMENT IS AN INPUT. `self.SOME_CONST` read at
                    # runtime changes the graph, and it is in-project source, so it is
                    # a member: its statement is hashed like any other entry. Skipping
                    # it left a constant that the graph depends on out of the key.
                    for target in _assignment_targets(stmt):
                        if isinstance(target, ast.Name):
                            members.setdefault(target.id, stmt)
        self.defs[qualpath] = members
        return members

    def bind(self, kind, name, payload):
        self.bindings.setdefault(name, []).append((kind, payload))

    def package(self):
        """This file's dotted package, computed once per walk (None if unpackage d)."""
        if self._package is False:
            self._package = _package_for_file(self.path)
        return self._package


class _ModuleScopeVisitor(ast.NodeVisitor):
    """Collect module-scope bindings, without entering function or class bodies.

    Entering them would record a local assignment as a module binding, which both
    over-covers (hashing unrelated nodes) and hides the unbound-name crash.
    """

    def __init__(self, mod):
        self.mod = mod

    def visit_body(self, body):
        for stmt in body:
            self.visit(stmt)

    def visit_FunctionDef(self, node):
        self.mod.bind("def", node.name, node)
        self.mod.module_members.setdefault(node.name, node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.mod.bind("def", node.name, node)
        self.mod.module_members.setdefault(node.name, node)
        self.mod.classes.setdefault((node.name,), node)

    def visit_Lambda(self, node):
        return

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.mod.bind("import", alias.asname, (alias.name, None))
            else:
                # `import a.b` binds the root package name `a`.
                self.mod.bind("import", alias.name.split(".")[0], (alias.name, None))

    def visit_ImportFrom(self, node):
        dotted = self._dotted(node)
        for alias in node.names:
            if alias.name == "*":
                if dotted:
                    self.mod.stars.append(dotted)
                continue
            self.mod.bind("import", alias.asname or alias.name,
                          (dotted, alias.name))

    def _dotted(self, node):
        """The absolute module a `from ... import` names, or None if unknown."""
        if not node.level:
            return node.module
        base = self.mod.package()
        if base is None:
            return None
        parts = [p for p in base.split(".") if p]
        if node.level > 1:
            parts = parts[:max(0, len(parts) - (node.level - 1))]
        if node.module:
            parts.extend(node.module.split("."))
        return ".".join(parts) or None

    def visit_Assign(self, node):
        for target in node.targets:
            self._bind_target(target, node)

    def visit_AnnAssign(self, node):
        self._bind_target(node.target, node)

    def visit_AugAssign(self, node):
        self._bind_target(node.target, node)

    def visit_For(self, node):
        self._bind_target(node.target, node)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars, node)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node):
        if node.name:
            self.mod.bind("node", node.name, node)
        self.generic_visit(node)

    def _bind_target(self, target, stmt):
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                self.mod.bind("node", child.id, stmt)


def _build_index(path, tree):
    mod = _Module(path, tree)
    _ModuleScopeVisitor(mod).visit_body(tree.body)
    return mod


def _package_for_file(path):
    """The dotted package of a file, for resolving relative imports.

    Derived from the `__init__.py` chain on disk: O(depth) stat calls. The first
    version scanned sys.modules and realpath'd every entry, which cost 340 stat
    calls PER IDENTITY on a real model file -- 88% of the whole walk, measured --
    and told us nothing the directory layout does not.

    Returns None when the file is not inside a package. A relative import from a
    file whose package cannot be named is then unresolvable, and unresolvable
    imports are skipped (library code) rather than guessed at.
    """
    directory = os.path.dirname(os.path.abspath(path))
    parts = []
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        parts.append(os.path.basename(directory))
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    if not parts:
        return None
    return ".".join(reversed(parts))


def _is_self_attr(target, name):
    """Whether `target` is the attribute `self.<name>`."""
    return (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == name)


def _assignment_targets(stmt):
    """The Name targets of an assignment statement (module or class scope)."""
    if isinstance(stmt, ast.Assign):
        targets = []
        for target in stmt.targets:
            targets.extend(n for n in ast.walk(target) if isinstance(n, ast.Name))
        return targets
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return [n for n in ast.walk(stmt.target) if isinstance(n, ast.Name)]
    return []


def _symbol_node(mod, qualpath, name):
    """The AST node for `name` at `qualpath`, or None."""
    node = mod.members(qualpath).get(name)
    if node is not None:
        return node
    if qualpath == ():
        # A module-level assignment: hash the statement that binds it, so a
        # changed constant is a changed digest.
        for kind, payload in mod.bindings.get(name, ()):
            if kind == "node":
                return payload
    return None


def _module_file_for(dotted):
    """The source file of a dotted module name, without importing anything."""
    import sys

    loaded = sys.modules.get(dotted)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        return origin or None           # a namespace package has no file
    relative = dotted.replace(".", os.sep)
    for entry in sys.path:
        if not entry:
            continue
        base = os.path.join(entry, relative)
        for candidate in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.isfile(candidate):
                return candidate
    return None


class _ScopeScanner(ast.NodeVisitor):
    """Names bound in ONE function's own scope.

    Stops at nested functions, lambdas and classes: their internals are different
    scopes, and treating their assignments as locals here would make an unrelated
    inner assignment hide a real module-level callee.
    """

    def __init__(self):
        self.bound = set()
        self.declared_global = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_arg(self, node):
        self.bound.add(node.arg)

    def visit_Global(self, node):
        self.declared_global.update(node.names)

    def visit_Nonlocal(self, node):
        # Bound in an enclosing function: local as far as this file's module
        # scope is concerned, so it is not a module-level callee.
        self.bound.update(node.names)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        return

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name != "*":
                self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)


def _local_names(fn_node):
    """Everything bound in this function's own scope (args included)."""
    scanner = _ScopeScanner()
    args = fn_node.args
    for arg in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
        scanner.bound.add(arg.arg)
    if args.vararg is not None:
        scanner.bound.add(args.vararg.arg)
    if args.kwarg is not None:
        scanner.bound.add(args.kwarg.arg)
    for stmt in fn_node.body:
        scanner.visit(stmt)
    return scanner.bound - scanner.declared_global


def _refs(fn_node):
    """References in one function's body.

    Each is (kind, payload, line, is_call):
      ("bare", name, line, is_call)
      ("attr", (root, (a, b)), line, is_call)

    `is_call` says the reference is the RECEIVER of a call. That distinction is
    what separates "there is code behind this name" (a receiver: it must resolve,
    or the walk is not hashing part of the graph) from "there is a value behind
    this name" (a plain read: skipped when opaque).

    The base Name of an attribute chain is reported as the CHAIN, never also as a
    bare name: `torch.zeros` is an attribute on an external module, not a bare
    `torch`. An attribute on a call result (`f().x`) is opaque and dropped.
    """
    root = ast.Module(body=list(fn_node.body), type_ignores=[])
    call_funcs = {id(n.func) for n in ast.walk(root) if isinstance(n, ast.Call)}
    attrs = [n for n in ast.walk(root) if isinstance(n, ast.Attribute)]
    inner = set()
    for node in attrs:
        value = node.value
        while isinstance(value, ast.Attribute):
            inner.add(id(value))
            value = value.value
    chains = []
    chain_bases = set()
    for node in attrs:
        if id(node) in inner:
            continue
        parts = []
        base = node
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if not isinstance(base, ast.Name):
            continue
        chain_bases.add(id(base))
        chains.append(("attr", (base.id, tuple(reversed(parts))), node.lineno,
                       id(node) in call_funcs))
    bares = []
    for node in ast.walk(root):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and id(node) not in chain_bases):
            bares.append(("bare", node.id, node.lineno, id(node) in call_funcs))
    return chains + bares


class _Walk:
    """One identity computation: parse cache per call, seen-set, resolution."""

    def __init__(self, region_label):
        self.region_label = region_label
        self.modules = {}       # path -> _Module  (per call, never across calls)
        self.mod_for = {}       # dotted name -> _Module or None  (per call)
        self.bases_of = {}      # (relpath, qualpath) -> bases       (per call)
        self.instance_of = {}   # (relpath, qualpath, name) -> classes (per call)
        self.init_assign = {}   # (relpath, qualpath, name) -> (mod, RHS) (per call)
        # ATTRIBUTES WHOSE VALUE IS A RUNTIME INPUT, not source: a call receiver
        # `self.<attr>...` where `<attr>` is assigned in __init__ to something no
        # static resolver can know (a parameter, a factory call, a literal). The
        # code behind it cannot be hashed, so its IDENTITY is captured in the
        # key's `instance` field at call time instead.
        self.runtime_dispatched = set()
        self.runtime_sites = []
        self.seen = {}          # canonical entry -> node

    # -- module access -----------------------------------------------------

    def module(self, path):
        mod = self.modules.get(path)
        if mod is not None:
            return mod
        tree = _parse(path)
        if tree is None:
            return None
        mod = _build_index(path, tree)
        self.modules[path] = mod
        return mod

    def module_for(self, dotted):
        """The _Module for a dotted import name, if its source is in-project.

        NOTHING IS IMPORTED TO ANSWER THIS. `importlib.util.find_spec` was tried
        and rejected: finding a DOTTED name imports its parent packages, so
        resolving one import during key computation executed vllm's package
        `__init__` and activated the HPU platform plugin -- a side effect in a
        function whose whole job is to read source. Two side-effect-free sources
        are used instead:

          1. `sys.modules` -- and this is also the CORRECT source: a callee whose
             module is not imported cannot be called at runtime, so a module the
             region really calls is necessarily already loaded.
          2. a path search over `sys.path` for the remaining names (a module that
             is on disk but not yet imported), which touches only the filesystem.
        """
        if not dotted:
            return None
        if dotted in self.mod_for:
            return self.mod_for[dotted]
        found = self._module_for_uncached(dotted)
        self.mod_for[dotted] = found
        return found

    def _module_for_uncached(self, dotted):
        path = _module_file_for(dotted)
        if not path or not path.endswith(".py"):
            return None
        if not _in_project(path):
            return None                     # external: opaque by design
        return self.module(path)

    # -- class members -----------------------------------------------------

    def class_member(self, mod, qualpath, attr, seen=None):
        """[(mod, containing qualpath, name)] for `attr` on the class at `qualpath`."""
        seen = set() if seen is None else seen
        key = (mod.relpath, qualpath, attr)
        if key in seen:
            return []
        seen.add(key)
        out = []
        if attr in mod.members(qualpath):
            out.append((mod, qualpath, attr))
        for base_mod, base_qp in self.bases(mod, qualpath):
            out.extend(self.class_member(base_mod, base_qp, attr, seen))
        return out

    def bases(self, mod, qualpath):
        """[(mod, qualpath)] for the in-project bases of the class at `qualpath`."""
        key = (mod.relpath, qualpath)
        if key in self.bases_of:
            return self.bases_of[key]
        found = self._bases_uncached(mod, qualpath)
        self.bases_of[key] = found
        return found

    def _bases_uncached(self, mod, qualpath):
        node = mod.classes.get(qualpath)
        if node is None:
            return []
        out = []
        for base in node.bases:
            if not isinstance(base, ast.Name):
                continue                     # `nn.Module`: external, opaque
            for target in self.bare_targets(mod, base.id):
                target_mod, target_qp, target_name = target
                member = _symbol_node(target_mod, target_qp, target_name)
                if isinstance(member, ast.ClassDef):
                    out.append((target_mod, target_qp + (target_name,)))
        return out

    def chain_in_class(self, mod, start_qp, parts):
        """Walk `parts` from the class at `start_qp`.

        A position is (module, class qualpath): the class whose namespace the next
        part is looked up in. Three things can happen to a part:
          * it is a member (method, nested class, class-level assignment) -> a final
            entry when it is the last part, otherwise a new position when it is a
            class, so `self.Inner.method` resolves;
          * it is assigned in `__init__` to an in-project class -> a new position,
            so `self.inner.forward` resolves to `Inner.forward`; a final entry when
            it is the last part, since the class itself is then the code;
          * neither -> this position yields nothing.
        """
        current = [(mod, start_qp)]
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            finals = []
            following = []
            for this_mod, qp in current:
                for member_mod, member_qp, member_name in self.class_member(
                        this_mod, qp, part):
                    if last:
                        finals.append((member_mod, member_qp, member_name))
                    else:
                        node = _symbol_node(member_mod, member_qp, member_name)
                        if isinstance(node, ast.ClassDef):
                            following.append((member_mod, member_qp + (member_name,)))
                for inst_mod, inst_qp in self.init_instance(this_mod, qp, part):
                    if last:
                        finals.append((inst_mod, inst_qp[:-1], inst_qp[-1]))
                    else:
                        following.append((inst_mod, inst_qp))
            if last:
                return finals
            if not following:
                return []
            current = following
        return []

    def init_assignment(self, mod, qualpath, name):
        """(module, RHS) for the first `self.<name> = <expr>` in __init__, or None.

        The RHS is returned unresolved, because the two callers want different
        things from it: `init_instance` asks "which in-project class?", and the
        resolver asks "is this attribute a runtime input at all?".

        BOUNDED ON PURPOSE. Only the FIRST assignment in `__init__` is considered,
        and the `__init__` may be INHERITED (returned with the module that writes
        it, since the RHS must resolve in the writing module's namespace).
        """
        key = (mod.relpath, qualpath, name)
        if key in self.init_assign:
            return self.init_assign[key]
        self.init_assign[key] = None        # cycle guard: never revisit
        out = None
        constructor = self.init_node(mod, qualpath)
        if constructor is not None:
            init_mod, init = constructor
            for stmt in ast.walk(ast.Module(body=list(init.body), type_ignores=[])):
                value = None
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if _is_self_attr(target, name):
                            value = stmt.value
                            break
                elif isinstance(stmt, ast.AnnAssign) and _is_self_attr(stmt.target, name):
                    value = stmt.value
                if value is not None:
                    # FIRST assignment only: a later reassignment is not analysed.
                    out = (init_mod, value)
                    break
        self.init_assign[key] = out
        return out

    def init_instance(self, mod, qualpath, name):
        """[(module, class qualpath)] for `self.<name> = <expr>` in this __init__.

        Only three forms resolve: a call to an in-project class (`Inner()`), an
        in-project class attribute, and a module-qualified in-project name. No
        dataflow, no return-value tracking. Anything else (a parameter, a factory
        function, a subscript, a literal) yields nothing -- and nothing is not a
        failure: the resolver classifies it as a RUNTIME input.
        """
        key = (mod.relpath, qualpath, name)
        cached = self.instance_of.get(key)
        if cached is not None:
            return cached
        self.instance_of[key] = []          # cycle guard: never revisit
        assignment = self.init_assignment(mod, qualpath, name)
        out = []
        if assignment is not None:
            out = self.class_of_expr(assignment[0], assignment[1])
        self.instance_of[key] = out
        return out

    def init_node(self, mod, qualpath):
        """(module, __init__ node) for instances of this class: own, else inherited.

        A class that does not define `__init__` is constructed by its base's, so
        the assignment that binds `self.<attr>` may live in another file. That
        module is returned with the node because the right-hand side must be
        resolved in the namespace it was written in.
        """
        own = mod.members(qualpath).get("__init__")
        if isinstance(own, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return mod, own
        for base_mod, base_qp in self.bases(mod, qualpath):
            found = self.init_node(base_mod, base_qp)
            if found is not None:
                return found
        return None

    def class_of_expr(self, mod, expr):
        """[(module, class qualpath)] when `expr` denotes an in-project class."""
        out = []
        if isinstance(expr, ast.Call):
            expr = expr.func
        if isinstance(expr, ast.Name):
            for target_mod, target_qp, target_name in self.bare_targets(mod, expr.id):
                node = _symbol_node(target_mod, target_qp, target_name)
                if isinstance(node, ast.ClassDef):
                    out.append((target_mod, target_qp + (target_name,)))
        elif isinstance(expr, ast.Attribute):
            parts = []
            base = expr
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name):
                for target_mod, target_qp, target_name in self.attr_targets(
                        mod, base.id, tuple(reversed(parts))):
                    node = _symbol_node(target_mod, target_qp, target_name)
                    if isinstance(node, ast.ClassDef):
                        out.append((target_mod, target_qp + (target_name,)))
        return out

    def walk_from_module(self, dotted, parts):
        """Resolve `parts` starting at module `dotted`."""
        if not parts:
            return []
        mod = self.module_for(dotted)
        if mod is None:
            return []
        head, rest = parts[0], parts[1:]
        sub = self.module_for(dotted + "." + head)
        if sub is not None:
            return self.walk_from_module(dotted + "." + head, rest) if rest else []
        node = _symbol_node(mod, (), head)
        if node is None:
            return []
        if rest:
            if isinstance(node, ast.ClassDef):
                return self.chain_in_class(mod, (head,), rest)
            return []
        return [(mod, (), head)]

    # -- name resolution ---------------------------------------------------

    def bare_targets(self, mod, name, depth=0):
        """Where a bare `name` is bound: [(mod, qualpath, name)]. Never raises."""
        if depth > 8:
            return []
        out = []
        for kind, payload in mod.bindings.get(name, ()):
            if kind == "def":
                out.append((mod, (), name))
            elif kind == "node":
                out.append((mod, (), name))
            elif kind == "import":
                dotted, symbol = payload
                if symbol is None:
                    # `import a.b` binds a module object; a module is not a callee.
                    continue
                if not dotted:
                    # A relative import whose package could not be named. Skipped
                    # rather than guessed: an unnameable package is library code
                    # until proven otherwise.
                    continue
                source_mod = self.module_for(dotted)
                if source_mod is not None:
                    # Ask the source module for the symbol WITHOUT pre-checking
                    # `bindings`: `vllm/distributed/__init__.py` is nothing but
                    # `from .communication_op import *`, so its names are bound by
                    # a star import and never appear in `bindings` directly. The
                    # pre-check skipped them, and a real region that calls
                    # `tensor_model_parallel_all_reduce` then crashed with "no
                    # binding anywhere" -- found by replaying the boot failure.
                    found = self.bare_targets(source_mod, symbol, depth + 1)
                    if found:
                        out.extend(found)
                        continue
                    sub_mod = self.module_for(dotted + "." + symbol)
                    if sub_mod is not None:
                        out.append((sub_mod, (), symbol))   # `from pkg import sub`
                    continue
                # The parent module is external, so nothing under it is project
                # source either: opaque by design.
                continue
        if not out:
            for dotted in mod.stars:
                star_mod = self.module_for(dotted)
                if star_mod is not None and name in star_mod.bindings:
                    out.extend(self.bare_targets(star_mod, name, depth + 1))
        return out

    def attr_targets(self, mod, root, attrs):
        """Resolve `root.attrs...` where the base is statically known."""
        out = []
        for kind, payload in mod.bindings.get(root, ()):
            if kind == "def":
                node = mod.members(()).get(root)
                if isinstance(node, ast.ClassDef):
                    out.extend(self.chain_in_class(mod, (root,), attrs))
            elif kind == "import":
                dotted, symbol = payload
                if symbol is None:
                    out.extend(self.walk_from_module(dotted, attrs))
                else:
                    sub = self.module_for((dotted + "." + symbol) if dotted else symbol)
                    if sub is not None:
                        out.extend(self.walk_from_module(
                            (dotted + "." + symbol) if dotted else symbol, attrs))
                    else:
                        source_mod = self.module_for(dotted)
                        if source_mod is not None:
                            node = _symbol_node(source_mod, (), symbol)
                            if isinstance(node, ast.ClassDef) and attrs:
                                out.extend(self.chain_in_class(
                                    source_mod, (symbol,), attrs))
                            elif not attrs:
                                out.append((source_mod, (), symbol))
        return out

    # -- the fixpoint ------------------------------------------------------

    def resolve(self, mod, qualpath, kind, payload, lineno, is_call, locals_):
        """Targets for one reference, or [] for opaque, or raise for unbound."""
        if kind == "attr":
            root, attrs = payload
            if root in ("self", "cls") and qualpath:
                found = self.chain_in_class(mod, qualpath, attrs)
                if found:
                    return found
                if is_call:
                    # A call receiver that did not resolve. Two very different
                    # cases, and conflating them is what broke serving:
                    #
                    # (a) THE FIRST ELEMENT RESOLVES but a later one does not.
                    #     A member name is wrong. That is a typo, not an input.
                    # (b) THE FIRST ELEMENT ITSELF is assigned in __init__ to
                    #     something we cannot resolve statically (a parameter, a
                    #     factory call, a literal). Its value is fixed at
                    #     construction, so what `self.quant_method.embedding(...)`
                    #     executes is a RUNTIME INPUT, not source. No source hash
                    #     can cover it and skipping it silently would be unsound,
                    #     so it is RECORDED and captured in the key's `instance`
                    #     field instead (see `_instance_attributes`).
                    # (c) nothing anywhere binds it: a typo. Fatal.
                    first = attrs[0]
                    if not self.chain_in_class(mod, qualpath, (first,)):
                        if self.init_assignment(mod, qualpath, first) is not None:
                            self.runtime_dispatched.add(first)
                            self.runtime_sites.append(
                                f"{'.'.join(qualpath)}.{first} at "
                                f"{mod.relpath}:{lineno}")
                            return []
                        # INSTALLED FROM OUTSIDE THE CLASS. A `<expr>.<attr> = ...`
                        # anywhere in project source makes the attribute a runtime
                        # input: hpu_model_runner.py:8048 assigns
                        # `adapter._hpu_raw_process_metadata = ...` onto the adapter
                        # instance and `HpuModelAdapter.fused_entry_prep` calls it.
                        # Knowing `adapter` is an HpuModelAdapter needs type
                        # inference, which we do not build.
                        #
                        # WHAT THIS COSTS, PLAINLY: typo detection weakens. A
                        # misspelled receiver now crashes only if that name is
                        # never an assignment target anywhere in the project -- so
                        # a typo that collides with any assigned attribute name in
                        # the tree is silently treated as a runtime input. That is
                        # a real loss of strictness, accepted because the
                        # alternative is crashing on code that works.
                        if _assigned_anywhere(mod.path, first):
                            self.runtime_dispatched.add(first)
                            self.runtime_sites.append(
                                f"{'.'.join(qualpath)}.{first} at "
                                f"{mod.relpath}:{lineno} (assigned from outside)")
                            return []
                    chain = root + "." + ".".join(attrs)
                    raise UnresolvedCallee(
                        f"cannot resolve code identity for {self.region_label}: the "
                        f"receiver '{chain}' at {mod.relpath}:{lineno} is not a "
                        f"member of {'.'.join(qualpath)}, not inherited from an "
                        f"in-project base, not assigned in "
                        f"{qualpath[-1]}.__init__, and not an assignment target "
                        f"anywhere in project source. A name bound nowhere is a "
                        f"typo, so this is fatal rather than hashed as if the call "
                        f"were absent.")
                return []                     # a VALUE read: opaque, skipped
            if root in locals_:
                return []                     # attribute on a local/unknown value
            if root in _IMPLICIT_MODULE_NAMES:
                return []
            return self.attr_targets(mod, root, attrs)

        name = payload
        if name in locals_ or name in _IMPLICIT_MODULE_NAMES:
            return []
        import builtins

        targets = self.bare_targets(mod, name)
        if targets:
            return targets
        if hasattr(builtins, name):
            return []                         # builtin: external by definition
        raise UnresolvedCallee(
            f"cannot resolve code identity for {self.region_label}: the name "
            f"'{name}' at {mod.relpath}:{lineno} has no binding anywhere -- not "
            f"local, not module-level in {mod.relpath}, not imported, and not a "
            f"builtin. A required part of the key cannot be set, so this is fatal "
            f"rather than hashed as if the call were absent.")

    def run(self, start_mod, start_qp, start_name):
        """Fixpoint over the inlined set. Returns {canonical entry: node}."""
        queue = [(start_mod, start_qp, start_name)]
        while queue:
            mod, qualpath, name = queue.pop()
            entry = (mod.relpath, qualpath, name)
            if entry in self.seen:
                continue
            node = _symbol_node(mod, qualpath, name)
            if node is None:
                # Only reachable for a callee that resolved to a name with no
                # node (an import alias of a module, say); treat as opaque.
                continue
            self.seen[entry] = node
            if isinstance(node, ast.ClassDef):
                # A class reached as an INSTANCE (`self.inner = Inner()` then
                # `self.inner.forward(x)`) contributes its own body, and the body
                # that actually runs may be INHERITED: Qwen3_5Model defines no
                # `forward`; Qwen3NextModel does. Without this the model forward's
                # digest covered the wrapper and an empty class, and an edit to the
                # real transformer body did not invalidate it (demonstrated on a
                # copy of qwen3_5.py + qwen3_next.py). In-project bases are entries
                # in their own right; external bases (nn.Module) stay opaque.
                for base_mod, base_qp in self.bases(mod, qualpath + (name,)):
                    queue.append((base_mod, base_qp[:-1], base_qp[-1]))
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                      # an assignment: hashed whole
            locals_ = _local_names(node)
            for kind, payload, lineno, is_call in _refs(node):
                for target_mod, target_qp, target_name in self.resolve(
                        mod, qualpath, kind, payload, lineno, is_call, locals_):
                    queue.append((target_mod, target_qp, target_name))
        return self.seen


def _canonical_name(qualpath, name):
    """The hash key for one inlined entry: its QUALIFIED NAME, and nothing else.

    NOT the file. A file rename does not change a lowered graph, so it must not
    invalidate a recipe (PROTOCOL §8); the same argument rules out the absolute
    path, which would also make the digest depend on where the checkout lives.
    Two distinct functions that share a qualname still both contribute: they are
    separate entries in the walk's seen-set, and their dumps are hashed in sorted
    order, so the pair (name, dump) is a total order.
    """
    return ".".join(tuple(qualpath) + (name,))


def code_identity(fn) -> str:
    """hash(normalised AST of fn + every project function inlined into it).

    Normalised: docstrings stripped, no line numbers, no attributes, fields
    unannotated. Two files differing only in comments or whitespace produce the
    same digest, which is the PROTOCOL §8 requirement.

    The inlined set is canonical: entries are keyed on the project-relative path
    and the qualified name, and hashed in sorted order, so the digest does not
    depend on which file the walk started from, on walk order, or on where the
    checkout lives.

    A function whose source cannot be resolved raises `UnnamedCompile`; a callee
    with no binding anywhere raises `UnresolvedCallee`. A callee whose source is
    outside `_PROJECT_ROOTS` is skipped -- library code is identified by version,
    not by source -- so nothing here ever reads torch or the stdlib.
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

    parts = qualname.split(".")
    if "<locals>" in parts:
        raise UnnamedCompile(
            f"cannot resolve code identity for {qualname} at {path}: the function is "
            f"defined in a local scope, so its source cannot be addressed by "
            f"qualified name. Regions must wrap a module-level function or a method.")
    name = parts[-1]
    qualpath = tuple(parts[:-1])

    walk = _Walk(region_label=f"{qualname} at {path}")
    mod = walk.module(path)
    if mod is None:
        raise UnnamedCompile(
            f"cannot resolve code identity for {qualname} at {path}: the file could "
            f"not be read or parsed, so its source cannot be part of the key.")
    if _symbol_node(mod, qualpath, name) is None:
        raise UnnamedCompile(
            f"cannot resolve code identity for {qualname} at {path}: no function or "
            f"method named {name!r} is defined in "
            f"{'.'.join(qualpath) or 'that module'}. (An inherited method carries "
            f"the DEFINING class in __qualname__, so this means the name is wrong.)")

    entries = walk.run(mod, qualpath, name)
    _record_runtime_dispatch(fn, walk)
    return _hash_entries(entries)


def _hash_entries(entries) -> str:
    """blake2b over {(relpath, qualpath, name): node}, canonically ordered.

    Keyed on the QUALIFIED NAME and the normalised AST, never on the file path:
    a rename does not change a lowered graph, so it must not invalidate a recipe
    (PROTOCOL §8), and two identical trees must agree wherever they live.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    rows = []
    for (_relpath, qp, entry_name), node in entries.items():
        normalised = _strip_docstrings(node)
        dump = ast.dump(normalised, annotate_fields=False, include_attributes=False)
        rows.append((_canonical_name(qp, entry_name), dump))
    for canonical, dump in sorted(rows):
        h.update(canonical.encode("utf-8"))
        h.update(b"\0")
        h.update(dump.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# region identity -> the attribute names that walk found to be RUNTIME inputs.
#
# WHY A REGISTRY AND NOT A RETURN VALUE: the names are needed by
# `_instance_attributes`, which `make_record` calls after the identity has been
# obtained -- and for a REGISTERED region the identity comes from `_region_code`,
# computed once at `Region` construction. So the names are recorded where the walk
# already runs, under the same "module:qualname" key the registry uses.
#
# SAME LIFECYCLE AS `_region_code`, AND THAT IS THE HONEST LIMIT: a registered
# region's digest is also fixed at construction, so if a source file changes while
# the process lives, both the digest and this list are from the same (stale)
# moment. A stale list can only produce a key that MISSES, never a false hit,
# because the stale list travels with the stale digest.
_runtime_dispatch: dict[str, tuple] = {}


def _record_runtime_dispatch(fn, walk) -> None:
    names = tuple(sorted(walk.runtime_dispatched))
    qualname = getattr(fn, "__qualname__", "")
    module = getattr(fn, "__module__", None)
    if module:
        _runtime_dispatch[f"{module}:{qualname}"] = names
    _runtime_dispatch.setdefault(qualname, names)
    if names:
        logger.info("runtime-dispatched receivers for %s: %s (%s)", qualname,
                    ", ".join(names), "; ".join(walk.runtime_sites))


def runtime_attrs(fn) -> tuple:
    """The attribute names that this region's walk classified as runtime inputs."""
    qualname = getattr(fn, "__qualname__", "")
    module = getattr(fn, "__module__", None)
    if module:
        hit = _runtime_dispatch.get(f"{module}:{qualname}")
        if hit is not None:
            return hit
    return _runtime_dispatch.get(qualname, ())


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


def key_for_region(name: str, *, shapes: tuple, ints: tuple, instance: tuple) -> str:
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
    # THE SPECIALISATION PARAMETERS ARE THE POINT OF THE KEY. Without them one
    # region produces one key for every distinct graph it compiles, the second
    # graph loads the first's recipe, and the launch dies with
    # `synStatus 30 [Sections' validation failure]`.
    return build(
        instance=tuple(instance),
        region=(("name", name), ("code", region_code(name))),
        shapes=tuple(shapes) + (("ints", tuple(ints)),),
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


class UnresolvedCallee(UnnamedCompile):
    """A callee with no binding anywhere. A required part of the key is unset.

    Raised rather than skipped: hashing a function as if the call were absent
    produces a digest that claims to describe a graph it never looked at. The
    message names the region, the unresolved name, and the file:line of the call
    site, because the only useful response is to fix the call.
    """


# ---------------------------------------------------------------------------
# The trace-free record
# ---------------------------------------------------------------------------

# Module attributes dynamo guards on. Read with getattr, never from a trace.
_GUARDED_ATTRS = ("_ACTIVE", "_hpu_cache_group_index", "compact_state_group_offset",
                  "num_query_per_req", "block_size")


def _tensor_signature(t):
    """(shape, dtype, stride) for one tensor, or raise."""
    try:
        shape = tuple(int(d) for d in t.shape)
        dtype = str(t.dtype)
        stride = tuple(int(s) for s in t.stride())
    except Exception as exc:  # noqa: BLE001 - re-raised with the tensor named
        raise RuntimeError(
            f"cannot read the signature of {t!r} for the compile key: {exc!r}. "
            f"Shape, dtype and stride are inputs to the key, so they must not be "
            f"omitted.") from exc
    return ("tensor", shape, dtype, stride)


def _signature_of(value, where: str):
    """One entry per argument. Tensors give shape/dtype/stride, ints give the value.

    NESTED CONTAINERS ARE FLATTENED. `aux_hidden_states` is a list of five tensors
    and is one of the regions that matters most; treating a list as opaque would
    put an empty entry in the key exactly where the specialisation lives.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return [_tensor_signature(value)]
    if isinstance(value, bool):
        return [("bool", value)]
    if isinstance(value, int):
        return [("int", value)]
    # TYPES THAT GENUINELY CHANGE THE LOWERED GRAPH. Each gets a STABLE CANONICAL
    # form -- `str(dtype)`, `str(device)`, the string itself -- never `repr` of an
    # object carrying an address, which would change the key every process.
    if isinstance(value, torch.dtype):
        return [("dtype", str(value))]
    if isinstance(value, torch.device):
        return [("device", str(value))]
    if isinstance(value, float):
        return [("float", repr(value))]
    if isinstance(value, str):
        return [("str", value)]
    # SYMBOLIC SCALARS. The model forward runs with backed dynamic shapes, so its
    # placeholders carry `SymInt` rather than `int`. A SymInt genuinely selects a
    # graph -- it is a dimension -- so it must be keyed, in a canonical form.
    # `str()` is used rather than forcing `int()`, which would specialise the
    # symbol and defeat the dynamic-shape machinery.
    if type(value).__name__ in ("SymInt", "SymFloat", "SymBool"):
        return [(type(value).__name__.lower(), str(value))]
    if value is None:
        # `None` IS A VALUE dynamo guards on. Recorded explicitly so it stays
        # distinguishable from an argument that is absent entirely.
        return [("none", None)]
    if isinstance(value, (list, tuple)):
        out = []
        for i, item in enumerate(value):
            out.extend(_signature_of(item, f"{where}[{i}]"))
        return out
    if isinstance(value, dict):
        out = []
        for k in sorted(value, key=repr):
            out.extend(_signature_of(value[k], f"{where}[{k!r}]"))
        return out
    # Anything else cannot be a compilation input we understand.
    raise RuntimeError(
        f"cannot key on argument {where} of type {type(value).__name__}. Only "
        f"tensors, ints, bools and containers of those are declared inputs; "
        f"silently dropping this argument would produce a key that omits it.")


# id(model) -> (model, locations). Keyed on the model OBJECT's identity, not a
# timestamp: the model is constructed once per process, so process lifetime is the
# right bound, and a rebuilt model gets a fresh id.
_LOCATION_CACHE: dict = {}


def _walk_modules(root):
    """(qualified_path, module) for the root and every descendant module."""
    yield "", root
    seen = {id(root)}
    stack = [("", root)]
    while stack:
        path, mod = stack.pop()
        for name, child in list(getattr(mod, "_modules", {}).items()):
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            child_path = f"{path}.{name}" if path else name
            yield child_path, child
            stack.append((child_path, child))


def _resolve_path(root, path):
    """`getattr` down a dotted path from `root`."""
    mod = root
    for part in path.split("."):
        if not part:
            continue
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


def _attribute_locations(model, names=_GUARDED_ATTRS) -> tuple:
    """Every (path, attribute) occurrence of a guarded attribute in the model tree.

    BUILT ONCE PER MODEL OBJECT and cached on its identity. The walk is the
    expensive part and the model's STRUCTURE does not change after construction;
    only the values do, which is read per compile through these paths.

    ALL LAYERS ARE INCLUDED, not just the ones a guard has named. Over-coarse costs
    a spurious miss; under-coarse gives two different graphs one key, which is the
    `synStatus 30` this whole thread exists to remove. That is a deliberate §8
    trade: a shared entry would violate "invalidate only if the recipe changed",
    and safety wins.
    """
    key = (id(model), tuple(names))
    hit = _LOCATION_CACHE.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    locations = []
    modules = 0
    for path, mod in _walk_modules(model):
        modules += 1
        for name in names:
            if hasattr(mod, name):
                locations.append((path, name))
    locations = tuple(locations)
    logger.info("attr-locations: %d paths over %d modules (%d names)",
                len(locations), modules, len(names))
    _LOCATION_CACHE[key] = (model, locations)
    return locations


def _project_root_dirs(path: str) -> tuple:
    """The project source trees this file belongs to, found by path only.

    Walks up from the file and, at each ancestor, takes the ancestor itself and any
    immediate child whose NAME is a project root. That is enough to find the file's
    own tree AND its siblings -- from `<...>/gaudi2-single-card-vllm/vllm-gaudi/x.py`
    it finds `vllm-gaudi`, its sibling `vllm-core`, and one level further up the
    bridge tree -- without importing anything or guessing beyond `listdir`.
    """
    try:
        current = os.path.dirname(os.path.abspath(path))
    except OSError:
        return ()
    found = []
    seen = set()
    while True:
        candidates = []
        if os.path.basename(current) in _PROJECT_ROOTS:
            candidates.append(current)
        try:
            for child in os.listdir(current):
                if child in _PROJECT_ROOTS:
                    candidates.append(os.path.join(current, child))
        except OSError:
            pass
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return tuple(found)


# root set -> {attribute name: (files that mention it as a possible target, ...)}
_ASSIGNMENT_INDEX: dict[tuple, dict] = {}

# A CANDIDATE FILTER, not the answer. These match any place a name could appear
# after a dot in a position that assigns it, plus `setattr(obj, "name", ...)`.
# Over-matching is deliberate and harmless: a candidate is CONFIRMED by parsing it
# (see `_is_attribute_target`), so the answer for the name actually asked about is
# exact while the index is only a superset.
_ATTR_TARGET_RE = (
    rb"\.\s*([A-Za-z_][A-Za-z_0-9]*)\s*"
    rb"(?::[^=\n]*|,|(?:\*\*|//|>>|<<|[-+*/%&|^@])?=(?!=))")
def _project_assignment_index(root_dirs) -> dict:
    """{attribute name: (files that might assign it)} for these project trees.

    BUILT ONCE PER PROCESS PER ROOT SET, and only when a receiver would otherwise
    be declared a typo, so the common path pays nothing.

    WHY NOT AN AST SET OF ALL NAMES, which is what I wrote first: parsing every
    project file to collect the exact set costs 29 s COLD on this checkout (6,455
    files), measured, because one region's first typo would pay for the whole tree.
    A byte pass costs 0.4 s for the same trees, so the index is byte-built and the
    answer is confirmed by parsing only the handful of files that mention the name
    -- usually one. Same result, 1/70th of the cost.

    A MISSED CANDIDATE IS STILL A MISSED ASSIGNMENT, so the patterns deliberately
    over-match: `x.attr =`, augmented and annotated forms, tuple targets (`,`) and
    `setattr(x, "attr", ...)`. `del x.attr` is covered by the `,`-less form only
    through `_is_attribute_target`, which sees Del contexts in the files the index
    proposes; a name assigned ONLY by `del` would be missed. That is stated rather
    than hidden.
    """
    key = tuple(sorted(root_dirs))
    hit = _ASSIGNMENT_INDEX.get(key)
    if hit is not None:
        return hit
    import re

    attr_re = re.compile(
        rb"\.\s*([A-Za-z_][A-Za-z_0-9]*)\s*"
        rb"(?::[^=\n]*|,|(?:\*\*|//|>>|<<|[-+*/%&|^@])?=(?!=))")
    setattr_re = re.compile(
        rb"setattr\s*\(\s*[^,()]+,\s*[\"']([A-Za-z_][A-Za-z_0-9]*)[\"']")
    index: dict[str, list] = {}
    files = 0
    for root in key:
        for dirpath, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs
                             if d not in ("__pycache__", ".git", "venv",
                                          "site-packages", "node_modules"))
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "rb") as stream:
                        blob = stream.read()
                except OSError:
                    continue
                files += 1
                for match in attr_re.finditer(blob):
                    index.setdefault(match.group(1).decode("ascii", "replace"),
                                     []).append(path)
                for match in setattr_re.finditer(blob):
                    index.setdefault(match.group(1).decode("ascii", "replace"),
                                     []).append(path)
    out = {name: tuple(sorted(set(paths))) for name, paths in index.items()}
    _ASSIGNMENT_INDEX[key] = out
    logger.info("project assignment index: %d names from %d files in %d root(s)",
                len(out), files, len(key))
    return out


def _is_attribute_target(path: str, name: str) -> bool:
    """Whether `path` really assigns `path.<name>` (or setattr's it). EXACT.

    This is the confirmation step: the index only proposes files, this decides.
    """
    tree = _parse(path)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == name
                and isinstance(node.ctx, (ast.Store, ast.Del))):
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "setattr" and len(node.args) >= 2):
            argument = node.args[1]
            if isinstance(argument, ast.Constant) and argument.value == name:
                return True
    return False


def _assigned_anywhere(region_path: str, name: str) -> bool:
    """Whether `name` is an attribute assignment target anywhere in project source."""
    index = _project_assignment_index(_project_root_dirs(region_path))
    for candidate in index.get(name, ()):
        if _is_attribute_target(candidate, name):
            return True
    return False


def _class_source_digest(cls) -> str:
    """Hash of a class's IN-PROJECT source: its body and its in-project bases.

    "external" when the class is not project source -- library code is identified
    by version, which is the existing rule, and nothing here reads it.
    "nosource" when it IS project source but cannot be located in its module (built
    by `type(...)`, or defined in a local scope). That string is a recorded value,
    so a class BECOMING locatable is itself a key change.
    """
    module_name = getattr(cls, "__module__", None)
    if not module_name:
        return "external"
    import sys

    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None) if module is not None else None
    if not path:
        path = _module_file_for(module_name)
    if not path or not path.endswith(".py") or not _in_project(path):
        return "external"
    qualname = str(getattr(cls, "__qualname__", ""))
    parts = qualname.split(".")
    if not qualname or "<locals>" in parts:
        return "nosource"
    walk = _Walk(region_label=f"{module_name}:{qualname}")
    mod = walk.module(path)
    if mod is None:
        return "nosource"
    entries = walk.run(mod, tuple(parts[:-1]), parts[-1])
    return _hash_entries(entries) if entries else "nosource"


def _type_name(value) -> str:
    """`module:qualname` of a value's TYPE. Never `repr`."""
    cls = type(value)
    return (f"{getattr(cls, '__module__', '') or ''}:"
            f"{getattr(cls, '__qualname__', '') or ''}")


def _owner_name(owner) -> str:
    """`module:qualname` for the object a method is bound to.

    For a classmethod the owner IS a class, and naming `type(owner)` would say
    `builtins:type` -- which identifies nothing.
    """
    if isinstance(owner, type):
        return (f"{getattr(owner, '__module__', '') or ''}:"
                f"{getattr(owner, '__qualname__', '') or ''}")
    return _type_name(owner)


def _function_source_digest(fn) -> str:
    """Hash of a function's IN-PROJECT source, including what it inlines.

    Same machinery as `code_identity` (the walk and `_hash_entries`), so a change
    anywhere in the callable's body -- or in anything it calls -- moves it.
    "external" for a C function, a slot wrapper, or any file outside the project
    roots: library code is identified by version, and nothing reads it.
    "nosource" for a project function that cannot be addressed by qualified name
    (one defined in a local scope); that string is a recorded value.

    An `UnresolvedCallee` raised by the walk is NOT caught: if the callable's own
    body cannot be identified, a required part of the key cannot be set.
    """
    if fn is None:
        return "external"
    path = getattr(getattr(fn, "__code__", None), "co_filename", None)
    qualname = getattr(fn, "__qualname__", None)
    if not path or not qualname:
        return "external"
    if not _in_project(path):
        return "external"
    parts = qualname.split(".")
    if "<locals>" in parts:
        return "nosource"
    walk = _Walk(region_label=f"{qualname} at {path}")
    mod = walk.module(path)
    if mod is None:
        return "nosource"
    entries = walk.run(mod, tuple(parts[:-1]), parts[-1])
    return _hash_entries(entries) if entries else "nosource"


def _type_identity(value, cache=None) -> str:
    """A stable identity for an object-valued attribute: its type, and its source.

    NEVER `repr`: that embeds an address and would change the key every process.
    The type's in-project SOURCE is included because the type name alone does not
    invalidate when the implementation changes -- `UnquantizedEmbeddingMethod`'s
    body decides what `self.quant_method.embedding(...)` executes, and a rename of
    nothing is not what changes. External types contribute their name only
    (`+external`), because external code is identified by version.
    """
    cls = type(value)
    key = id(cls)
    if cache is not None and key in cache:
        return cache[key]
    ident = (f"{getattr(cls, '__module__', '') or ''}:"
             f"{getattr(cls, '__qualname__', '') or ''}"
             f"+{_class_source_digest(cls)}")
    if cache is not None:
        cache[key] = ident
    return ident


def _instance_value(value, cache=None, depth=0):
    """A STABLE signature for one guarded attribute's value.

    Scalars record their VALUE: an int that selects a graph is an input. Objects
    record their TYPE (see `_type_identity`), never `repr`. A TENSOR keeps the
    module's tensor signature -- shape, dtype, stride -- because it is stable and
    says more than "torch:Tensor" does.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("none", None)
    # torch is CONSULTED, never imported here: a plain object attribute must not
    # drag torch (and the HPU autoload shim behind it) into a process to answer a
    # question about its type name. If torch is not loaded, no value in this
    # process can be a torch value, so the checks are correctly skipped.
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        if isinstance(value, torch.dtype):
            return ("dtype", str(value))
        if isinstance(value, torch.device):
            return ("device", str(value))
        if isinstance(value, torch.Tensor):
            return _tensor_signature(value)
    # CALLABLE VALUES ARE IDENTIFIED BY THE CALLABLE, NOT BY `type(value)`.
    # `self._hpu_raw_process_metadata = adapter.metadata_processor.process_metadata`
    # installs a BOUND METHOD, and `type(value)` is `builtins:method` for EVERY
    # bound method in the process: it discriminates nothing, so two different
    # implementations would share a key. A bound method is identified by its owner
    # type, its function's qualname, and that function's source.
    func = getattr(value, "__func__", None)
    owner = getattr(value, "__self__", None)
    if func is not None and owner is not None:
        return ("method",
                f"{_owner_name(owner)}.{getattr(func, '__qualname__', '?')}"
                f"+{_function_source_digest(func)}")
    if (getattr(value, "__code__", None) is not None
            and getattr(value, "__qualname__", None)):
        return ("function",
                f"{getattr(value, '__module__', '') or ''}:{value.__qualname__}"
                f"+{_function_source_digest(value)}")
    import functools

    if isinstance(value, functools.partial):
        if depth > 4:
            return ("partial", "deep")
        inner = _instance_value(value.func, cache, depth + 1)
        args = tuple(_instance_value(a, cache, depth + 1) for a in value.args)
        kwargs = tuple(sorted((k, _instance_value(v, cache, depth + 1))
                              for k, v in value.keywords.items()))
        # Every element is already a stable value, so repr() of the tuple is stable
        # too -- there is no address anywhere inside it.
        return ("partial", repr((inner, args, kwargs)))
    if callable(value):
        return ("callable",
                f"{_type_name(value)}+"
                f"{_function_source_digest(getattr(type(value), '__call__', None))}")
    return ("object", _type_identity(value, cache))


def _instance_attributes(module, extra_names=()) -> tuple:
    """The guarded attributes, read through the cached location map.

    `ABSENT` IS A RECORDED VALUE, not a skip. Dynamo guards on absence too --
    `KeyError on self._modules['linear_attn']` is a guard we have seen fire -- so
    "this module has no linear_attn" is a different graph from "it has one".
    """
    if module is None:
        return ()
    names = tuple(_GUARDED_ATTRS) + tuple(
        n for n in extra_names if n not in _GUARDED_ATTRS)
    type_cache = {}          # one source hash per TYPE per call, not per attribute
    out = []
    for path, name in _attribute_locations(module, names):
        obj = _resolve_path(module, path)
        if obj is None or not hasattr(obj, name):
            out.append((f"{path}.{name}" if path else name, "ABSENT"))
            continue
        out.append((f"{path}.{name}" if path else name,
                    _instance_value(getattr(obj, name), type_cache)))
    return tuple(out)


def make_record(fn, args, kwargs, module) -> CompileInputs:
    """The record, built from what the CALLER already holds. No trace.

    `args`/`kwargs` are the real call arguments, so the shapes and ints are the
    ones the compile will actually see. `module` supplies the guarded attributes.

    Uses `build()`, whose inputs are all required -- so a field that cannot be
    produced raises here rather than silently weakening the key.
    """
    shapes = []
    for i, a in enumerate(args):
        shapes.extend(_signature_of(a, f"args[{i}]"))
    for k in sorted(kwargs, key=repr):
        shapes.extend(_signature_of(kwargs[k], f"kwargs[{k!r}]"))
    # THE REGISTRY IS KEYED BY "module:qualname". `Region.__init__` registers
    # under `f"{fn.__module__}:{fn.__qualname__}"`, so looking the registry up
    # under the BARE `__qualname__` never hit: every consult fell through to the
    # source walk (~284 us) and the registry was dead code. Look up the SAME form
    # the registrar used first, then the bare qualname (a caller may have
    # registered under it), and only then compute the identity.
    #
    # CONSEQUENCE, DELIBERATE: for a REGISTERED region the `region` field becomes
    # ("name", "module:qualname") instead of ("name", __qualname__), so every key
    # for those regions changes. That is acceptable only because the recipe cache
    # has not yet been populated under this scheme; it must land before the first
    # populate boot. Unregistered targets keep the old bare-qualname value.
    qualname = getattr(fn, "__qualname__", repr(fn))
    fn_module = getattr(fn, "__module__", None)
    name = f"{fn_module}:{qualname}" if fn_module else qualname
    code = region_code(name)
    if code == "unknown":
        name = qualname
        code = region_code(name)
    if code == "unknown":
        # LEGITIMATE: arbitrary `torch.compile` targets are not Regions and are
        # never registered, so the source walk is the only way to identify them.
        code = code_identity(fn)
    # THE RUNTIME-DISPATCHED ATTRIBUTES COME FROM THE WALK, not from a list.
    # `code_identity` records them under the region's "module:qualname" as it runs,
    # so by here they are known -- whether the identity came from the registry
    # (recorded at Region construction) or from the walk just above.
    return build(
        instance=_instance_attributes(module, runtime_attrs(fn)),
        region=(("name", name), ("code", code)),
        shapes=tuple(shapes),
    )


#: Ordered placeholder signatures per FRAME key, for BINDING verification.
#:
#: WHY THIS EXISTS. The recipe key says which graph a call needs; it does not say
#: which of the caller's arguments goes into which of the graph's inputs. The
#: consult passes `tuple(args)` positionally and the C++ `update_patching_table`
#: binds strictly positionally, so if recipe input i is not the caller's argument
#: i, a launch puts the WRONG tensor in the right place -- silent corruption
#: rather than an error. There is already an artifact whose input is named
#: `%arg1_1` where the region's first parameter is `dest`, so the obvious binding
#: cannot be assumed.
#:
#: The list is the same one the key is built from (`placeholder_signatures`), so
#: the manifest and the key cannot disagree about what the frame expects.
_FRAME_BINDINGS: "dict[str, list]" = {}


def remember_frame_binding(frame_key: str, shapes) -> None:
    """Record the ordered placeholder signatures for `frame_key`.

    Called from `frame_record`, which is the one place that has both the frame's
    placeholders and its key. Keyed by the frame key because that is what the
    bundle (`launch_ledger.take_bundle`) and the manifest carry.
    """
    if not frame_key:
        return
    _FRAME_BINDINGS[frame_key] = list(shapes)


def frame_binding(frame_key: str):
    """Pop the ordered placeholder signatures for `frame_key`, or None.

    POPPED, not read, for the same reason `take_bundle` pops: a second trace of
    the same frame must use its own binding, and a stale read would silently
    verify a call against a previous trace's inputs.

    None means "this process does not know what this frame expects", which the
    consult treats as unverifiable -- a MISS, never a launch.
    """
    if not frame_key:
        return None
    return _FRAME_BINDINGS.pop(frame_key, None)


def placeholder_signatures(gm, name: str, identity: str) -> list:
    """The ordered signature of every placeholder in `gm`, flattened.

    Extracted from `frame_record` so the manifest's binding record and the
    recipe key are produced by ONE function. Two functions would eventually
    disagree, and the disagreement would be a launch with the wrong tensor in
    the wrong slot.
    """
    shapes = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        # THREE NAMES FOR THE SAME FACT. `example_value` is the modern one; older
        # graphs carry `val` or `tensor_meta`. Reading only the first made every
        # weight-loading placeholder fall through to the weak form below, where
        # two frames that both take a tensor called `weight` -- shapes 16384 and
        # 5120 -- produced the SAME key and collided exactly as before.
        value = node.meta.get("example_value")
        if value is None:
            value = node.meta.get("val")
        if value is None:
            value = node.meta.get("tensor_meta")
        if value is None:
            # NO SHAPE, NO KEY. Recording the placeholder's *name* instead is
            # what turned a missing shape into a silent collision; a weak key is
            # indistinguishable from a good one at the cache. Fail here, where
            # the frame can be named, rather than at a launch with an arity
            # mismatch that points nowhere.
            raise RuntimeError(
                f"recipe cache key: frame '{name}' ({identity}) has a placeholder "
                f"'{node.name}' with no example_value, val or tensor_meta, so its "
                f"shape cannot be read. Refusing to build a key that omits it.")
        shapes.extend(_signature_of(value, node.name))
    return shapes


def frame_record(name: str, identity: str, gm) -> CompileInputs:
    """The record for ONE FX frame, from what the frame already holds.

    PER FRAME, NOT PER REGION. One `torch.compile` target can produce many
    frames -- one per inner function dynamo compiles -- and keying them all by
    the *outer* call's arguments makes every frame after the first load the first
    frame's recipe. Measured: key `c34a07e9` was asked for by both

        graph_0115  identity=FusedRMSNorm.py:forward#f5cddcdf
        graph_0116  identity=qwen3_5.py:_resolve_state_indices#3ae6a6e6

    both `region=model.forward`, and the arity mismatch that killed serving
    (`running index 4 mismatch with num_inputs 3`) came from exactly that.

    The frame's own placeholders are the inputs the compile actually sees, so
    they are the inputs the key has to describe.

    COST, stated plainly: this key is computed AFTER tracing. A cache hit
    therefore skips the Synapse compile (2-3 s per graph) but NOT the trace.
    That is a real win, and it is not a trace-free hit -- the trace-free hit was
    given up deliberately when the unit moved from region to frame.
    """
    shapes = placeholder_signatures(gm, name, identity)
    record = build(
        instance=(),
        region=(("name", name), ("code", identity)),
        shapes=tuple(shapes),
    )
    # Remember what this frame expects, under the key this record produces, so
    # the manifest can carry the BINDING and a later boot can verify that the
    # caller's positional arguments still line up with the recipe's inputs.
    # Keyed here because this is the only place that holds both.
    remember_frame_binding(record.key(), shapes)
    return record
