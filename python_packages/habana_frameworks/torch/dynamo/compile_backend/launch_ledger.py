###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified here. The original licence above is Apache-2.0; the changes in this
# file are Copyright Nat Tuck under the GNU AGPL-3.0-only. The conflict is
# recorded rather than hidden, as AGENTS.md requires.
###############################################################################

"""The compile-time half of the launch ledger.

The C++ half (`backend/synapse_helpers/launch_ledger.h`) writes one fixed-size
record per recipe launch, and one row per compiled recipe holding the recipe's
name and the FNV-1a key of that name. That is everything the launch path can
know cheaply. It does not know which *region* a recipe belongs to, because a
region is a dynamo frame and a recipe is what the compiler built for it.

This module supplies the missing half, at compile time:

    recipe_id  ->  (region name, region digest, `foo.N` index)

`recipe_id` is the id `graph_compile` returns. It is recoverable from the
recipe name on the C++ side without any extra plumbing, because
`GraphExecsGroup::generate_graph_index()` encodes it:

    graph_index = 1_000_000 + recipe_id * 1000 + variant

and `HabanaLaunchOpPT::syn_graph_name_` embeds that in the recipe name, e.g.

    .graph_dumps/graph_0004_fused_0_syn_1004000_1004000_5021

so `recipe_id = (1004000 - 1000000) // 1000 = 4`. The join is done at report
time, off the hot path, by `tools/launch_ledger_report.py`.

Numbering is `tools/region_identity.assign_ordinals`, unchanged: recipes are
labelled `foo.1 .. foo.N` ordered by minimum source line. Nothing here invents
a numbering.

Region *identity*: `region_identity.region_id` is the digest of a decorated
function's code object, which needs `@region` at the `torch.compile` call site.
The vLLM plugin has many such call sites and changing them would change the arm
under measurement, so the identity used here is derived from the frame's own
source position instead (file:function of the minimum-line node, plus a digest
of the partitions' structure). Both halves are stable across boots; the report
says which one it used and what the decorator transport would add.

Everything is a no-op unless `VLLM_HPU_LAUNCH_LEDGER` names a directory, the
same variable the C++ half reads.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading

_ENV_DIR = "VLLM_HPU_LAUNCH_LEDGER"
_ENV_REGION_DIR = "VLLM_HPU_REGION_IDENTITY_DIR"
_DEFAULT_REGION_DIR = "/home/nat/Qwen27B/tools/region_identity"

_lock = threading.Lock()
_regions: dict[str, "_Region"] = {}
_ri = None
_ri_tried = False
_warned = set()

# ---- Caller-declared region identity --------------------------------------
#
# A region's identity is normally derived from its FX nodes' meta['stack_trace'].
# That works for a model region, whose source position is the only record of
# where it came from. It does not work for a hand-written region
# (vllm_gaudi.device_regions.Region), because the ops such a region performs are
# minted inside torch: aot_autograd's functionalisation represents an input
# mutation as a fresh `aten.copy_.default` carrying no meta at all, so a region
# that is a bare `dest.copy_(src)` reaches the seal with zero traced nodes and
# is refused. The source of such a region is not lost -- it is the module-level
# function the Region wraps, which has a real `__module__` and `__qualname__`,
# and Region already computes that string as its name.
#
# So Region may *declare* its identity for the duration of its call. This is an
# explicit opt-in by our own code, not a general fallback: a model region never
# sets it, so Requirement 2 is unchanged for every region that does not name
# itself. The declaration is a thread-local stack because compilation is
# re-entrant (a region's compile may reach another region's call).
_declared = threading.local()


def _declared_stack() -> list:
    st = getattr(_declared, "stack", None)
    if st is None:
        st = _declared.stack = []
    return st


def set_declared_identity(name: str, file: str | None = None, line: int | None = None):
    """Declare the identity of the region being compiled on this thread.

    Called by ``vllm_gaudi.device_regions.Region`` immediately before it compiles
    or runs its function. ``file``/``line`` are the wrapped function's own
    definition site, which is the region's real source position. Returns a token
    to pass to :func:`clear_declared_identity`.
    """
    _declared_stack().append((name, file, line))
    return _declared_stack()[-1]


def clear_declared_identity(token) -> None:
    """Undo the matching :func:`set_declared_identity`."""
    st = _declared_stack()
    if st and st[-1] is token:
        st.pop()
        return
    # Tolerate out-of-order teardown (an exception unwound a nested region).
    for i in range(len(st) - 1, -1, -1):
        if st[i] is token:
            del st[i]
            return


def declared_identity():
    """The innermost declared identity on this thread, or None."""
    st = _declared_stack()
    return st[-1] if st else None


def _declared_shape(ri, gm, declared):
    """A RecipeShape for a region that names itself.

    The region's source position is the wrapped function's definition, not its
    FX nodes' traces (there are none -- the ops are minted inside torch). Every
    field is derived from that one position plus the partition's node count, so
    the shape is as stable as the function's own source, which is exactly what
    the numbering needs.
    """
    name, file, line = declared
    f = file or "<declared>"
    ln = int(line) if line and line > 0 else 0
    func = name.rsplit(":", 1)[-1]
    return ri.RecipeShape(
        min_line=ln,
        max_line=ln,
        spread=0,
        distinct_lines=1,
        contiguous=True,
        n_nodes=len(list(gm.graph.nodes)),
        n_sites=1,
        files=(f,),
        funcs=(func,),
        multi_source_nodes=0,
    )


def out_dir() -> str | None:
    d = os.environ.get(_ENV_DIR)
    return d if d else None


def enabled() -> bool:
    return bool(os.environ.get(_ENV_DIR))


def _warn_once(what: str, detail: str) -> None:
    if what in _warned:
        return
    _warned.add(what)
    sys.stderr.write(f"[launch-ledger] {what}: {detail}\n")
    sys.stderr.flush()


class RegionIdentityError(RuntimeError):
    """A compiled region could not supply a name and a source annotation.

    Raised at seal time, so it propagates out of the compiled region's call and
    the compilation fails. That is the point: the `foo.N` numbering is defined by
    source position, so a region without one has no well-defined input, and
    substituting a placeholder is how 75 region identities ended up collapsed
    onto `foo.1` across 83.9 shape variants each.

    A *legitimate* failure is a region whose FX nodes genuinely carry no
    ``meta['stack_trace']`` -- a graph built by hand, or one whose meta was
    stripped by a pass. A *bug in this check* looks different: it would fire on
    a region that does have traces, or it would be
    :class:`RegionIdentityEnvironmentError` below, which is raised when the
    checker itself cannot run. The two are kept as separate types so that a
    broken import is never reported as a nameless region.
    """


class RegionIdentityEnvironmentError(RegionIdentityError):
    """The check could not run. Not a fact about the region."""


def _identity_error(reg: "_Region", missing: str, detail: str) -> RegionIdentityError:
    parts = len(reg.parts)
    nodes = 0
    for _, gm in reg.parts:
        try:
            nodes += len(list(gm.graph.nodes))
        except Exception:  # noqa: BLE001 - evidence only, never fatal
            pass
    return RegionIdentityError(
        f"region {reg.graph_name!r} cannot be identified: missing {missing}.\n"
        f"  missing      : {missing}\n"
        f"  detail       : {detail}\n"
        f"  what is there: {parts} partition(s), {nodes} FX node(s)\n"
        f"  requirement  : a compiled region must carry BOTH a name and a source\n"
        f"                 annotation. The `foo.N` numbering is ordered by source\n"
        f"                 position, so without one it has no well-defined input.\n"
        f"  legitimate?  : this is a region defect -- its nodes carry no\n"
        f"                 meta['stack_trace']. A fault in the check itself would\n"
        f"                 raise RegionIdentityEnvironmentError instead, or would\n"
        f"                 fire on a region whose nodes do have traces.\n"
        f"  to inspect   : PT_HPU_GRAPH_DUMP_MODE=compile\n"
    )


def _region_identity():
    """Import tools/region_identity, or return None and say why once."""
    global _ri, _ri_tried
    if _ri_tried:
        return _ri
    _ri_tried = True
    path = os.environ.get(_ENV_REGION_DIR, _DEFAULT_REGION_DIR)
    try:
        if path not in sys.path:
            sys.path.insert(0, path)
        import region_identity  # noqa: PLC0415

        _ri = region_identity
    except Exception as exc:  # noqa: BLE001 - the ledger must never break a boot
        _warn_once("region_identity unavailable", f"{path}: {exc!r}")
        _ri = None
    return _ri


class _Region:
    """The recipes of one dynamo frame, before and after numbering."""

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.parts: list[tuple[object, object]] = []  # (RecipeRecord, gm)
        self.sealed = False
        self.name = ""
        self.human = ""
        self.region_id = ""
        self.notes: list[str] = []
        self.n_recipes = 0


class Handle:
    """What `get_callable_recipe` hands back to `HabanaGraphModule`."""

    __slots__ = ("graph_name", "index")

    def __init__(self, graph_name: str, index: int) -> None:
        self.graph_name = graph_name
        self.index = index


def _partition_identity(gm, index: int):
    """Name, source annotation and index for one partition, or None.

    Computed here rather than at seal time because this is the only point that
    runs *before* `graph_compile`, and the identity has to be in the C++ side's
    hands before the recipe is compiled -- the disk cache entry is written during
    the compile (`DiskCache::Add` <- `RecipeCacheLRU::Add`), so anything learned
    afterwards never reaches the artifact.
    """
    ri = _region_identity()
    if ri is None:
        return None
    try:
        sites = ri.node_sites(gm)
        if not sites:
            # A region that names itself supplies the identity instead of its
            # FX nodes' stack traces. See `set_declared_identity`.
            declared = declared_identity()
            if declared:
                dname = declared[0]
                digest = hashlib.sha256(ri.structural_digest(gm).encode()).hexdigest()[:8]
                return (f"{dname}#{digest}", dname, index)
            return None
        first = min(sites, key=lambda s: s.line)
        if not first.file or first.line <= 0:
            return None
        human = f"{os.path.basename(first.file)}:{first.func}"
        digest = hashlib.sha256(ri.structural_digest(gm).encode()).hexdigest()[:8]
        return (f"{human}#{digest}", f"{first.file}:{first.line}", index)
    except Exception:  # noqa: BLE001 - Requirement 2 reports the real failure
        return None


def record_partition(graph_name: str, gm) -> "Handle | None":
    """Called once per partition, from `get_callable_recipe`.

    Cheap on purpose: it stores the graph module and numbers nothing. The
    numbering has to wait until the frame's partitions are all known.

    It registers whether or not the ledger is recording, because Requirement 2
    (a region must supply a name and a source annotation) is a property of the
    compilation, not of the measurement.

    Clause 3: it also hands the partition's identity to the C++ side *before*
    the compile, so that the identity is written into the recipe's cache entry
    and survives into a warm boot. Without this the identity only exists in this
    process, which is exactly the side table a warm boot cannot use.
    """
    if graph_name is None:
        return None
    with _lock:
        reg = _regions.get(graph_name)
        if reg is None:
            reg = _regions[graph_name] = _Region(graph_name)
        index = len(reg.parts)
        reg.parts.append((None, gm))
    ident = _partition_identity(gm, index)
    if ident is not None:
        name, source, idx = ident
        try:
            from ._recipe_compiler_C import note_region  # noqa: PLC0415

            note_region(graph_name, name, source, idx)
        except Exception as exc:  # noqa: BLE001
            _warn_once("cannot register the region identity", f"{graph_name}: {exc!r}")
    return Handle(graph_name, index)


def _seal(reg: _Region) -> None:
    """Validate the region identity, then -- when recording -- number it.

    This is where Requirement 2 is enforced, and this is the only place it can
    be: the *name* and the *source annotation* both exist here and nowhere else.
    The C++ side has the graph name (`graph_0004_fused_0_syn_1004000_1004000`)
    and no source at all; the FX side has the `stack_trace` and no name. Only
    this seal holds both.

    There is no `unplaced:` fallback any more. A placeholder is not a fact about
    the system, it is a gap we created: it is what let 75 region identities be
    reported as if they had one recipe each.
    """
    ri = _region_identity()
    if ri is None:
        # NOT a region defect: the checker could not run at all. Kept as a
        # distinct type so a broken import is never read as a nameless region.
        raise RegionIdentityEnvironmentError(
            f"cannot check region {reg.graph_name!r}: tools/region_identity is not "
            f"importable (PT_HPU_REGION_IDENTITY_DIR="
            f"{os.environ.get(_ENV_REGION_DIR, _DEFAULT_REGION_DIR)!r}). "
            f"This is an environment fault, not a region without a source; fix "
            f"the import path. The check cannot be skipped, by design."
        )

    reg.sealed = True

    # ---- Requirement 2: the source annotation -----------------------------
    sites = []
    untraced = []
    for i, (_, gm) in enumerate(reg.parts):
        try:
            part_sites = ri.node_sites(gm)
        except Exception as exc:  # noqa: BLE001
            raise _identity_error(
                reg, "source annotation", f"partition {i}: node_sites raised {exc!r}"
            ) from exc
        if not part_sites:
            untraced.append(i)
        sites.extend(part_sites)

    # A region that names itself (vllm_gaudi.device_regions.Region) supplies its
    # identity instead of its FX nodes' stack traces. Its source is the
    # module-level function it wraps, which is *better* evidence than a line
    # number, and it is set only by our own code -- so a model region with no
    # trace still fails exactly as before. See `set_declared_identity`.
    declared = declared_identity()
    if not sites and declared:
        dname, dfile, dline = declared
        # The name is the declaration; the source annotation is the wrapped
        # function's own definition site, which is where a hand-written region
        # really lives -- its FX nodes carry no trace because the ops are minted
        # inside torch.
        reg.human = dname
        reg.name = dname
        reg.region_id = dname
        if not enabled():
            return
        structures = [ri.structural_digest(gm) for _, gm in reg.parts]
        digest = hashlib.sha256("\0".join(structures).encode()).hexdigest()[:8]
        reg.name = f"{dname}#{digest}"
        reg.region_id = reg.name
        records = []
        slot_of = {}
        for i, (_, gm) in enumerate(reg.parts):
            rec = ri.RecipeRecord(
                region_id=reg.region_id,
                name=reg.name,
                shape=_declared_shape(ri, gm, declared),
                structure=ri.structural_digest(gm),
                discovery_index=i,
                gm=gm,
            )
            records.append(rec)
            slot_of[id(rec)] = i
        numbering = ri.assign_ordinals(records)
        reg.notes.extend(numbering.notes)
        for rec in numbering.ordered:
            reg.parts[slot_of[id(rec)]] = (rec, rec.gm)
        reg.n_recipes = len(records)
        return

    if not sites:
        raise _identity_error(
            reg,
            "source annotation",
            "no FX node in any partition carries meta['stack_trace'] "
            f"(partitions with no trace: {untraced})",
        )

    # ---- Requirement 2: the name -----------------------------------------
    first = min(sites, key=lambda s: s.line)
    human = f"{os.path.basename(first.file)}:{first.func}" if first.file else ""
    if not first.file or first.line <= 0 or not human or human == ":":
        raise _identity_error(
            reg,
            "name",
            f"the earliest source annotation is {first.as_json()!r}, which yields "
            f"no usable name",
        )
    reg.human = human

    if not enabled():
        # Validated; nothing to record. Name the region anyway so that a caller
        # that asks sees the real name rather than an empty string.
        reg.name = human
        reg.region_id = human
        return

    # ---- The numbering (recording only) ----------------------------------
    structures = [ri.structural_digest(gm) for _, gm in reg.parts]
    digest = hashlib.sha256("\0".join(structures).encode()).hexdigest()[:8]
    reg.name = f"{human}#{digest}"
    reg.region_id = reg.name

    records = []
    slot_of = {}
    for i, (_, gm) in enumerate(reg.parts):
        shape = ri.recipe_shape(gm)
        structure = ri.structural_digest(gm)
        rec = ri.RecipeRecord(
            region_id=reg.region_id,
            name=reg.name,
            shape=shape,
            structure=structure,
            discovery_index=i,
            gm=gm,
        )
        records.append(rec)
        slot_of[id(rec)] = i
    numbering = ri.assign_ordinals(records)
    reg.notes.extend(numbering.notes)
    for rec in numbering.ordered:
        reg.parts[slot_of[id(rec)]] = (rec, rec.gm)
    reg.n_recipes = len(records)


def note_recipe(recipe_id, handle: "Handle | None") -> None:
    """Called from `HabanaGraphModule.__call__` once `graph_compile` returned.

    This is the first moment at which a recipe id exists for a partition, and it
    is always after the frame's partitions were created, so it is also the
    moment the numbering can be sealed -- and the moment Requirement 2 is
    checked, whether or not the ledger is recording.
    """
    if handle is None:
        return
    with _lock:
        reg = _regions.get(handle.graph_name)
        if reg is None:
            return
        if not reg.sealed:
            _seal(reg)
        if not enabled():
            # Validated and nothing to write. Release the graph modules: holding
            # every compiled region's FX graph for the process lifetime is a
            # leak this check would otherwise introduce.
            reg.parts.clear()
            return
        if recipe_id is None:
            return
        d = out_dir()
        if d is None:
            return
        rec = reg.parts[handle.index][0] if handle.index < len(reg.parts) else None
        label = getattr(rec, "label", None) if rec is not None else None
        min_line = getattr(getattr(rec, "shape", None), "min_line", None)
        max_line = getattr(getattr(rec, "shape", None), "max_line", None)
        n_nodes = getattr(getattr(rec, "shape", None), "n_nodes", None)
        notes = " | ".join(reg.notes)
        row = "\t".join(
            str(v)
            for v in (
                int(recipe_id),
                label if label is not None else "",
                reg.region_id,
                reg.name,
                min_line if min_line is not None else "",
                max_line if max_line is not None else "",
                n_nodes if n_nodes is not None else "",
                reg.n_recipes,
                handle.index,
                handle.graph_name,
                notes,
            )
        )
    path = os.path.join(d, f"regions.{os.getpid()}.tsv")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (row + "\n").encode())
        finally:
            os.close(fd)
    except Exception as exc:  # noqa: BLE001
        _warn_once("cannot write the region table", f"{path}: {exc!r}")
