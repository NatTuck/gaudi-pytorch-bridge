"""Replay a captured serving run's region calls, at warmup.

WHAT THIS SOLVES, MEASURED (analysis/zero-eager/census-logs/boot-b8.log). The
explicit warm mints the regions it declares, at shapes it reads from the
configuration. That leaves the regions serving actually calls to compile at
serving, because a configuration says which shapes are ALLOWED, not which ones a
call passes:

  hpu_dflash2:_combine_aux_hidden_states   64 misses   recipe records 6 input
                                                       slots, the call supplies 7
  prefix_state_cache:_index_copy_rows      40 misses   recipe records one shape
                                                       list, the call supplies
                                                       [162,48,128,128] and
                                                       [162,10,10240]
  prefix_state_cache:_zeros_rows           10 misses   recipe records 0 slots,
                                                       the call supplies 1
  extension.ops:_dynamic_quant_naive        4 misses   never warmed
  device_regions:_zeros_like                1 miss      never warmed
  spec_launch_regions:_scatter_draft_rows   2 misses    arities 7 and 14
  model.forward                             1 miss      never warmed
  Qwen3NextDecoderLayer.forward             9 compiles at serving shapes

`warm_all`'s own docstring states the rule those misses break: a warm with
assumed shapes "mints a full set of recipes for shapes that never occur, logs
success, and leaves the serving path to compile anyway". No better guess fixes
that. The cure is to let a real serving run SAY what it called
(`recipe_consult`'s `VLLM_WARM_CAPTURE`), and let the next boot's warm call
exactly that.

THE PROPERTY THIS MODULE HAS TO PRESERVE, AND IT IS THE WHOLE DESIGN. A replayed
call must be INDISTINGUISHABLE from the recorded one. The capture writes the
signature `compile_inputs._signature_of` produced -- the same normalisation the
recipe key is built from -- and the synthesis below inverts it exactly: the same
shapes, the same dtypes, the same STRIDES, the same scalars. So `make_record` over
a rebuilt call produces the same key, and the consult that runs is the consult
serving runs. Nothing here computes a signature of its own; a second normaliser
would eventually disagree with the key about the same call, and the disagreement
would be a warm that looks like it worked.

WHERE THE CALL GOES. Through the SAME object the serving path calls -- a
`Region`/`PathCheckedRegion`, or the `torch.compile` wrapper -- supplied by
`resolve`. That matters twice: the consult only runs because it is inside that
object, and a MISS from it mints the permit the trace needs. Calling the region's
undecorated function instead would mint nothing and warm nothing.

WHAT IS NOT `torch.zeros`, AND WHY. A synthesised tensor is `torch.empty`
(`empty_strided` when the recorded stride is not contiguous). `torch.zeros` is an
allocate PLUS a `zero_`, and `zero_` is an eager launch that this project's
census counts -- under `PT_HPU_EAGER_GUARD=strict` it aborts the boot. So the
values are arbitrary bytes.

  CONSEQUENCE, STATED RATHER THAN DISCOVERED LATER: an argument whose VALUE is
  interpreted as an index carries garbage. `_index_copy_rows` and
  `_take_row` can therefore raise (an out-of-bounds index) on data no serving call
  would produce. The recipe is still minted -- the trace and the compile happen
  before the launch that fails -- and the failure is reported here by name. Do not
  "fix" that with a fill: it would put an eager launch back on the path this
  whole exercise exists to clear, and a strict-mode boot would die of it.

WHAT THE REPORT MEANS. `{"replayed": n, "skipped": m, "failed": {name: reason}}`,
plus the consult's own `hits`/`minted` for the same window. `skipped` is a
DUPLICATE -- the same record replayed already, because a capture of a decode loop
holds thousands of identical calls (a blank line is not a record and is not
counted at all). `failed` is everything else, keyed by the region name, first
reason kept, and every entry is logged at ERROR:
a warm that does not match must not be able to look like one that worked. A
failure never propagates -- one region that cannot be replayed must not cost the
warm the other hundred -- but it is never silent either.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

#: The device a synthesised tensor goes on. `hpu` is the serving device; a check
#: that runs without a device stack passes `cpu`, which changes nothing the key
#: sees -- `_signature_of` records shape, dtype and stride, never device.
DEFAULT_DEVICE = "hpu"


class Unreplayable(Exception):
    """One recorded call cannot be rebuilt from its signature. Named, not guessed."""


def _torch():
    """`torch`, imported here rather than at module scope.

    This module is imported by the warm, which runs on a path where torch is
    already up; it is also imported by checks that must not drag the device stack
    in to read a file.
    """
    import torch  # noqa: PLC0415

    return torch


def _dtype(name: str):
    """`"torch.bfloat16"` -> `torch.bfloat16`. Raises rather than guessing."""
    torch = _torch()
    short = str(name).rsplit(".", 1)[-1]
    value = getattr(torch, short, None)
    if not isinstance(value, torch.dtype):
        raise Unreplayable(
            f"{name!r} is not a torch dtype in this process, so a tensor cannot "
            f"be rebuilt with it")
    return value


def _contiguous_stride(shape) -> tuple:
    """The stride `torch.empty(shape)` would give this shape.

    Needed because STRIDE IS IN THE KEY: a recorded non-contiguous argument
    rebuilt as a contiguous tensor produces a different key, and the replay would
    report a miss for a call that is in fact already cached -- a warm that does
    not match, in the other direction.
    """
    stride = []
    accumulator = 1
    for dim in reversed(tuple(shape)):
        stride.append(accumulator)
        accumulator *= dim
    return tuple(reversed(stride))


def _value_of_entry(entry, device: str):
    """The value one signature entry denotes.

    The inverse of `compile_inputs._signature_of`, entry by entry, and it refuses
    anything that inverse cannot express rather than approximating it.
    """
    torch = _torch()
    if not isinstance(entry, (list, tuple)) or not entry:
        raise Unreplayable(f"malformed signature entry {entry!r}")
    kind = entry[0]

    if kind == "tensor":
        if len(entry) != 4:
            raise Unreplayable(f"tensor entry {entry!r} does not carry "
                               f"(shape, dtype, stride)")
        shape = tuple(int(dim) for dim in entry[1])
        dtype = _dtype(entry[2])
        stride = tuple(int(step) for step in entry[3])
        if stride == _contiguous_stride(shape):
            # NEVER `torch.zeros`: an allocate plus a `zero_` eager launch, which
            # the census counts and a strict-mode boot aborts on.
            return torch.empty(shape, dtype=dtype, device=device)
        # A NON-CONTIGUOUS ARGUMENT MUST STAY NON-CONTIGUOUS. Stride is part of
        # the key, so rebuilding it contiguous would produce a key that is not the
        # recorded one.
        return torch.empty_strided(shape, stride, dtype=dtype, device=device)

    if kind == "bool":
        return bool(entry[1])
    if kind == "int":
        return int(entry[1])
    if kind == "float":
        # `_signature_of` records `repr(value)`, so this round-trips exactly --
        # including `nan`, `inf` and the shortest form that names the same double.
        return float(entry[1])
    if kind == "str":
        return str(entry[1])
    if kind == "dtype":
        return _dtype(entry[1])
    if kind == "device":
        return torch.device(entry[1])
    if kind == "none":
        return None
    if kind in ("symint", "symfloat", "symbool"):
        # A symbolic scalar is recorded as its STRING on purpose: `_signature_of`
        # will not force it, because forcing would specialise the symbol and
        # defeat the dynamic-shape machinery. Only a spelling that names a plain
        # Python value can be rebuilt; a real symbol cannot be, and it is refused
        # by name instead of being replaced with a constant.
        text = str(entry[1])
        try:
            if kind == "symint":
                return int(text)
            if kind == "symfloat":
                return float(text)
            return bool(int(text)) if text.isdigit() else text == "True"
        except (TypeError, ValueError):
            raise Unreplayable(
                f"a {kind} recorded as {text!r} is a symbolic value, not a plain "
                f"one, and this replay will not substitute a constant for it")
    raise Unreplayable(f"unknown signature kind {kind!r} in entry {entry!r}")


def _infer_kind(entries) -> str:
    """The container kind for a record written without `kinds`.

    A capture from before `kinds` existed, or one written by hand. One entry means
    the argument was not a container; anything else (including none) means a flat
    container. That is a weaker claim than the recorded `kinds`, so a record that
    carries them always uses them.
    """
    return "scalar" if len(entries) == 1 else "list"


def _value_of_argument(entries, kind, where: str, device: str):
    """One argument, rebuilt from its entries and its recorded container kind."""
    if kind == "dict":
        raise Unreplayable(
            f"{where} was a dict: `_signature_of` folds a dict by sorting its keys "
            f"by `repr` and records only the VALUES, so the mapping cannot be "
            f"rebuilt and this replay will not guess one")
    if kind == "nested":
        raise Unreplayable(
            f"{where} was a container inside a container: the signature flattens "
            f"nested containers, so the grouping is not in the file and rebuilding "
            f"it flat would produce the SAME key with a different call -- the one "
            f"way a replay could be wrong while looking right")
    if kind in (None, "scalar"):
        if len(entries) != 1:
            raise Unreplayable(
                f"{where} is not a container but its signature holds "
                f"{len(entries)} entries")
        return _value_of_entry(entries[0], device)
    if kind == "list":
        # A LIST, not a tuple: `_signature_of` treats the two identically, so the
        # key cannot say which it was, and every container argument in this region
        # set is either a list of tensors or `torch.full`'s shape (which torch
        # accepts as a list). If a region ever needs a tuple it will say so by
        # failing, loudly, rather than by silently getting the wrong type.
        return [_value_of_entry(entry, device) for entry in entries]
    raise Unreplayable(f"{where} has an unknown container kind {kind!r}")


def synthesise_call(record: dict, *, device: "str | None" = None):
    """`(args, kwargs)` for one captured record.

    Raises `Unreplayable` for anything the file does not determine, which the
    caller reports by name.
    """
    if device is None:
        device = DEFAULT_DEVICE
    raw_args = record.get("args") or []
    if not isinstance(raw_args, list):
        raise Unreplayable("'args' is not a list")
    kinds = record.get("kinds")
    if not isinstance(kinds, list) or len(kinds) != len(raw_args):
        kinds = None

    args = []
    for index, entries in enumerate(raw_args):
        if not isinstance(entries, list):
            raise Unreplayable(f"args[{index}] is not a list of entries")
        kind = kinds[index] if kinds is not None else _infer_kind(entries)
        args.append(_value_of_argument(entries, kind, f"args[{index}]", device))

    raw_kwargs = record.get("kwargs") or {}
    if not isinstance(raw_kwargs, dict):
        raise Unreplayable("'kwargs' is not an object")
    kwkinds = record.get("kwkinds")
    if not isinstance(kwkinds, dict):
        kwkinds = {}
    kwargs = {}
    for key, entries in raw_kwargs.items():
        if not isinstance(entries, list):
            raise Unreplayable(f"kwargs[{key!r}] is not a list of entries")
        kind = kwkinds.get(key) or _infer_kind(entries)
        kwargs[key] = _value_of_argument(entries, kind, f"kwargs[{key!r}]", device)
    return tuple(args), kwargs


def _iter_records(path: str):
    """`(lineno, record, reason)` per line. `record` is None when the line is bad."""
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                yield lineno, None, f"not JSON ({exc})"
                continue
            if not isinstance(record, dict):
                yield lineno, None, "not a JSON object"
                continue
            if not isinstance(record.get("region"), str):
                yield lineno, None, "no 'region' name"
                continue
            if not isinstance(record.get("args", []), list):
                yield lineno, None, "'args' is not a list"
                continue
            yield lineno, record, None


def _call_region(target, args: tuple, kwargs: dict):
    """Call the region the way the serving path does.

    `replay_call` when the object has one. `Region` defines it: same consult, same
    dispatch, but a fault raised on synthesised data does not leave the region
    permanently in its eager fallback -- see the note there. Anything without it
    (the `torch.compile` wrappers) is called directly; they have no fallback to
    poison.
    """
    replay = getattr(target, "replay_call", None)
    if callable(replay):
        return replay(*args, **kwargs)
    return target(*args, **kwargs)


def replay_capture(path: str, *, resolve, device: "str | None" = None) -> dict:
    """Replay every distinct call in `path` through `resolve`.

    `resolve(name)` returns the object the serving path calls for that region
    name, or None. None is a FAILURE with a reason, never a skip: the whole point
    is that a region the warm could not replay must not look like one it did.

    Returns `{"replayed", "skipped", "failed", "hits", "minted"}`. Never raises
    for a fault in one record -- a replay failure must not cost the warm the rest
    of the file -- and never raises for a missing file either: it reports it, so
    the caller's warm can finish and the log says what is missing.

    CAPTURE IS SWITCHED OFF FOR THE DURATION. A replayed call is not a serving
    call: recording it would write the warm's own shapes into the file a later
    warm replays, and the replay would be reading lines it was itself appending.
    The previous setting is restored on the way out.
    """
    from . import recipe_consult  # noqa: PLC0415

    report = {"replayed": 0, "skipped": 0, "failed": {}, "hits": 0, "minted": 0}

    if not path:
        return report
    if not os.path.isfile(path):
        report["failed"][path] = "the capture file does not exist"
        logger.error(
            "warm replay: %s does not exist, so NOTHING was replayed. The warm "
            "covered only the shapes it declares, and the serving path will "
            "compile whatever else it calls.", path)
        return report

    if device is None:
        device = DEFAULT_DEVICE

    # EVERY RECORD IS READ BEFORE ANY CALL IS MADE, so a file that is being
    # appended to (by a concurrent capture) cannot feed this loop its own writes,
    # and so a failure halfway through does not leave the rest unread.
    records = list(_iter_records(path))
    hits_before, misses_before = recipe_consult.stats_snapshot()
    seen = set()
    capture_before = recipe_consult.set_capture_path(None)
    if capture_before is not None:
        logger.info(
            "warm replay: capture was appending to %s and is OFF while this replay "
            "runs, so the warm's own calls are not recorded as serving shapes. It "
            "is restored when the replay finishes.", capture_before)

    try:
        _replay_records(records, path, resolve, device, report, seen)
    finally:
        recipe_consult.set_capture_path(capture_before)

    hits_after, misses_after = recipe_consult.stats_snapshot()
    report["hits"] = hits_after - hits_before
    report["minted"] = misses_after - misses_before
    _log_report(path, report)
    return report


def _replay_records(records, path, resolve, device, report, seen) -> None:
    """The loop body of `replay_capture`, so the capture switch can be restored."""
    for lineno, record, reason in records:
        if record is None:
            report["failed"][f"<line {lineno}>"] = reason
            logger.error("warm replay: %s line %d is unusable (%s), so the call it "
                         "described was NOT replayed.", path, lineno, reason)
            continue
        name = record["region"]
        fingerprint = json.dumps(record, sort_keys=True)
        if fingerprint in seen:
            # A duplicate: a capture of a decode loop holds thousands of identical
            # calls, and replaying one of them twice warms nothing new.
            report["skipped"] += 1
            continue
        seen.add(fingerprint)

        try:
            args, kwargs = synthesise_call(record, device=device)
        except Unreplayable as exc:
            _failed(report, name, f"cannot rebuild the call: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - a bad record is not a bad boot
            _failed(report, name,
                    f"cannot rebuild the call: {type(exc).__name__}: {exc}")
            continue

        try:
            target = resolve(name)
        except Exception as exc:  # noqa: BLE001
            _failed(report, name,
                    f"the resolver failed: {type(exc).__name__}: {exc}")
            continue
        if target is None:
            _failed(report, name,
                    "no object is registered under this name in this process, so "
                    "the call cannot be replayed")
            continue

        try:
            _call_region(target, args, kwargs)
        except Exception as exc:  # noqa: BLE001 - one region must not cost the warm
            # THE RECIPE IS USUALLY STILL MINTED: the consult ran, the trace ran,
            # and the launch is what raised. The line says so, because "replayed
            # but the launch failed" and "not replayed at all" call for different
            # responses.
            _failed(report, name,
                    f"{type(exc).__name__}: {exc}".splitlines()[0]
                    + " (the consult and the trace had already run)")
            continue

        report["replayed"] += 1


def _failed(report: dict, name: str, reason: str) -> None:
    """Record and LOG one failure. First reason kept, so a repeated region is
    still one named entry rather than a dict that grows without bound."""
    report["failed"].setdefault(name, reason)
    logger.error("warm replay: region=%s FAILED: %s", name, reason)


def _log_report(path: str, report: dict) -> None:
    """Report the replay. Loud when it did not fully work, always with numbers."""
    failed = report["failed"]
    if failed:
        logger.error(
            "warm replay: %s: replayed=%d skipped=%d FAILED=%d (consult hits=%d "
            "minted=%d). The warm is INCOMPLETE: %s. Every region above is named "
            "with its reason; a region that could not be replayed is still "
            "compiled on the serving path, so this boot is not compile-free.",
            path, report["replayed"], report["skipped"], len(failed),
            report["hits"], report["minted"], ", ".join(sorted(failed)))
        return
    logger.info(
        "warm replay: %s: replayed=%d skipped=%d failed=0 (consult hits=%d "
        "minted=%d). Every distinct call the captured serving run made has been "
        "exercised at warmup.",
        path, report["replayed"], report["skipped"], report["hits"],
        report["minted"])
