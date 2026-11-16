/**
 * Copyright (c) 2021-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified here. The original licence above is Apache-2.0; the changes in this
 * file are Copyright Nat Tuck under the GNU AGPL-3.0-only. The conflict is
 * recorded rather than hidden, as AGENTS.md requires.
 */

// Eager launch guard: make an eager launch name its own Python source, and make
// it impossible to do quietly.
//
//   PT_HPU_EAGER_GUARD = off | census | strict      (default strict)
//   PT_HPU_EAGER_GUARD_DEPTH = N                    (default 0)
//
//     off     no behaviour change: one cached load and a not-taken branch.
//     census  print the Python source of the calling frame, then continue. One
//             boot yields the complete list of eager sites.
//     strict  print the same, then exit non-zero with a diagnostic.
//
// DEPTH, AND WHY IT EXISTS
// ------------------------
// A large class of eager launches is reported at a frame inside torch itself:
// `torch/utils/_device.py:122` in `DeviceContext.__torch_function__`,
// `torch/_tensor.py:1120` in `Tensor.__rdiv__`, `torch/nn/init.py:83` in
// `_no_grad_normal_`. Those frames are torch's own dispatch wrappers. The launch
// is real and the site is real, but the *caller* we could edit is one or more
// frames further up, and the guard's default report hides it. An agent cannot
// fix what it cannot locate.
//
// PT_HPU_EAGER_GUARD_DEPTH=N walks up to N frames back past frames whose
// co_filename is under a `torch/` directory and reports the first frame that is
// not torch's. The torch frame is kept, as a parenthetical, so nothing is lost.
//
// Default 0 preserves the previous behaviour exactly: no walk, the frame is
// reported as-is. The walk changes only WHICH FRAME IS REPORTED. It never
// changes whether a launch is refused: strict stays strict, and an eager launch
// with depth set is still a controlled exit. A launch that walks past every
// frame it has still refuses; it just says so.
//
// The depth is read with plain getenv rather than through the fork's generated
// env-flag machinery. The generated headers are produced by scripts/generate_env.py
// from scripts/env_flags.yaml and require a cmake re-run to regenerate; adding a
// flag there would make this a build-system change for a diagnostic knob. The
// flag is read once into a function-local static, so the cost is one getenv per
// process.
//
// WHERE, AND WHY NOT ONE FRAME LATER
// ----------------------------------
// The brief names the chain
//     EagerExec::launch -> HabanaLaunchOpPT -> RecipeLauncher::Launch
// and asks for a Python frame. That chain has no Python frame at any link, and
// the reason is structural rather than incidental:
//
//   * With PT_HPU_EAGER_PIPELINE_ENABLE (default true), EagerOpBase::run hands
//     the launch to EagerLoweringTask on the *lowering thread*. The Python
//     thread that dispatched the op has already returned to the interpreter.
//   * With the pipeline disabled, EagerOpBase::run constructs an AutoNoGIL and
//     calls Acquire() -- releasing the GIL -- *before* hlexec.launch().
//
// So the last point on the eager path that still has a Python frame is
// EagerOpBase::run, which is also the one function every eager op passes
// through: every EagerOp<>::call() variant funnels into it, and it runs on the
// Python thread with the GIL held. That is the eager dispatch point, and it is
// where this guard is called.
//
// To keep the instrument honest about what it cannot see, the dispatch records
// that it fired (EagerOpMetaData::guard_reported_), and EagerExec::launch --
// the link the brief names -- reports any eager launch that arrives there
// *without* a dispatch site instead of staying silent. That is the difference
// between a census with a known denominator and a census that quietly misses
// rows.
//
// THE ABORT PRECEDES THE COMPILE (why strict means ZERO eager compiles)
// ---------------------------------------------------------------------
// The rule is "eager launches should be crashes, not compiles", so the one
// thing that matters about this hook's position is whether the recipe is
// already minted when it fires. It is not. The chain, with the file and line of
// every link:
//
//   EagerOpBase::run                  habana_eager/ops/eager_op.cpp:103
//     -> on_eager_dispatch            eager_op.cpp:110   <-- FIRST STATEMENT
//        ... strict: std::_Exit(70) here ...
//     -> ScheduleWorkAndUpdateLoweringThreadHandle(EagerLoweringTask, ...)
//                                     eager_op.cpp:140   (pipeline on)
//     or hlexec.launch()              eager_op.cpp:166   (pipeline off)
//     -> EagerExec::launch            habana_eager/eager_exec.cpp:345
//        -> on_launch_without_site    eager_exec.cpp:356   (safety net)
//     -> HabanaLaunchOpPT ctor        eager_exec.cpp:518
//     -> CompileSynapseGraph()        backend/kernel/
//                                     hpu_habana_launch_op_pt_lowering.cpp:761
//                                     (reached from CompileSynapseGraphAndPatch-
//                                      Table, hpu_habana_launch_op_pt.cpp:6122,
//                                      6342, 6592, 7011, ...)
//
// on_eager_dispatch is the first statement of run(): it runs before the stack is
// converted, before the lowering task is scheduled, and before hlexec.launch().
// The compile is five links downstream of it, so in strict mode the process is
// gone before any code that could mint a recipe runs. A strict boot therefore
// produces exactly zero eager compiles -- not "few", zero -- and the only
// recipes it mints are the ones a compiled region asked for.
//
// The one caveat, stated rather than hidden: ops dispatched *before* the first
// eager one have already compiled, and they always will, because an op cannot be
// known to be eager before it is dispatched. That is why the deliverable is a
// boot that dies at its first eager dispatch: the compile it prevents is its
// own, and everything after it.
//
// THE MECHANISM
// -------------
// PyGILState_Ensure + PyThreadState_GetFrame + PyFrame_GetCode. There is no
// cheaper mechanism that produces a Python frame:
//   * an FX node's `meta["stack_trace"]` is the right answer for a *compiled*
//     region and is what Requirement 2 enforces on the Python side, but an
//     eager op has no FX node and no trace -- there is nothing to read;
//   * torch's C++ backtrace helpers (c10::get_backtrace) produce a C++ stack,
//     which is exactly what the brief says is not wanted: it would name
//     habana_kernels/*.cpp, not the Python line that called the op.
// The bridge links plain Python C-API (Python 3.12, no Py_LIMITED_API; the
// python_embed_shim is commented out of pytorch_helpers/CMakeLists.txt), and
// the TUs that include this header already include Python.h.
//
// Header-only on purpose: a new translation unit needs a cmake re-run and this
// build tree's configure-time variables are in neither the environment nor the
// cache (analysis/precompiled-region/REPORT.md section 8).

#pragma once

#include <Python.h>

#include <ATen/ATen.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"

namespace habana::eager_guard {

// Exit status for strict mode. 70 is EX_SOFTWARE in sysexits.h: the process
// stopped because the code refused to continue, not because it crashed.
constexpr int kStrictExitCode = 70;

enum class Mode : int { off = 0, census = 1, strict = 2 };

namespace detail {

// Read once, through the fork's own env-flag machinery so that the value is
// validated by the constraint declared in scripts/env_flags.yaml and is visible
// to `habana_frameworks.torch.internal.bridge_config`.
//
// DEFAULTS TO strict. An unset or empty PT_HPU_EAGER_GUARD means strict, and so
// does anything unrecognised: the rule is "eager launches should be crashes, not
// compiles", and a guard that fails open on a typo is a guard that lets 139 k
// eager compiles through because someone mistyped a variable name. `off` and
// `census` are still here, but they have to be asked for by name.
inline Mode parse_mode() {
  const char* v = GET_ENV_FLAG_NEW(PT_HPU_EAGER_GUARD);
  if (v == nullptr || *v == '\0' || std::strcmp(v, "strict") == 0) {
    return Mode::strict;
  }
  if (std::strcmp(v, "off") == 0) {
    return Mode::off;
  }
  if (std::strcmp(v, "census") == 0) {
    return Mode::census;
  }
  // The enum constraint in env_flags.yaml rejects anything else, so reaching
  // here means the constraint was bypassed (SET_ENV_FLAG_NEW). Fail closed:
  // an unrecognised value must not be a way to turn the guard off.
  return Mode::strict;
}

// How many torch-internal frames to walk past when reporting a site.
//
// DEFAULT 0 = today's behaviour exactly: report the frame that made the C++
// call, even when that frame is torch's own dispatch wrapper.
//
// Read with plain getenv on purpose: this variable is a diagnostic knob for the
// guard alone, and adding it to scripts/env_flags.yaml would regenerate the
// whole env-flag machinery (a cmake re-run plus a rebuilt Bindings.cpp) for a
// number nothing else reads. It never changes WHETHER a launch is refused --
// only which frame is named in the report.
inline int parse_depth() {
  const char* v = std::getenv("PT_HPU_EAGER_GUARD_DEPTH");
  if (v == nullptr || *v == '\0') {
    return 0;
  }
  const long n = std::strtol(v, nullptr, 10);
  if (n <= 0) {
    return 0;
  }
  // A walk this deep is a diagnostic, not a hot path; cap it so a typo cannot
  // turn one eager launch into a long stack walk.
  constexpr long kMaxDepth = 64;
  return static_cast<int>(n > kMaxDepth ? kMaxDepth : n);
}

// True when the frame's file lives under a torch/ directory -- either the venv's
// site-packages/torch/ or a source checkout's torch/. A wrapper in torch/ is
// the frame that hid the caller, so the walk steps past it.
inline bool is_torch_frame(const std::string& file) {
  if (file.empty()) {
    return false;
  }
  // Match "torch/" as a whole path component, so a file named "mytorch/x.py"
  // does not count.
  const std::string needle = "torch/";
  for (size_t pos = file.find(needle); pos != std::string::npos;
       pos = file.find(needle, pos + 1)) {
    if (pos == 0 || file[pos - 1] == '/') {
      return true;
    }
  }
  return false;
}

inline std::string unicode_utf8(PyObject* o) {
  if (o == nullptr) {
    return std::string();
  }
  const char* s = PyUnicode_AsUTF8(o);
  return s != nullptr ? std::string(s) : std::string();
}

} // namespace detail

inline Mode mode() {
  static const Mode m = detail::parse_mode();
  return m;
}

inline int depth() {
  static const int d = detail::parse_depth();
  return d;
}

// The hot-path predicate. When off: a load of a function-local static and a
// not-taken branch.
inline bool active() {
  static const bool a = mode() != Mode::off;
  return a;
}

struct Site {
  bool ok = false;
  std::string file;
  std::string func;
  int line = 0;
  std::string why; // set when !ok

  // Set when the depth walk moved past one or more torch frames. The reported
  // site above is then the first non-torch frame; these keep the torch frame
  // that was actually reported before the walk, so nothing is lost.
  bool walked = false;
  std::string torch_file;
  std::string torch_func;
  int torch_line = 0;
  int depth_used = 0;
};

namespace detail {

inline bool read_frame(PyFrameObject* frame, Site& site) {
  PyCodeObject* code = PyFrame_GetCode(frame); // new ref
  site.line = PyFrame_GetLineNumber(frame);
  if (code != nullptr) {
    site.file = unicode_utf8(code->co_filename);
    site.func = unicode_utf8(code->co_name);
#if PY_VERSION_HEX >= 0x030b0000
    // co_qualname distinguishes Class.method from a bare method name; it has
    // existed since 3.11 and is what makes two sites in one file tell apart.
    const std::string qual = unicode_utf8(code->co_qualname);
    if (!qual.empty()) {
      site.func = qual;
    }
#endif
    Py_DECREF(code);
  }
  return true;
}

} // namespace detail

// The Python frame of the caller. `skip` frames are walked back from the
// current one; skip = 0 is the Python function that made the C++ call.
//
// With depth > 0 and the reported frame inside torch, the walk continues past
// torch frames up to `depth` frames and reports the first frame that is not
// torch's. The torch frame is preserved in the Site so the message can name it.
inline Site python_site(int skip = 0, int depth = 0) {
  Site site;
  if (!Py_IsInitialized()) {
    site.why = "the Python interpreter is not initialised";
    return site;
  }
  // Do not let PyGILState_Ensure invent a thread state for a thread that has
  // none: an eager op issued from a non-Python thread has no Python caller to
  // name, and a dummy thread state would report "no frame" as if it were the
  // op's fault.
  if (PyGILState_GetThisThreadState() == nullptr) {
    site.why = "this thread has no Python thread state (not a Python thread)";
    return site;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyFrameObject* frame = PyThreadState_GetFrame(PyThreadState_Get()); // new ref
  for (int i = 0; i < skip && frame != nullptr; ++i) {
    PyFrameObject* back = PyFrame_GetBack(frame); // new ref
    Py_DECREF(frame);
    frame = back;
  }
  if (frame == nullptr) {
    site.why = "no Python frame is executing on this thread";
    PyGILState_Release(gil);
    return site;
  }
  detail::read_frame(frame, site);

  // The depth walk. Only moves the report up the stack; never changes whether a
  // launch is refused. A walk that runs out of frames leaves the torch frame as
  // the reported site and records how far it got, so the caller can see that our
  // code was not found rather than being told nothing.
  if (depth > 0 && detail::is_torch_frame(site.file)) {
    const std::string first_file = site.file;
    const std::string first_func = site.func;
    const int first_line = site.line;
    PyFrameObject* walk = PyFrame_GetBack(frame); // new ref
    int steps = 0;
    while (walk != nullptr && steps < depth) {
      Site candidate;
      detail::read_frame(walk, candidate);
      ++steps;
      if (!detail::is_torch_frame(candidate.file)) {
        Py_DECREF(walk);
        Py_DECREF(frame);
        site = candidate;
        site.ok = true;
        site.walked = true;
        site.torch_file = first_file;
        site.torch_func = first_func;
        site.torch_line = first_line;
        site.depth_used = steps;
        PyGILState_Release(gil);
        return site;
      }
      PyFrameObject* back = PyFrame_GetBack(walk); // new ref
      Py_DECREF(walk);
      walk = back;
    }
    if (walk != nullptr) {
      Py_DECREF(walk);
    }
    // Ran out of frames (or depth) while still inside torch: keep the torch
    // frame as the site, and say how far the walk went. If the walk never moved
    // (no back frame at all) there is nothing to report beyond the torch frame.
    if (steps > 0) {
      site.walked = true;
      site.torch_file = first_file;
      site.torch_func = first_func;
      site.torch_line = first_line;
      site.depth_used = steps;
      site.file = first_file;
      site.func = first_func;
      site.line = first_line;
    }
  }

  Py_DECREF(frame);
  PyGILState_Release(gil);
  site.ok = true;
  return site;
}

// Where the guard's output goes: PT_HPU_EAGER_GUARD_LOG if set (keeps the lines
// out of the boot log being measured), otherwise stderr.
inline void emit(const std::string& text) {
  static FILE* out = [] {
    const char* p = GET_ENV_FLAG_NEW(PT_HPU_EAGER_GUARD_LOG);
    if (p != nullptr && *p != '\0') {
      FILE* f = std::fopen(p, "a");
      if (f != nullptr) {
        return f;
      }
    }
    return stderr;
  }();
  std::fwrite(text.data(), 1, text.size(), out);
  std::fflush(out);
}

// Shapes, for the diagnostic. Capped so that one line stays one line.
inline std::string format_shapes(const std::vector<at::IValue>& inputs) {
  constexpr size_t kMax = 6;
  std::string s;
  s += '[';
  size_t shown = 0;
  for (const auto& v : inputs) {
    if (shown >= kMax) {
      s += ", ...";
      break;
    }
    if (shown > 0) {
      s += ", ";
    }
    ++shown;
    if (v.isTensor()) {
      const auto& t = v.toTensor();
      if (!t.defined()) {
        s += "<undefined>";
        continue;
      }
      s += '[';
      const auto sizes = t.sizes();
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) {
          s += ',';
        }
        s += std::to_string(sizes[i]);
      }
      s += ']';
      s += at::toString(t.scalar_type());
    } else if (v.isScalar()) {
      s += "scalar";
    } else if (v.isIntList()) {
      s += "int[]";
    } else if (v.isDoubleList()) {
      s += "float[]";
    } else if (v.isNone()) {
      s += "None";
    } else if (v.isBool()) {
      s += v.toBool() ? "True" : "False";
    } else {
      s += v.tagKind();
    }
  }
  s += ']';
  return s;
}

namespace detail {

// The tag is the line's CLASS, not the configured mode. They coincide for a
// reported violation, but an exempted launch is not a violation and must not be
// labelled STRICT: the first strict boot printed
//     [eager-guard] EXEMPT-WARMUP (policy A allows eager during compile) [eager-guard] STRICT op=...
// because the mode word was baked in before the exemption was decided. A line
// that says STRICT on a launch that was allowed through is exactly the kind of
// mislabel that hides a defect, so the class is passed in.
inline std::string prefix(const char* tag = nullptr) {
  if (tag != nullptr && *tag != '\0') {
    return std::string("[eager-guard] ") + tag + " ";
  }
  return mode() == Mode::strict ? "[eager-guard] STRICT " : "[eager-guard] CENSUS ";
}

// op=... site=... shapes=... -- one line, greppable, aggregable by sort|uniq -c.
// When the depth walk moved the report, the real caller is the headline and the
// torch frame it was reported at is a parenthetical, so both survive.
inline std::string site_line(
    const char* op,
    const char* op_name,
    const Site& site,
    const std::string& shapes,
    const char* tag = nullptr) {
  std::string s;
  s.reserve(512);
  s += prefix(tag);
  s += "op=";
  s += (op != nullptr && *op != '\0') ? op : "<unnamed>";
  if (site.ok) {
    s += " site=";
    s += site.file;
    s += ':';
    s += std::to_string(site.line);
    s += " in ";
    s += site.func;
    if (site.walked) {
      s += " (torch frame: ";
      s += site.torch_file;
      s += ':';
      s += std::to_string(site.torch_line);
      s += " in ";
      s += site.torch_func;
      s += "; walked ";
      s += std::to_string(site.depth_used);
      s += " frame";
      s += (site.depth_used == 1 ? "" : "s");
      s += ")";
    }
  } else {
    s += " site=<unavailable: ";
    s += site.why;
    s += ">";
  }
  s += " shapes=";
  s += shapes;
  if (op_name != nullptr && *op_name != '\0' &&
      (op == nullptr || std::strcmp(op, op_name) != 0)) {
    s += " op_name=";
    s += op_name;
  }
  s += '\n';
  return s;
}

[[noreturn]] inline void strict_exit(
    const std::string& line,
    const char* op,
    const Site& site) {
  emit(line);
  std::string d;
  d += "[eager-guard] STRICT: refusing to continue past an eager launch.\n";
  d += "[eager-guard]   op     : ";
  d += (op != nullptr && *op != '\0') ? op : "<unnamed>";
  d += "\n[eager-guard]   source : ";
  if (site.ok) {
    d += site.file;
    d += ':';
    d += std::to_string(site.line);
    d += " in ";
    d += site.func;
    if (site.walked) {
      d += "\n[eager-guard]   torch  : ";
      d += site.torch_file;
      d += ':';
      d += std::to_string(site.torch_line);
      d += " in ";
      d += site.torch_func;
      d += " (walked ";
      d += std::to_string(site.depth_used);
      d += " frame";
      d += (site.depth_used == 1 ? "" : "s");
      d += " past torch)";
    }
  } else {
    d += "<no Python frame: ";
    d += site.why;
    d += ">";
  }
  d += "\n[eager-guard]   why    : this launch is outside a compiled region, so it "
       "mints and\n"
       "[eager-guard]            re-mints its own recipe instead of reusing one. "
       "The eager\n"
       "[eager-guard]            path is not allowed to launch quietly while "
       "PT_HPU_EAGER_GUARD=strict.\n"
       "[eager-guard]   this is a controlled exit (status 70), not a crash or a "
       "segfault.\n"
       "[eager-guard]   to collect the whole list instead of stopping at the "
       "first, re-run with:\n"
       "[eager-guard]     PT_HPU_EAGER_GUARD=census\n"
       "[eager-guard]   and, to keep those lines out of the boot log:\n"
       "[eager-guard]     PT_HPU_EAGER_GUARD_LOG=/path/to/eager-guard.log\n"
       "[eager-guard]   to report the real caller past torch's own frames:\n"
       "[eager-guard]     PT_HPU_EAGER_GUARD_DEPTH=8\n";
  emit(d);
  std::fflush(nullptr);
  std::_Exit(kStrictExitCode);
}

} // namespace detail

// Called at the eager dispatch point, on the Python thread, once per eager op.
// Returns true when the guard was active, so the caller can record that a
// dispatch site was reported.

// ---------------------------------------------------------------------------
// WARMUP / COMPILE PHASE.
//
// Two separate policies, both default-on:
//   A. no eager user code      -- PT_HPU_EAGER_GUARD (strict by default)
//   B. no post-warmup compile  -- PT_HPU_POSTWARMUP_COMPILE (strict by default)
// Both can be downgraded to a warning that names file:line.
//
// Policy A exempts eager launches issued *while compiled regions are being
// built*, because that is when a region legitimately traces torch-internal code
// and an eager op there is construction, not a serving-path defect. Measured:
// 39 such launches in a full boot, 0 growth across 5 served requests.
//
// WHY FRAME-BASED AND NOT A FLAG. A thread-local "we are in warmup" flag would
// need a C++<->Python binding (no PYBIND11_MODULE TU includes this header), and
// worse, a flag that got stuck open would silently turn strict into off for the
// rest of the process -- the fail-open shape that has already cost this campaign
// real defects. Reading the actual stack cannot get stuck, and if the marker
// function is ever renamed the exemption stops applying: strict then aborts
// during warmup, which is loud and immediately obvious. Fail closed.
//
// The marker is our own code, not torch's. `torch/_ops.py OpOverload.__call__`
// was rejected as a discriminator: it fires identically during tracing and at
// runtime, so it cannot tell the two phases apart.
// ---------------------------------------------------------------------------

// How far up the stack to look for a warmup frame. Generous: the frame can be
// deep (region -> contextmanager -> dummy scenario -> warmup_model -> worker).
inline int warmup_walk() {
  return 128;
}

inline bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// A frame that means "compiled regions are being built right now".
// NOTE ON THE MARKER SET, and why it is the init chain rather than just warmup.
// The first version listed only the warmup functions, and strict then aborted on
// a launch issued from `Region.__call__` during MODEL CONSTRUCTION -- correct
// about the launch, wrong about the phase. A region legitimately first-compiles
// during construction (site #0's `_zeros`), during weight loading, and during
// warmup, and all of that is "regions are being built".
//
// So the markers are the functions that only ever run while the engine is
// initialising. Deliberately NOT included: `execute_model`, `sample_tokens`,
// `_execute_model_generic` and the rest of the serving path -- those are exactly
// what policy A is for.
// `Site::func` carries `co_qualname`, not `co_name` (see read_frame: the
// qualname is what tells two sites in one file apart). For a method that is
// "Class.method", so a bare marker must be matched against the LAST component.
//
// THIS WAS A REAL, SILENT BUG, and it is the reason the first strict boot
// aborted at a warmup launch with ZERO "EXEMPT-WARMUP" lines: every marker
// compared against a qualname that could never equal it, so the exemption never
// applied at all. It failed closed (loud abort), which is the design, but it
// cost a boot and an earlier wrong diagnosis -- I blamed the marker SET being
// too narrow and widened it, when the comparison was wrong for every name in it.
inline std::string bare_name(const std::string& qual) {
  const size_t dot = qual.rfind('.');
  return dot == std::string::npos ? qual : qual.substr(dot + 1);
}

inline bool is_warmup_frame(const Site& s) {
  if (s.file.empty() || s.func.empty()) {
    return false;
  }
  const std::string fn = bare_name(s.func);
  if (ends_with(s.file, "hpu_worker.py")) {
    return fn == "__init__" || fn == "load_model" ||
           fn == "initialize_from_config" ||
           fn == "compile_or_warm_up_model";
  }
  if (ends_with(s.file, "hpu_model_runner.py")) {
    return fn == "__init__" || fn == "load_model" || fn == "warmup_model" ||
           fn == "warmup_graphs" || fn == "warmup_sampler" ||
           fn == "_prepare_dummy_scenario" || fn == "_execute_dummy_scenario";
  }
  return false;
}

// True when a warmup/compile marker frame is on this thread's Python stack.
inline bool in_warmup_phase() {
  if (!Py_IsInitialized() || PyGILState_GetThisThreadState() == nullptr) {
    return false;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  bool found = false;
  PyFrameObject* walk = PyThreadState_GetFrame(PyThreadState_Get()); // new ref
  int steps = 0;
  const int limit = warmup_walk();
  while (walk != nullptr && steps < limit) {
    Site s;
    detail::read_frame(walk, s);
    if (is_warmup_frame(s)) {
      found = true;
      break;
    }
    PyFrameObject* back = PyFrame_GetBack(walk); // new ref
    Py_DECREF(walk);
    walk = back;
    ++steps;
  }
  if (walk != nullptr) {
    Py_DECREF(walk);
  }
  PyGILState_Release(gil);
  return found;
}

template <typename ShapesFn>
inline bool on_eager_dispatch(const char* op, const char* op_name, ShapesFn&& shapes) {
  if (!active()) {
    return false;
  }
  const Site site = python_site(0, depth());
  const std::string shape_text = shapes();

  // Policy A: no eager user code. Eager launches issued while compiled regions
  // are being built are exempt -- but they are still RECORDED and LABELLED, so
  // an exempted launch can never be mistaken for a clean one. Silence is what
  // let a 2047-of-2048-element corruption through earlier in this campaign.
  if (in_warmup_phase()) {
    emit(detail::site_line(op, op_name, site, shape_text, "EXEMPT-WARMUP"));
    return true;
  }

  const std::string line = detail::site_line(op, op_name, site, shape_text);
  if (mode() == Mode::strict) {
    detail::strict_exit(line, op, site);
  }
  emit(line);
  return true;
}

// Called from EagerExec::launch when an eager launch arrives without a dispatch
// site. That is the one thing this instrument must not swallow: it means the
// launch took a path that bypasses EagerOpBase::run, and the census has a hole
// exactly there. It is reported, not hidden.
template <typename ShapesFn>
inline void on_launch_without_site(const char* op, ShapesFn&& shapes) {
  if (!active()) {
    return;
  }
  Site site;
  site.why =
      "this eager launch reached EagerExec::launch without passing through "
      "EagerOpBase::run, and EagerExec::launch runs on the lowering thread "
      "(or with the GIL released), where no Python frame exists";
  const std::string shape_text = shapes();
  if (in_warmup_phase()) {
    emit(detail::site_line(op, nullptr, site, shape_text, "EXEMPT-WARMUP"));
    return;
  }
  const std::string line = detail::site_line(op, nullptr, site, shape_text);
  if (mode() == Mode::strict) {
    detail::strict_exit(line, op, site);
  }
  emit(line);
}

} // namespace habana::eager_guard
