# This branch: profiling identity for Gaudi 2

A small, deliberately narrow patch set against upstream `v1.24.2`. It exists because the profiling
trace on this stack **cannot be joined to the code that produced it**, which is the difference
between a timing instrument and a diagnostic one.

The reasoning is written up in a Gaudi 2 optimization manual chapter on the identity chain
(`g2-opt-manual/05-profiling.md` in the project this came out of): the five joins from a device
event to a source line, which of them are keys and which are inferences, and what breaks.

| item | state |
|---|---|
| 0. `-pie` on shared libraries (build fix) | **applied, verified** — without it the backend does not link |
| 1. `export_chrome_trace` signature | **applied, verified** |
| 2. recipe identity in the trace | **applied, verified end to end** |

---

## 1. `export_chrome_trace` rejects torch 2.13's call signature — APPLIED, VERIFIED

`python_packages/habana_frameworks/torch/activity_profiler/activity_profiler.py`, both occurrences
(`habana_profile_wrapper` and `habana_autograd_profile_wrapper`):

```python
def export_chrome_trace(self, path, *args, **kwargs):
    super().export_chrome_trace(path, *args, **kwargs)
```

torch 2.13 calls `export_chrome_trace(path, metadata)`. The unpatched shim takes `(self, path)`, so
**any** use of `torch.profiler.tensorboard_trace_handler` raises at profiler exit:

```
TypeError: ...habana_autograd_profile_wrapper.export_chrome_trace() takes 2 positional arguments but 3 were given
```

It fires *after* the window has completed, so the run is lost. Filed upstream as
[HabanaAI/gaudi-pytorch-bridge#23](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/23).

**Verified** by monkeypatching the method to accept the extra argument and calling the torch-level
export: 790 events, 5 `Torch-Compiled Region` spans, 494 device events. Previously: `TypeError`.

---

## 2. The recipe identity is emitted but filtered out — IDENTIFIED, NOT APPLIED

### The problem

To attribute device time to a source line you need `compile_id -> recipe`. The trace has both
halves, on different events, with no shared field:

| event | thread | carries | inside the `Torch-Compiled Region` span? |
|---|---|---|---|
| `launch_recipe` | Python/main | **no args** | **yes** (1576/1576) |
| `LaunchRecipe` | runtime | **no args** | no |
| `enqueueWithExternalEventsExt` | runtime | `recipeHandle`, `recipeId`, `recipeName` | mostly no |

Pairing the two host streams by position works only when they are 1:1 — exact on a small program
(20/20/20), unusable on a real model (1576 launches vs 1604 enqueues, 24.6 % modal share, 0 of 118
compile ids consistent). Constant-lag alignment fails too (25 % at every offset from −40 to +40).
Torch's `External id` correlation never reaches HPU device events (0 of 3,880 carry it).

### What the source says — the missing link is one string substitution

The bridge **already** emits an annotation carrying the FX graph name, on the Python thread, inside
the region span:

- `habana_eager/graph_exec.cpp:794` — `PT_EAGER_TRACE_WITH_NAME(m_graph_name)` in `GraphExec::launch`

It never reaches the trace because `HpuActivityProfilerSession::start()`
(`backend/profiling/kineto_activity_profiler.cpp:517`) passes a `mandatory_events` list, and
`BridgeLogsSourceImpl` drops every event whose name matches none of those patterns. The list holds
`launch_recipe`, `LaunchRecipeTask`, `add_new_recipe`, `launch`, `run`, `Launch` — a graph name
matches none of them. The filter is `std::regex_search`
(`backend/profiling/trace_sources/bridge_logs_source.cpp:110-113`), so **a longer name still
passes**: appending one pattern is enough to let the annotation through.

And the name in the dump is derived from that same FX name by one substitution. In
`backend/kernel/hpu_habana_launch_op_pt.cpp:247-248`:

```cpp
if (syn_graph_name_.find("_jit") != std::string::npos) {
  syn_graph_name_.replace(syn_graph_name_.find("jit"), 3, "syn");
}
```

`GetSynapseGraphName()` is that result, and the recipe name handed to `synGraphCompile` is
`get_unique_recipe_name(name_, eager_mode)` = `graph_dump_dir + name_ + "_" + suffix`
(`backend/synapse_helpers/graph.cpp:643`). That is exactly the `graph_name` in the post-graph dump.

**So: `dump.graph_name` == `<dump_dir>/` + `m_graph_name` with `jit`→`syn` + `_<suffix>`.**

Two earlier candidates were checked and are **wrong** — recording them so the search is not
repeated: `m_graphs_group_name` is `"graphs_group_" + recipe_id` (`graph_execs_group.cpp:112`), a
GraphStorage index; `m_parent_graph_name` is the pre-substitution FX name.

### The patch that follows

1. `backend/profiling/kineto_activity_profiler.cpp` — add a pattern for the graph-name family to the
   eager `mandatory_events` list (e.g. `^graph_`; it is a regex, and the names are
   `graph_<NNNN>_fused_…`).
2. `habana_eager/graph_exec.cpp:794` — emit the *synapse-form* name, so the event name is directly
   the dump prefix:

```cpp
// PTFuncLog stores std::string_view (pytorch_helpers/habana_helpers/logging.h:519-523),
// so this must outlive the logger -- a temporary passed into the macro would dangle.
const std::string syn_graph_name = [] (std::string n) {
  const auto pos = n.find("_jit");
  if (pos != std::string::npos) n.replace(n.find("jit"), 3, "syn");
  return n;
}(m_graph_name);
PT_EAGER_TRACE_WITH_NAME(syn_graph_name);
```

### Verified end to end

Applied and measured on a Gaudi 2 (torch 2.13.0a0+git2469d18, SynapseAI 1.24.2.481). A three-block
program — one block per engine, each its own `torch.compile` region — now attributes as a key join:

```
=== region -> recipe: KEY JOIN via graph name (patched bridge)
recipe_chain_demo.py:63:mme_block  MME 0.038 ms   recipe 57369  dump graph_0000_fused_0_syn_1000000_1000000_4.5.57369
recipe_chain_demo.py:76:dma_block  DMA 0.016 ms   recipe 61189  dump graph_0002_fused_0_syn_1002000_1002000_6.5.61189
recipe_chain_demo.py:68:tpc_block  TPC 0.014 ms   recipe 62418  dump graph_0001_fused_0_syn_1001000_1001000_5.5.62418
verification: PASS -- engine attribution matches the code's intent
```

Three compile ids resolve to three source lines, three recipes, three dumps, and the node types
confirm the intent (`GEMM x2` for the matmul block, one fused TPC kernel for the elementwise block,
`DmaMemcpy + add_fwd_bf16` for the copy block). Every `_syn_` event name mapped to **exactly one**
dump.

Note the join is a **prefix** match, not equality: the event carries
`graph_0000_fused_0_syn_1000000` while the dump is `…_syn_1000000_1000000_4`, because
`get_unique_recipe_name` appends a compile-time suffix. The dumper names each file after the graph,
so matching the event name against the *filename* is unambiguous in practice — but a consumer should
treat it as a prefix and assert uniqueness, as the reference implementation does.

---

## Building

Unchanged from stock, except that cmake 3.31.6 is currently required (see above). The build must use
the same torch as the runtime — here `2.13.0a0+git2469d18`.

---

## 0. `-pie` is applied to shared libraries, so the backend cannot link — APPLIED, VERIFIED

`cmake/Targets.cmake`, in `set_up_link_options()`:

```cmake
target_link_options(${target_name} PRIVATE -pie)   # applied to every target, SHARED included
```

`-pie` is an **executable** link option. On a `-shared` link it makes `ld` produce a PIE executable:
it pulls in `Scrt1.o`, demands `main`, and forces full symbol resolution. Every symbol the backend
legitimately leaves for the plugin to resolve at load time then becomes a hard error:

```
/usr/bin/ld: Scrt1.o: in function `_start': undefined reference to `main'
/usr/bin/ld: … undefined reference to `Logger::habana_assert(…)'
/usr/bin/ld: … undefined reference to `lazy_to_backend::is_lazy_inference_call_context()'
```

Those references are **correct**: the shipped working `libhabana_pytorch_backend.so` leaves both
undefined (`nm -D --undefined-only` → 1 each), because `habana_pytorch_plugin_common` is an OBJECT
library the *plugin* links, not the backend.

The guard below fixes it; the flag ordering makes an override ineffective (`-no-pie` placed before
the target's `-pie` loses — verified, the link still failed).

```cmake
get_target_property(_hpu_target_type ${target_name} TYPE)
if(NOT _hpu_target_type MATCHES "LIBRARY")
  target_link_options(${target_name} PRIVATE -pie)
endif()
```

This is a genuine upstream defect rather than a workaround: the build currently depends on flag
ordering to avoid linking a shared library as an executable. Worth reporting separately from #23,
which is about the profiler.

## Build environment

Nine environment variables are required (`HABANA_SOFTWARE_STACK`, `HCL_INCLUDE_DIR`,
`SPECS_EXT_ROOT`, **`SPECS_EMBEDDED_ROOT`**, `SYNAPSE_INCLUDE_DIR`, `SYNAPSE_UTILS_INCLUDE_DIR`,
`SWTOOLS_SDK_ROOT`, `BUILD_ROOT_LATEST`, `PYTORCH_MODULES_ROOT_PATH`), plus
`latest/libhl_logger.so` symlinked from `/usr/lib/habanalabs/`.

`SPECS_EMBEDDED_ROOT` is **not** in the project's `REQUIRED_ENV_VARIABLES` list, so omitting it is
silent — and without it `cmake/PrepareDependencies.cmake` never creates `npu::specs_embedded`,
`mem_hlml.cpp` is dropped, and the resulting library is not a drop-in (it lacks the `HlMlMemory*`
symbols the installed `_hpu_C.so` needs, so `import torch` fails).

Built and verified with **cmake 4.4.x** — the earlier belief that an older cmake was required was
wrong; the `-pie` flag was the cause.
