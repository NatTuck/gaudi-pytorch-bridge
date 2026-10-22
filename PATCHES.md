# This branch: profiling identity for Gaudi 2

A small, deliberately narrow patch set against upstream `v1.24.2`. It exists because the
profiling trace on this stack **cannot be joined to the code that produced it**, which is the
difference between a timing instrument and a diagnostic one.

Everything here is written up in a Gaudi 2 optimization manual chapter on the identity chain
(`g2-opt-manual/05-profiling.md` in the project this came out of): the five joins from a device
event to a source line, which of them are keys and which are inferences, and what breaks.

## 1. `export_chrome_trace` rejects torch 2.13's call signature — APPLIED, VERIFIED

`python_packages/habana_frameworks/torch/activity_profiler/activity_profiler.py`, both
occurrences (`habana_profile_wrapper` and `habana_autograd_profile_wrapper`):

```python
def export_chrome_trace(self, path, *args, **kwargs):
    super().export_chrome_trace(path, *args, **kwargs)
```

torch 2.13 calls `export_chrome_trace(path, metadata)`. The unpatched shim takes `(self, path)`,
so **any** use of `torch.profiler.tensorboard_trace_handler` raises at profiler exit:

```
TypeError: ...habana_autograd_profile_wrapper.export_chrome_trace() takes 2 positional arguments but 3 were given
```

It fires *after* the window has completed, so the run is lost. Filed upstream as
[HabanaAI/gaudi-pytorch-bridge#23](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/23).

**Verified** by monkeypatching the method to accept the extra argument and calling the torch-level
export: 790 events, 5 `Torch-Compiled Region` spans, 494 device events. Previously: `TypeError`.

## 2. The recipe identity the profiler needs is emitted but filtered out — NOT APPLIED

This is the interesting one, and it is **not** in this branch yet, because the correct string is
not yet identified. Recording the state so nobody repeats the search.

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
Torch's `External id` correlation does not reach HPU device events (0 of 3,880 carry it).

### What the source says

The bridge **already** emits the annotation we need, on the Python thread, inside the region span:

- `habana_eager/graph_execs_group.cpp:123` — `PT_EAGER_TRACE_WITH_NAME(m_graphs_group_name)`
- `habana_eager/graph_exec.cpp:208` and `:794` — `PT_EAGER_TRACE_WITH_NAME(m_graph_name)`

They never reach the trace because `HpuActivityProfilerSession::start()`
(`backend/profiling/kineto_activity_profiler.cpp:517`) passes a `mandatory_events` list, and
`BridgeLogsSourceImpl` drops any event whose name does not match one of those patterns. The list
contains `launch_recipe`, `LaunchRecipeTask`, `add_new_recipe`, `launch`, `run`, `Launch` — none of
which match a graph name.

The filter is `std::regex_search` (`backend/profiling/trace_sources/bridge_logs_source.cpp:110-113`),
so **a longer name still passes**: appending a pattern to that list is enough to let the annotation
through. No new plumbing is needed, and `addActivity` already supports an args map if we later want
the id as a field rather than in the name.

### Why the patch is not here yet

Two candidate strings, and **both are wrong**:

- `m_graphs_group_name` is `"graphs_group_" + std::to_string(recipe_id)` (`graph_execs_group.cpp:112`)
  — a GraphStorage index, not the dump's `graph_name`.
- `m_graph_name` is the *FX* name (`graph_execs_group.h:64` → `graph_exec.cpp:367`, with an
  `_fx`→`_jit` rewrite at `:379-380`), not the Synapse recipe name.

The name that appears in the dump is generated at compile time:
`get_unique_recipe_name(name_, eager_mode)` (`backend/synapse_helpers/graph.cpp:643`) =
`graph_dump_dir + name_ + "_" + suffix`, and that string is what is handed to `synGraphCompile`.

**So the next step is an instrumented build**: print `m_graphs_group_name`, `m_graph_name`,
`recipe_id`, and the compiled recipe name at launch, run a small program, and diff against
`<dump>/.graph_dumps/*.json`. Whichever string reproduces the dump's `graph_name` is the one to
annotate — and if it is only a prefix, the suffix must come from the compile site, which moves the
patch to `GraphExec`/`GraphExecsGroup` where the compiled recipe lives.

Note for whoever does it: `PTFuncLog` stores `std::string_view` (`pytorch_helpers/habana_helpers/logging.h:519-523`),
so a dynamic name must be a named `std::string` that outlives the logger — a temporary passed
straight into the macro dangles for the whole function scope.

## Building

The patch set is intended to be built the same way the stock bridge is; nothing here changes the
build. `cmake -G Ninja` against the same torch the runtime uses (here `2.13.0a0+git2469d18`).

## Status

| item | state |
|---|---|
| 1. `export_chrome_trace` signature | applied, verified, safe to carry |
| 2. recipe identity in the trace | **not applied** — correct string under investigation |
