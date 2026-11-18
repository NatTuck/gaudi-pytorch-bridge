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

// Clause 3: the region identity travels with the recipe into the cache.
//
// The problem this replaces: the identity used to live in a *side table* -- the
// Python-side `recipe_id -> (region, index)` map written by
// `compile_backend/launch_ledger.py`. That map is per-process, so on a warm boot
// it is empty while the recipes come from disk. `DiskCache` serialises the
// recipe *name*, and that name embeds the frame counter and recipe id of the
// boot that wrote the cache, so a warm boot's names are stale and the join
// through the side table is unsound. (That is why the ledger had to mint a key
// in `RecipeLauncher(std::istream&, ...)` at all: without it a warm boot
// reported every launch as unattributed.)
//
// What replaces it: three fields written into the recipe's `.metadata` stream,
// which is the same stream that already carries the serialised
// `RecipeValueSpec`. A warm boot reads them back off disk, so the identity is a
// field of the artifact rather than a counter in the process that happens to be
// running.
//
// This registry is the *compile-time transport* only: it carries the identity
// from the Python frame (which is the only place that has the FX `stack_trace`)
// to `RecipeLauncher::Serialize` (which is the only place that writes the cache
// entry). It is keyed by the graph name the Python side already passes as
// `parent_graph_name` and which `RecipeLauncher::graph_name_` already holds, so
// it needs no new plumbing through `add_new_recipe` and no change to `graph.h`.
// Once the entry is written, the registry is irrelevant to it: the identity is
// in the file.
//
// Header-only on purpose: a new translation unit needs a cmake re-run and this
// build tree's configure-time variables are in neither the environment nor the
// cache (analysis/precompiled-region/REPORT.md section 8).

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace habana::region_identity {

struct Entry {
  std::string name;   // the region's name, e.g. "hpu_gdn_pytorch.py:_l2norm_last_dim#a1b2c3d4"
  std::string source; // the source annotation, e.g. "/path/hpu_gdn_pytorch.py:1642"
  uint32_t index{0};  // the `foo.N` ordinal of this recipe within the region
  // The CompileInputs record key for this graph. Empty when nothing registered
  // one, which is the honest answer for an eager recipe. When set, THIS is the
  // recipe cache filename key -- the record is a pure function of the inputs, so
  // the toolchain does not have to be part of the key.
  std::string compile_key;
};

namespace detail {

inline std::unordered_map<std::string, Entry>& map() {
  static std::unordered_map<std::string, Entry> m;
  return m;
}

inline std::mutex& mutex() {
  static std::mutex m;
  return m;
}

// The two sides name the same frame differently:
//
//   Python side  `parent_graph_name`          graph_0004_base_fx
//   C++ side     RecipeValueSpec::graph_name  graph_0004_fused_0
//                (set from GetSynapseGraphName() = name_ + "_" + graph_index_,
//                 hpu_habana_launch_op_pt.cpp:261; name_ is the fused partition
//                 name, not the frame name)
//
// They share the `graph_<frame id>` prefix, which is what identifies the frame
// both sides are talking about. Keying on that is deliberate: it is the one
// token the FX partitioner and the synapse lowering both derive from the same
// dynamo frame, and it is stable within a boot.
inline std::string frame_key(const std::string& s) {
  constexpr const char* kPrefix = "graph_";
  constexpr size_t kPrefixLen = 6;
  if (s.size() <= kPrefixLen || s.compare(0, kPrefixLen, kPrefix) != 0) {
    return {};
  }
  size_t i = kPrefixLen;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
    ++i;
  }
  if (i == kPrefixLen) {
    return {};
  }
  return s.substr(0, i);
}

} // namespace detail

// Called from Python (compile_backend/launch_ledger.py) once per partition,
// before `graph_compile`. Cheap: a map insert, compile-time only.
inline void note(
    const std::string& graph_name,
    const std::string& name,
    const std::string& source,
    uint32_t index,
    const std::string& compile_key = {}) {
  if (graph_name.empty()) {
    return;
  }
  const Entry e{name, source, index, compile_key};
  std::lock_guard<std::mutex> lock(detail::mutex());
  detail::map()[graph_name] = e;
  // Also key it by the frame, because the C++ side that writes the cache entry
  // holds the *synapse* name rather than this one (see detail::frame_key).
  const std::string fk = detail::frame_key(graph_name);
  if (!fk.empty()) {
    detail::map()[fk] = e;
  }
}

// Called from `RecipeLauncher::Serialize` at the moment the cache entry is
// written. Returns false when nothing was registered for this graph, which is
// the honest answer for an eager recipe (no region) and for a precompiled
// artifact loaded from disk (its identity came from the file, not from here).
inline bool lookup(const std::string& graph_name, Entry& out) {
  if (graph_name.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(detail::mutex());
  auto& m = detail::map();
  auto it = m.find(graph_name);
  if (it != m.end()) {
    out = it->second;
    return true;
  }
  const std::string fk = detail::frame_key(graph_name);
  if (!fk.empty()) {
    it = m.find(fk);
    if (it != m.end()) {
      out = it->second;
      return true;
    }
  }
  return false;
}

// For the ledger/report tooling: how many regions were registered in this
// process. Zero on a boot that compiled nothing.
inline size_t size() {
  std::lock_guard<std::mutex> lock(detail::mutex());
  return detail::map().size();
}

} // namespace habana::region_identity
