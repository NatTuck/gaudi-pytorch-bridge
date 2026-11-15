/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <variant>
#include <vector>

#include "habana_eager/graph_execs_group.h"
#include "jit_fork/ir/ir.h"

namespace habana::graph {

class GraphStorage {
 public:
  static GraphStorage& get();

  // One storage entry is either a compiled recipe group (what graph_compile
  // creates) or a region loaded from a persisted artifact. Both answer the same
  // launch_recipe call, which is what makes graph_launch work unchanged for a
  // precompiled region.
  using StorageEntry = std::variant<GraphExecsGroup, PrecompiledRegion>;

  size_t add_new_recipe(
      std::shared_ptr<habana_torch::jit::Graph> graph,
      const std::string& parent_graph_name,
      torch::jit::Stack& example_inputs,
      const std::vector<bool>& is_reusable,
      bool dynamic,
      bool inference,
      bool has_preallocated_outputs,
      bool has_randoms,
      InputSymbolIndexMap& in_symbol_idx_map,
      std::vector<habana_helpers::RangeInfo>& range_infos,
      std::vector<int64_t>& const_indexes,
      bool has_dynamic_marked_tensors);
  // Loads <dir>/<key>.{recipe,metadata} into a new storage entry. Throws if the
  // artifact does not load: a region that cannot be loaded must not be launchable.
  size_t add_precompiled_recipe(
      const std::string& dir,
      const std::string& key);
  torch::jit::Stack launch_recipe(
      size_t recipe_id,
      torch::jit::Stack& inputs,
      std::vector<at::Tensor>& outputs,
      std::string& parent_graph_name);
  void reset_seeds();

 private:
  GraphStorage() {};
  GraphStorage(const GraphStorage&) = delete;
  GraphStorage& operator=(const GraphStorage&) = delete;
  GraphStorage(GraphStorage&&) = delete;
  GraphStorage& operator=(GraphStorage&&) = delete;

  std::vector<StorageEntry> m_storage_vec;
};

} // namespace habana::graph
