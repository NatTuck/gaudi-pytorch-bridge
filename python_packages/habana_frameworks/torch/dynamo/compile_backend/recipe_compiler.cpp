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

#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include <string>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_eager/graph_storage.h"
#include "backend/synapse_helpers/region_identity_registry.h"

#include "habana_helpers/logging.h"
namespace {

using InputSymbolIndexMap = std::unordered_map<std::string, int64_t>;
struct EmptyBatchData {
  std::vector<int64_t> size;
  py::object dtype;
  std::optional<std::vector<int64_t>> stride;
  EmptyBatchData(
      std::vector<int64_t> size,
      py::object dtype,
      std::optional<std::vector<int64_t>> stride)
      : size(std::move(size)), dtype(dtype), stride(std::move(stride)) {}
};

std::vector<at::Tensor> batch_empty(const std::vector<EmptyBatchData>& batch) {
  auto* allocator = habana::getHABANADeviceAllocator();
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);

  std::vector<at::Tensor> result;
  for (const auto& el : batch) {
    at::ScalarType dtype_c =
        reinterpret_cast<THPDtype*>(el.dtype.ptr())->scalar_type;
    auto dtype = dtype_or_default(dtype_c);
    HABANA_ASSERT(habana_helpers::is_supported_type(dtype));

    if (!el.stride.has_value()) {
      result.emplace_back(
          at::detail::empty_generic(el.size, allocator, hpu_ks, dtype, {}));
    } else {
      result.emplace_back(
          at::detail::empty_strided_generic(
              el.size, el.stride.value(), allocator, hpu_ks, dtype));
    }
  }
  return result;
}

std::size_t calculate_symval_hashcode(
    const py::tuple& inputs,
    InputSymbolIndexMap& symbol_idx_map) {
  size_t hash_code = 0;
  std::for_each(
      symbol_idx_map.begin(),
      symbol_idx_map.end(),
      [&](const std::pair<std::string, int64_t>& p) {
        int64_t scalar_index = p.second;
        HABANA_ASSERT(
            scalar_index >= 0 ||
                static_cast<size_t>(scalar_index) < inputs.size(),
            "Symbol index received is out of bounds!!",
            scalar_index);
        const auto& obj = inputs[scalar_index];
        HABANA_ASSERT(
            py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj),
            "Expected scalar but got non-scalar object!!",
            scalar_index);
        double value;
        if (py::isinstance<py::float_>(obj)) {
          value = obj.cast<double>(); // Cast directly to double if it's a float
        } else {
          value = static_cast<double>(
              obj.cast<int64_t>()); // Cast to double if it's an int
        }

        auto symbol_hash = c10::get_hash(p.first);
        hash_code = c10::hash_combine(hash_code, symbol_hash);
        auto value_hash = c10::get_hash(value);
        hash_code = c10::hash_combine(hash_code, value_hash);
      });
  return hash_code;
}
}; // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "graph_compile",
      [](py::object graph_obj,
         const std::string& parent_graph_name,
         const py::tuple& inputs,
         const py::tuple& is_reusable,
         bool dynamic,
         bool inference,
         bool has_preallocated_outputs,
         bool has_randoms,
         InputSymbolIndexMap& in_symbol_idx_map,
         std::vector<habana_helpers::RangeInfo>& range_infos,
         std::vector<int64_t>& const_indexes,
         bool has_dynamic_marked_tensors) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (const auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }
        std::vector<bool> is_reusable_vec;
        is_reusable_vec.reserve(is_reusable.size());
        for (const auto& obj : is_reusable) {
          is_reusable_vec.push_back(obj.cast<bool>());
        }

        if (graph_obj.is_none()) {
          throw std::invalid_argument("Graph object is None");
        }
        std::shared_ptr<habana_torch::jit::Graph> graph;
        try {
          // get raw_ptr
          auto* raw_ptr = graph_obj.cast<habana_torch::jit::Graph*>();
          if (raw_ptr) {
            graph = std::shared_ptr<habana_torch::jit::Graph>(
                raw_ptr, [](habana_torch::jit::Graph*) {});
          } else {
            throw std::invalid_argument("All cast methods failed");
          }
        } catch (const py::cast_error& e3) {
          throw std::invalid_argument(
              "Unable to convert graph object to expected type");
        }

        if (!graph) {
          HABANA_ASSERT(0, "Received null graph pointer in graph_compile");
          throw std::invalid_argument(
              "Graph conversion resulted in null pointer");
        }

        auto& graph_storage{habana::graph::GraphStorage::get()};
        return graph_storage.add_new_recipe(
            graph,
            parent_graph_name,
            stack,
            is_reusable_vec,
            dynamic,
            inference,
            has_preallocated_outputs,
            has_randoms,
            in_symbol_idx_map,
            range_infos,
            const_indexes,
            has_dynamic_marked_tensors);
      },
      py::return_value_policy::copy,
      py::arg("graph"),
      py::arg("parent_graph_name"),
      py::arg("inputs"),
      py::arg("is_reusable"),
      py::arg("dynamic"),
      py::arg("inference"),
      py::arg("has_preallocated_outputs"),
      py::arg("has_randoms"),
      py::arg("in_symbol_idx_map"),
      py::arg("range_infos"),
      py::arg("const_indexes"),
      py::arg("has_dynamic_marked_tensors"));
  m.def(
      "graph_launch",
      [](size_t recipe_id,
         const py::tuple& inputs,
         std::vector<at::Tensor>& outputs,
         std::string& parent_graph_name) {
        torch::jit::Stack stack;
        stack.reserve(inputs.size());
        for (const auto& obj : inputs) {
          stack.push_back(torch::jit::toTypeInferredIValue(obj));
        }

        auto& graph_storage{habana::graph::GraphStorage::get()};
        stack = graph_storage.launch_recipe(
            recipe_id, stack, outputs, parent_graph_name);

        if (outputs.empty()) {
          return torch::jit::createPyObjectForStack(std::move(stack));
        }

        torch::jit::Stack out_stack;
        for (size_t idx = 0; idx < outputs.size(); idx++) {
          out_stack.emplace_back(outputs[idx]);
        }
        return torch::jit::createPyObjectForStack(std::move(out_stack));
      },
      py::return_value_policy::copy,
      py::arg("recipe_id"),
      py::arg("inputs"),
      py::arg("outputs"),
      py::arg("parent_graph_name"));
  m.def("reset_seeds", []() {
    auto& graph_storage{habana::graph::GraphStorage::get()};
    graph_storage.reset_seeds();
  });
  // Load a persisted (recipe, metadata) pair into GraphStorage as a region.
  // The id it returns is launched with the ordinary graph_launch above, which is
  // the point: the launch path is unchanged, only where the RecipeHolder came
  // from differs.
  m.def(
      "precompiled_load",
      [](const std::string& dir, const std::string& key) {
        auto& graph_storage{habana::graph::GraphStorage::get()};
        return graph_storage.add_precompiled_recipe(dir, key);
      },
      py::arg("dir"),
      py::arg("key"));
  py::class_<EmptyBatchData>(m, "EmptyBatchData")
      .def(
          py::init<
              std::vector<int64_t>,
              py::object,
              std::optional<std::vector<int64_t>>>())
      .def_readwrite("size", &EmptyBatchData::size);
  m.def("batch_empty", &batch_empty, "Create empty tensors");
  py::class_<habana_helpers::RangeInfo>(m, "RangeInfo")
      .def(
          py::init<
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::string,
              std::string,
              int>())
      .def_readwrite("min_shape", &habana_helpers::RangeInfo::min_shape)
      .def_readwrite("max_shape", &habana_helpers::RangeInfo::max_shape)
      .def_readwrite("expr", &habana_helpers::RangeInfo::expr)
      .def_readwrite("expr_strides", &habana_helpers::RangeInfo::expr_strides)
      .def_readwrite("index", &habana_helpers::RangeInfo::index);
  m.def(
      "calculate_symval_hashcode",
      &calculate_symval_hashcode,
      "Calculate hash key of graph input tensor shapes and strides");
  // Clause 3: register the region identity for a graph name before it compiles,
  // so that `RecipeLauncher::Serialize` can write it into the recipe's
  // `.metadata` and the identity travels with the recipe into the cache.
  // Called from compile_backend/launch_ledger.py, which is the only place that
  // has the FX stack_trace. See backend/synapse_helpers/region_identity_registry.h.
  m.def(
      "note_region",
      [](const std::string& graph_name,
         const std::string& name,
         const std::string& source,
         uint32_t index,
         const std::string& compile_key) {
        habana::region_identity::note(graph_name, name, source, index, compile_key);
      },
      "Register (graph name -> region name, source file:line, index) so the "
      "identity is written into the recipe cache entry",
      py::arg("graph_name"),
      py::arg("name"),
      py::arg("source"),
      py::arg("index"),
      py::arg("compile_key") = "");
  m.def(
      "region_identity_count",
      []() { return habana::region_identity::size(); },
      "How many region identities this process registered");
}
