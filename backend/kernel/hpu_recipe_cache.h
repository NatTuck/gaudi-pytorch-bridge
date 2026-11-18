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
 */
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include "habana_helpers/habana_serialization/include/habana_serialization/recipe_cache.h"

#define PGM_LRU_MAX_EAGER_NRECIPES 100000
#define PGM_LRU_MAX_LAZY_NRECIPES 30000
#define PGM_LRU_MIN_NRECIPES 3

namespace habana {

struct RecipeHolder;
struct RecipeArgumentSpec;

struct RecipeArgumentSpecHash {
 public:
  size_t operator()(const std::shared_ptr<RecipeArgumentSpec>& v) const;
};

// Comparator for RecipeArgumentSpec
struct RecipeArgumentSpecEqual {
 public:
  bool operator()(
      const std::shared_ptr<RecipeArgumentSpec>& v1,
      const std::shared_ptr<RecipeArgumentSpec>& v2) const;
};

class DiskCache {
 public:
  DiskCache(const serialization::RecipeCacheConfig& recipe_cache_config);
  // `compile_key` is the CompileInputs record key, computed ONCE at the request
  // site (`HabanaLaunchOpPT::GetCachedRecipe`) and used for BOTH the store and
  // the load. Computing it separately on each side is what broke the disk
  // cache: `Add` wrote under the record key while `Find` looked under the graph
  // hash, so every store was invisible to every load and the cache could never
  // hit.
  void Add(
      const RecipeHolder& val,
      const RecipeArgumentSpec& spec,
      const std::string& compile_key);
  // THE CACHE ARTIFACT IS THE SET OF RECIPES, plus whatever is needed to run
  // them all on a hit. Today that set is a SINGLETON: a `Region.__call__` traces
  // to one FX frame, the partitioner leaves it as one `fused_0`, and one recipe
  // is produced. Measured, not assumed: 2789 graph names across the boot logs,
  // every one `_fused_0`; five distinct recipe names, all `fused_0`; zero
  // occurrences of `fused_1` or higher.
  //
  // So there is no routing to store and no replay driver to write, and building
  // that machinery now would be a serialisation format and an executor to carry
  // a set that always has one element. The shape is chosen deliberately, not by
  // accident: if a split ever appears, this is where the plan belongs -- an
  // ordered partition list plus, per partition, the source of each input slot
  // (a region argument, an earlier partition's output, or a constant).
  std::shared_ptr<RecipeHolder> Find(const std::string& compile_key,
                                     const std::string& graph_name);
  // in case RecipeValueSpec creation failed, DiskCache is leaving lock files on
  // disk. This ensures a cleanup.
  void flush();

 private:
  serialization::RecipeCache recipe_cache_;
  // library specific suffix to determine for what TF, Synapse, etc. the cache
  // entry was produced
  std::string cache_id_suffix_;
};

class RecipeCacheLRU {
 public:
  bool empty() {
    return (map_.empty());
  }

  size_t get_length() {
    return map_.size();
  }

  const std::string& get_cache_path() const {
    return recipe_cache_config_.path();
  }

  void clear() {
    std::lock_guard<std::mutex> lg(mutex_);
    map_.clear();
    list_.clear();
  }

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  using dropped_recipe_t = std::optional<std::pair<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeHolder>>>;

  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeHolder>& val,
      const std::string& compile_key);
  std::shared_ptr<RecipeHolder> get(
      std::shared_ptr<RecipeArgumentSpec>& key,
      const std::string& compile_key,
      const std::string& graph_name);
  dropped_recipe_t drop_lru();
  void remove_oldest();
  void ResetDiskCache();
  void DeleteDiskCache();
  void FlushDiskCache();
  void UpdateCachePath(const std::string& new_path);
  void Serialize();
  void Deserialize();

  static void SetHostMemoryThreshold(
      uint32_t host_memory_threshold = default_host_memory_threshold);

  size_t Size() const;
  size_t SynapseRecipeSize() const;
  static void DumpRecipeMemoryStat();
  static void DumpSynapseRecipeMemoryStat();
  static void DumpDynamicShapeMemoryStat();

  // friend std::ostream& operator<<(std::ostream& O, const RecipeCacheLRU& v);
  RecipeCacheLRU();
  ~RecipeCacheLRU() = default;

 private:
  RecipeCacheLRU(const RecipeCacheLRU&) = delete;
  RecipeCacheLRU& operator=(const RecipeCacheLRU&) = delete;
  dropped_recipe_t drop_lru_impl(bool mem_exhausted = false);
  void insert(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeHolder>& val);
  void InitDiskCache();

  std::mutex mutex_;
  static size_t max_size_;
  std::unique_ptr<DiskCache> disk_cache_;
  static const uint32_t default_host_memory_threshold = 90;

  std::list<std::pair<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeHolder>>>
      list_;

  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::list<std::pair<
          std::shared_ptr<RecipeArgumentSpec>,
          std::shared_ptr<RecipeHolder>>>::iterator,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;

  serialization::RecipeCacheConfig recipe_cache_config_;
};

struct RecipeValueSpec;

class TemporaryRecipeStore {
 public:
  static TemporaryRecipeStore& get() {
    static TemporaryRecipeStore instance;
    return instance;
  }

  void Add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::future<void>&& recipe_ready,
      std::shared_ptr<RecipeValueSpec> rvs);

  void Remove(std::shared_ptr<RecipeArgumentSpec>& key) {
    std::lock_guard lg(mtx_);
    map_.erase(key);
  }

  std::shared_ptr<RecipeValueSpec> GetRVS(
      std::shared_ptr<RecipeArgumentSpec>& key);

  void Wait(std::shared_ptr<RecipeArgumentSpec>& key);

 private:
  TemporaryRecipeStore() = default;

  std::mutex mtx_;
  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::pair<std::future<void>, std::shared_ptr<RecipeValueSpec>>,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;
};

} // namespace habana
