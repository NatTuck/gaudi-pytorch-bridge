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

// Launch ledger: one fixed-size record per recipe launch, plus a side table
// written once per recipe compile.
//
// Why this point: `synapse_helpers::graph::launch` is the only function every
// launch passes through. The compiled-region path reaches it as
//     GraphExecsGroup::launch -> GraphExec::launch -> RecipeLauncher::Launch -> graph::launch
// and the eager path as
//     EagerExec::launch -> HabanaLaunchOpPT -> RecipeLauncher::Launch -> graph::launch
// (the lazy/operator path enters one frame earlier, at
// `synapse_helpers::recipe::launch`, which also ends in `graph::launch`).
// So a record here is exactly one record per launch, and a launch that does not
// appear here is not a recipe launch at all.
//
// Why a side table: the launch point knows the recipe, not the region. The
// region identity (name + `foo.N` index) is computed in Python at compile time
// by tools/region_identity. The two halves are joined afterwards, on a key that
// both sides mint from the same string. The hot path therefore writes one
// uint64 that was precomputed in `graph::compile`, and no strings, no
// formatting, no allocation.
//
// Header-only on purpose: adding a translation unit would need a cmake re-run,
// and this build tree's configure-time variables are not in the environment or
// in the cache (see analysis/precompiled-region/REPORT.md section 8).
//
// Off by default. When `VLLM_HPU_LAUNCH_LEDGER` is unset, `record_launch` is a
// load of a function-local static bool and a not-taken branch.
//
// Files, all per-process so that several processes cannot interleave records:
//   <dir>/launches.<pid>.bin   header + ring of LaunchRecord
//   <dir>/compiles.<pid>.bin   header + array of CompileRecord
// MAP_SHARED, so the records are in the page cache even if the process is
// SIGKILLed (which is how a vLLM EngineCore usually dies).

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace habana::ledger {

// One launch. 32 bytes: half a cache line, trivially copyable so a reader can
// memcpy a whole page out of the mmap.
struct LaunchRecord {
  uint64_t seq;   // index in the ring; a reader uses it to detect a torn slot
  uint64_t t_ns;  // steady_clock, nanoseconds
  uint64_t key;   // FNV-1a of the recipe name, minted at compile time
  uint32_t tid;   // ledger-local thread id (1, 2, 3, ... in order of first use)
  uint32_t flags; // reserved, 0
};
static_assert(sizeof(LaunchRecord) == 32, "LaunchRecord must stay 32 bytes");

// One compiled recipe. Written once, at compile time. 256 bytes.
//
// Two names, because one is not enough:
//   name        the unique recipe name, which embeds the recipe id for a
//               compiled region's recipe and is a bare counter for an eager
//               op's (get_unique_recipe_name returns `dir + suffix++` when
//               eager_mode is set, so the op name is not in it);
//   graph_name  the graph's own name, which is the aten symbol for an eager op
//               (`aten::mul_57`) and `graph_0004_fused_0_syn_1004000_1004000`
//               for a compiled region. This is the field that names the op.
struct CompileRecord {
  uint64_t seq;       // publish order
  uint64_t key;       // FNV-1a of name
  uint64_t t_ns;      // when it was compiled
  uint32_t eager;     // 1 if synapse_helpers::graph was created in eager mode
  uint32_t name_len;  // bytes of `name` that are valid
  uint32_t gname_len; // bytes of `graph_name` that are valid
  uint32_t pad;
  char name[108];
  char graph_name[108];
};
static_assert(sizeof(CompileRecord) == 256, "CompileRecord must stay 256 bytes");

// Records start at this offset in both files (header + counter, padded to a
// 64-byte boundary so the first record is aligned).
constexpr size_t kDataOffset = 64;

struct LedgerHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t cap;         // number of records (a power of two)
  uint64_t t0_wall_ns;  // CLOCK_REALTIME at init, to place the run on a clock
  uint64_t t0_steady_ns;// steady_clock at init, the records' time origin
  uint64_t written;     // append counter; use __atomic_fetch_add on it
  uint64_t pad;
};
static_assert(sizeof(LedgerHeader) == 48, "LedgerHeader must stay 48 bytes");
static_assert(kDataOffset >= sizeof(LedgerHeader), "header must fit before the data");

constexpr uint64_t kMagic = 0x3147454c48435541ull; // "AUCHLEG1" little-endian
constexpr uint32_t kVersion = 1;
constexpr uint32_t kDefaultCap = 1u << 20;        // 1 M launches = 32 MiB
constexpr uint32_t kDefaultCompileCap = 1u << 14; // 16 K recipes = 4 MiB

inline uint64_t fnv1a(const char* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(p[i]));
    h *= 0x100000001b3ull;
  }
  return h;
}
inline uint64_t fnv1a(const std::string& s) {
  return fnv1a(s.data(), s.size());
}

inline uint64_t now_steady_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline uint32_t next_thread_id() {
  static std::atomic<uint32_t> counter{0};
  thread_local uint32_t id = 0;
  if (id == 0) {
    id = counter.fetch_add(1, std::memory_order_relaxed) + 1;
  }
  return id;
}

namespace detail {

inline uint32_t env_u32(const char* name, uint32_t dflt) {
  const char* v = ::getenv(name);
  if (v == nullptr || *v == '\0') {
    return dflt;
  }
  const long parsed = ::strtol(v, nullptr, 10);
  if (parsed <= 0) {
    return dflt;
  }
  return static_cast<uint32_t>(parsed);
}

inline uint32_t floor_pow2(uint32_t v) {
  uint32_t p = 1;
  while ((p << 1) != 0 && (p << 1) <= v) {
    p <<= 1;
  }
  return p;
}

struct State {
  LaunchRecord* ring{nullptr};
  uint64_t mask{0};
  LedgerHeader* launch_hdr{nullptr};
  CompileRecord* comp{nullptr};
  uint64_t comp_mask{0};
  LedgerHeader* compile_hdr{nullptr};

  State() {
    const char* dir = ::getenv("VLLM_HPU_LAUNCH_LEDGER");
    if (dir == nullptr || *dir == '\0') {
      return;
    }
    const uint32_t cap =
        floor_pow2(env_u32("VLLM_HPU_LAUNCH_LEDGER_CAP", kDefaultCap));
    const uint32_t ccap =
        floor_pow2(env_u32("VLLM_HPU_LAUNCH_LEDGER_COMPILE_CAP", kDefaultCompileCap));
    const long pid = static_cast<long>(::getpid());
    const uint64_t t0w = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    const uint64_t t0s = now_steady_ns();

    ring = map_ring<LaunchRecord>(
        std::string(dir) + "/launches." + std::to_string(pid) + ".bin",
        cap, t0w, t0s, &launch_hdr);
    comp = map_ring<CompileRecord>(
        std::string(dir) + "/compiles." + std::to_string(pid) + ".bin",
        ccap, t0w, t0s, &compile_hdr);
    if (ring != nullptr) {
      mask = static_cast<uint64_t>(cap) - 1;
    }
    if (comp != nullptr) {
      comp_mask = static_cast<uint64_t>(ccap) - 1;
    }
  }

  // Returns the record array, or nullptr. On any failure the ledger is simply
  // off: a measurement instrument must never take the process down.
  static void* map_file(const std::string& path, size_t bytes) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      return nullptr;
    }
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      ::close(fd);
      return nullptr;
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd); // the mapping keeps the pages
    if (p == MAP_FAILED) {
      return nullptr;
    }
    return p;
  }

  template <typename RecT>
  static RecT* map_ring(
      const std::string& path,
      uint32_t cap,
      uint64_t t0_wall,
      uint64_t t0_steady,
      LedgerHeader** hdr_out) {
    const size_t bytes = kDataOffset + static_cast<size_t>(cap) * sizeof(RecT);
    auto* base = static_cast<unsigned char*>(map_file(path, bytes));
    if (base == nullptr) {
      return nullptr;
    }
    auto* hdr = reinterpret_cast<LedgerHeader*>(base);
    hdr->magic = kMagic;
    hdr->version = kVersion;
    hdr->cap = cap;
    hdr->t0_wall_ns = t0_wall;
    hdr->t0_steady_ns = t0_steady;
    hdr->written = 0;
    hdr->pad = 0;
    *hdr_out = hdr;
    return reinterpret_cast<RecT*>(base + kDataOffset);
  }

};

inline State& state() {
  static State s;
  return s;
}

} // namespace detail

// One relaxed load of a function-local static and a not-taken branch when off.
inline bool enabled() {
  static const bool on = [] {
    const char* d = ::getenv("VLLM_HPU_LAUNCH_LEDGER");
    return d != nullptr && *d != '\0';
  }();
  return on;
}

// The hot path. Callers pass the key that `graph::compile` precomputed.
inline void record_launch(uint64_t key) {
  if (!enabled()) {
    return;
  }
  detail::State& s = detail::state();
  if (s.ring == nullptr || s.launch_hdr == nullptr) {
    return;
  }
  LedgerHeader* hdr = s.launch_hdr;
  const uint64_t seq =
      __atomic_fetch_add(&hdr->written, 1, __ATOMIC_RELAXED);
  LaunchRecord& r = s.ring[seq & s.mask];
  r.t_ns = now_steady_ns();
  r.key = key;
  r.tid = next_thread_id();
  r.flags = 0;
  // Publish last, so a reader that validates `seq` sees a whole record.
  __atomic_store_n(&r.seq, seq, __ATOMIC_RELEASE);
}

// Compile-time half. Returns the key, whether or not the ledger is on, so the
// caller can store it unconditionally and the launch path never hashes.
inline uint64_t note_compile(
    const std::string& name,
    const std::string& graph_name,
    bool eager) {
  const uint64_t key = fnv1a(name);
  if (!enabled()) {
    return key;
  }
  detail::State& s = detail::state();
  if (s.comp == nullptr || s.compile_hdr == nullptr) {
    return key;
  }
  LedgerHeader* hdr = s.compile_hdr;
  const uint64_t seq = __atomic_fetch_add(&hdr->written, 1, __ATOMIC_RELAXED);
  CompileRecord& c = s.comp[seq & s.comp_mask];
  c.key = key;
  c.t_ns = now_steady_ns();
  c.eager = eager ? 1u : 0u;
  const size_t n = name.size() < sizeof(c.name) ? name.size() : sizeof(c.name) - 1;
  ::memcpy(c.name, name.data(), n);
  c.name[n] = '\0';
  c.name_len = static_cast<uint32_t>(n);
  const size_t g =
      graph_name.size() < sizeof(c.graph_name) ? graph_name.size() : sizeof(c.graph_name) - 1;
  ::memcpy(c.graph_name, graph_name.data(), g);
  c.graph_name[g] = '\0';
  c.gname_len = static_cast<uint32_t>(g);
  c.pad = 0;
  __atomic_store_n(&c.seq, seq, __ATOMIC_RELEASE);
  return key;
}

} // namespace habana::ledger
