/*!
 * \file runtime/perfgunner.h
 * \brief Lightweight performance measurement utilities for TileLang.
 *
 * Provides inline GPU kernel timing via HIP events, host CPU wall-clock
 * timing, and snapshot-based resource accounting (VRAM bytes allocated
 * via the HIP runtime, host RSS via /proc/self/status on Linux and
 * GetProcessMemoryInfo on Windows).
 *
 * All measurement is opt-in through TILELANG_ENABLE_PERF.  When disabled
 * every function is a no-op (or returns zeros) and the linker strips
 * the associated HIP/GLIBC calls.
 */
#ifndef TVM_TL_RUNTIME_PERFGUNNER_H_
#define TVM_TL_RUNTIME_PERFGUNNER_H_

#include <tvm/runtime/logging.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace tl {

#ifdef TILELANG_ENABLE_PERF
#define TILELANG_ENABLE_PERF 1
#endif

struct KernelTrace {
  std::string name;
  float gpu_time_us{0.0f};
  float cpu_wall_us{0.0f};
  uint64_t gpu_alloc_bytes{0};
  uint64_t gpu_free_bytes{0};
  uint64_t host_rss_bytes{0};
  int grid_size{0};
  int block_size{0};
};

struct PerfSnapshot {
  uint64_t gpu_alloc_bytes{0};
  uint64_t gpu_free_bytes{0};
  uint64_t host_rss_bytes{0};
  uint64_t host_vsize_bytes{0};
};

class PerfGunner {
 public:
  static PerfGunner &Instance();

  PerfGunner(const PerfGunner &) = delete;
  PerfGunner &operator=(const PerfGunner &) = delete;

  void Start(const std::string &label);
  void Stop();
  void RecordGpuKernel(const std::string &name, int grid, int block,
                        float gpu_us, float cpu_us);
  PerfSnapshot Snapshot() const;
  std::vector<KernelTrace> DrainTraces();
  void Clear();

  bool enabled() const { return enabled_; }
  void SetEnabled(bool on) { enabled_ = on; }

 private:
  PerfGunner();
  ~PerfGunner() = default;

  bool enabled_;
  std::mutex mutex_;
  std::string current_label_;
  std::chrono::high_resolution_clock::time_point start_;
  std::vector<KernelTrace> traces_;
};

struct ScopedPerf {
  explicit ScopedPerf(const std::string &label);
  ~ScopedPerf();
};

PerfSnapshot TakePerfSnapshot();

}  // namespace tl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_TL_RUNTIME_PERFGUNNER_H_
