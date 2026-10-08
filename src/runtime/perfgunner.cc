/*!
 * \file runtime/perfgunner.cc
 * \brief Implementation of PerfGunner — see perfgunner.h.
 */
#include "perfgunner.h"
#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <fstream>
#include <sstream>

namespace tvm {
namespace runtime {
namespace tl {

namespace {

uint64_t GetProcessRss() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS info;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return static_cast<uint64_t>(info.WorkingSetSize);
  }
  return 0;
#else
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
  }
  return 0;
#endif
}

uint64_t GetProcessVSize() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS info;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return static_cast<uint64_t>(info.PagefileUsage);
  }
  return 0;
#else
  std::ifstream stat("/proc/self/stat");
  if (stat.is_open()) {
    std::string line;
    std::getline(stat, line);
    std::istringstream iss(line);
    std::string token;
    for (int i = 0; i < 22; ++i) {
      std::getline(iss, token, ' ');
    }
    std::getline(iss, token, ' ');
    return std::stoull(token) * getpagesize();
  }
  return 0;
#endif
}

}  // namespace

PerfGunner &PerfGunner::Instance() {
  static PerfGunner instance;
  return instance;
}

PerfGunner::PerfGunner() : enabled_(false) {
  const char *env = std::getenv("TILELANG_PERF");
  enabled_ = (env != nullptr && (std::string(env) == "1" ||
                                   std::string(env) == "true" ||
                                   std::string(env) == "on"));
}

void PerfGunner::Start(const std::string &label) {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  current_label_ = label;
  start_ = std::chrono::high_resolution_clock::now();
}

void PerfGunner::Stop() {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto end = std::chrono::high_resolution_clock::now();
  float elapsed_us =
      std::chrono::duration<float, std::micro>(end - start_).count();

  auto snap = Snapshot();
  traces_.push_back(KernelTrace{
      .name = current_label_,
      .cpu_wall_us = elapsed_us,
      .gpu_alloc_bytes = snap.gpu_alloc_bytes,
      .gpu_free_bytes = snap.gpu_free_bytes,
      .host_rss_bytes = snap.host_rss_bytes,
  });
  current_label_.clear();
}

void PerfGunner::RecordGpuKernel(const std::string &name, int grid, int block,
                                  float gpu_us, float cpu_us) {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto snap = Snapshot();
  traces_.push_back(KernelTrace{
      .name = name,
      .gpu_time_us = gpu_us,
      .cpu_wall_us = cpu_us,
      .gpu_alloc_bytes = snap.gpu_alloc_bytes,
      .gpu_free_bytes = snap.gpu_free_bytes,
      .host_rss_bytes = snap.host_rss_bytes,
      .grid_size = grid,
      .block_size = block,
  });
}

PerfSnapshot PerfGunner::Snapshot() const {
  PerfSnapshot snap;
  snap.host_rss_bytes = GetProcessRss();
  snap.host_vsize_bytes = GetProcessVSize();
#ifdef USE_ROCM
  {
    uint64_t free = 0, total = 0;
    if (hipMemGetMemPoolInfo(nullptr, &free, &total) == hipSuccess) {
      snap.gpu_alloc_bytes = total - free;
      snap.gpu_free_bytes = free;
    }
  }
#endif
  return snap;
}

std::vector<KernelTrace> PerfGunner::DrainTraces() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = std::move(traces_);
  traces_.clear();
  return out;
}

void PerfGunner::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  traces_.clear();
  current_label_.clear();
}

ScopedPerf::ScopedPerf(const std::string &label) {
  PerfGunner::Instance().Start(label);
}

ScopedPerf::~ScopedPerf() {
  PerfGunner::Instance().Stop();
}

PerfSnapshot TakePerfSnapshot() {
  return PerfGunner::Instance().Snapshot();
}

/*!
 * \brief Return a perf snapshot as a JSON string for Python consumption.
 * Fields: gpu_alloc_bytes, gpu_free_bytes, host_rss_bytes, host_vsize_bytes
 */
std::string PerfSnapshotAsJson(const PerfSnapshot &snap) {
  std::ostringstream os;
  os << "{\"gpu_alloc_bytes\":" << snap.gpu_alloc_bytes
     << ",\"gpu_free_bytes\":" << snap.gpu_free_bytes
     << ",\"host_rss_bytes\":" << snap.host_rss_bytes
     << ",\"host_vsize_bytes\":" << snap.host_vsize_bytes << "}";
  return os.str();
}

}  // namespace tl
}  // namespace runtime
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  using namespace tvm::runtime::tl;

  refl::GlobalDef()
      .def("tl.PerfEnable",
           [](int flag) {
             PerfGunner::Instance().SetEnabled(flag != 0);
           })
      .def("tl.PerfIsEnable",
           []() { return PerfGunner::Instance().enabled(); })
      .def("tl.PerfStart",
           [](tvm::ffi::String label) {
             PerfGunner::Instance().Start(std::string(label));
           })
      .def("tl.PerfStop",
           []() { PerfGunner::Instance().Stop(); })
      .def("tl.PerfSnapshot",
           []() -> tvm::ffi::String {
             auto snap = PerfGunner::Instance().Snapshot();
             return PerfSnapshotAsJson(snap).c_str();
           })
      .def("tl.PerfDrainTraces",
           []() -> tvm::ffi::String {
             auto traces = PerfGunner::Instance().DrainTraces();
             std::ostringstream os;
             os << "[";
             for (size_t i = 0; i < traces.size(); ++i) {
               if (i > 0) os << ",";
               os << "{\"name\":\"" << traces[i].name << "\""
                  << ",\"gpu_time_us\":" << traces[i].gpu_time_us
                  << ",\"cpu_wall_us\":" << traces[i].cpu_wall_us
                  << ",\"gpu_alloc_bytes\":" << traces[i].gpu_alloc_bytes
                  << ",\"gpu_free_bytes\":" << traces[i].gpu_free_bytes
                  << ",\"host_rss_bytes\":" << traces[i].host_rss_bytes
                  << ",\"grid_size\":" << traces[i].grid_size
                  << ",\"block_size\":" << traces[i].block_size << "}";
             }
             os << "]";
             return tvm::ffi::String(os.str());
           });
}
