#include "neuromesh/device/cuda_executor.hpp"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace neuromesh::device {
namespace {
void checked(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
class DeviceScope {
public:
  explicit DeviceScope(int device) { checked(cudaGetDevice(&previous_)); checked(cudaSetDevice(device)); }
  ~DeviceScope() { if (cudaSetDevice(previous_) != cudaSuccess) std::terminate(); }
private:
  int previous_{};
};
Status failure(cudaError_t code) { return {ErrorCode::execution_failed, cudaGetErrorString(code)}; }
}
struct CudaExecutor::Impl {
  struct Job { Launch launch; std::optional<CompletionSignal> signal; };
  explicit Impl(int device) : device_id(device), registry(1) {
    if (device < 0) throw std::invalid_argument("negative CUDA device");
    DeviceScope scope(device_id);
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      checked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      worker = std::thread([this] { run(); });
    } catch (...) {
      if (event) cudaEventDestroy(event);
      cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Impl() {
    close();
    DeviceScope scope(device_id);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
  }
  void run() noexcept {
    // A lost context/device without proof of quiescence is fail-stop. Never mark
    // completion and recycle possibly live memory after a failed fallback sync.
    if (cudaSetDevice(device_id) != cudaSuccess) std::terminate();
    for (;;) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return stopping || pending; });
        if (!pending) return;
        job = std::move(pending);
      }
      Status result;
      try { result = job->launch(stream); }
      catch (const std::exception& error) { result = {ErrorCode::execution_failed, error.what()}; }
      catch (...) { result = {ErrorCode::execution_failed, "unknown launch exception"}; }
      auto code = cudaEventRecord(event, stream);
      if (code == cudaSuccess) code = cudaEventSynchronize(event);
      if (code != cudaSuccess) {
        result = failure(code);
        if (cudaStreamSynchronize(stream) != cudaSuccess) std::terminate();
      }
      // Captured objects are also destroyed after quiescence, before status.
      job->launch = {};
      job->signal->complete(std::move(result));
    }
  }
  void close() {
    std::lock_guard<std::mutex> shutdown(shutdown_mutex);
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    changed.notify_all();
    if (worker.joinable()) worker.join();
    registry.close_and_drain();
  }
  int device_id;
  cudaStream_t stream{};
  cudaEvent_t event{};
  InFlightRegistry registry;
  std::mutex mutex, shutdown_mutex;
  std::condition_variable changed;
  bool stopping{false};
  std::unique_ptr<Job> pending;
  std::thread worker;
};
CudaExecutor::CudaExecutor(int device_id) : impl_(std::make_unique<Impl>(device_id)) {}
CudaExecutor::~CudaExecutor() = default;
CompletionTicket CudaExecutor::submit(std::vector<BufferLease> leases, Launch launch) {
  if (!launch) throw std::invalid_argument("missing CUDA launch function");
  for (const auto& lease : leases)
    if (lease.memory() == MemoryKind::cuda_device && lease.device_id() != impl_->device_id)
      throw std::invalid_argument("CUDA allocation belongs to another device");
  auto job = std::make_unique<Impl::Job>();
  job->launch = std::move(launch);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->stopping) throw std::logic_error("CUDA executor closed");
  auto submission = impl_->registry.reserve(std::move(leases));
  job->signal.emplace(std::move(submission.signal));
  impl_->pending = std::move(job);
  impl_->changed.notify_one();
  return std::move(submission.ticket);
}
void CudaExecutor::close_and_drain() { impl_->close(); }
}  // namespace neuromesh::device
