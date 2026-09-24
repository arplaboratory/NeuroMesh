#pragma once
#include "neuromesh/device/completion.hpp"
#include <cuda_runtime_api.h>
#include <functional>

namespace neuromesh::device {
// Optional CUDA layer; the generic tensor/completion target remains CUDA-free.
// Launch callbacks run on the owned worker, not the submitter or a ROS callback.
// Every device access must be ordered onto the provided stream. Other streams
// require an explicit dependency into it before the callback returns.
class CudaExecutor {
public:
  using Launch = std::function<Status(cudaStream_t)>;
  explicit CudaExecutor(int device_id);
  ~CudaExecutor();
  CudaExecutor(const CudaExecutor&) = delete;
  CudaExecutor& operator=(const CudaExecutor&) = delete;
  CompletionTicket submit(std::vector<BufferLease> leases, Launch launch);
  // Reject new work, join completion worker and drain registry. Destruction
  // subsequently releases CUDA resources. Must not be called by a launch callback or allocation deleter.
  void close_and_drain();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace neuromesh::device
