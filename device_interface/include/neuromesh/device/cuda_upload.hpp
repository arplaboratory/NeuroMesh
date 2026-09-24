#pragma once
#include "neuromesh/device/cuda_executor.hpp"
namespace neuromesh::device {
struct UploadedTensors {
  std::vector<TensorBinding> tensors;
  CompletionTicket completion;
};
// Fixed named tensor profile with a bounded preallocated pinned/device pool.
// Submit is intended for a worker: it stages host bytes synchronously, then
// returns a ticket for one asynchronous slab transfer on CudaExecutor's stream.
class CudaUpload {
public:
  // source_specs must be host/-1; slots in [1,8]. max_slot_bytes bounds checked
  // allocation arithmetic (including internal 256-byte alignment padding).
  CudaUpload(std::vector<TensorSpec> source_specs, int device_id,
             std::size_t slots, std::size_t max_slot_bytes);
  ~CudaUpload();
  CudaUpload(const CudaUpload&) = delete;
  CudaUpload& operator=(const CudaUpload&) = delete;
  // Rejects invalid bindings, exhausted pool, executor busy or closed state.
  // Output leases pin slots; callers must bound cached/request snapshot count.
  UploadedTensors submit(const std::vector<TensorBinding>& sources);
  void close_and_drain();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
