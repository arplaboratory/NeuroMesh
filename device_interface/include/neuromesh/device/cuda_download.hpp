#pragma once
#include "neuromesh/device/cuda_executor.hpp"
namespace neuromesh::device {
struct DownloadedTensors {
  std::vector<TensorBinding> tensors;
  CompletionTicket completion;
};
// Bounded pinned-host staging for already-completed device outputs. The caller
// must establish producer completion before submit; leases alone do not do so.
class CudaDownload {
public:
  CudaDownload(std::vector<TensorSpec> specs, int device, std::size_t slots,
               std::size_t max_slot_bytes);
  ~CudaDownload();
  CudaDownload(const CudaDownload&) = delete;
  CudaDownload& operator=(const CudaDownload&) = delete;
  // Host outputs must not be read until completion succeeds. Output leases and
  // the executor retain staging even if the caller drops its ticket.
  DownloadedTensors submit(const std::vector<TensorBinding>& sources);
  void close_and_drain();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
