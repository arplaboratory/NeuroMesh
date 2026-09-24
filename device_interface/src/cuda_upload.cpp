#include "neuromesh/device/cuda_upload.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
namespace neuromesh::device {
namespace {
void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct DeviceScope {
  int previous{};
  explicit DeviceScope(int device) { checked(cudaGetDevice(&previous)); checked(cudaSetDevice(device)); }
  ~DeviceScope() { if (cudaSetDevice(previous) != cudaSuccess) std::terminate(); }
};
struct Slot {
  int device;
  void* host{};
  void* gpu{};
  Slot(int d, std::size_t size) : device(d) {
    DeviceScope scope(device);
    checked(cudaMallocHost(&host,size));
    const auto error = cudaMalloc(&gpu,size);
    if (error != cudaSuccess) { cudaFreeHost(host); host = nullptr; checked(error); }
    std::memset(host,0,size); // alignment padding is initialized once
  }
  ~Slot() {
    DeviceScope scope(device);
    if (cudaFree(gpu) != cudaSuccess || cudaFreeHost(host) != cudaSuccess) std::terminate();
  }
};
}
struct CudaUpload::Impl {
  std::vector<TensorSpec> specs;
  std::vector<std::size_t> offsets, sizes;
  std::vector<std::shared_ptr<Slot>> pool;
  std::size_t bytes{};
  int device;
  std::mutex mutex;
  bool closed{false};
  // Destroy executor before pool; external output leases can outlive both.
  std::unique_ptr<CudaExecutor> executor;
  Impl(std::vector<TensorSpec> s, int d, std::size_t slots, std::size_t budget)
      : specs(std::move(s)), device(d) {
    if (specs.empty() || specs.size() > 8 || !slots || slots > 8 || !budget || d < 0)
      throw std::invalid_argument("invalid CUDA upload pool configuration");
    std::unordered_set<std::string> names;
    for (const auto& spec : specs) {
      std::size_t size = 0;
      const auto status = tensor_bytes(spec,size);
      if (!status || spec.memory != MemoryKind::host || spec.device_id != -1 || !names.insert(spec.name).second)
        throw std::invalid_argument("invalid host upload profile");
      const auto padding = (256 - bytes % 256) % 256;
      if (padding > budget - bytes || size > budget - bytes - padding)
        throw std::length_error("upload slot budget exceeded");
      bytes += padding; offsets.push_back(bytes); sizes.push_back(size); bytes += size;
    }
    for (std::size_t i = 0; i < slots; ++i) pool.push_back(std::make_shared<Slot>(device,bytes));
    executor = std::make_unique<CudaExecutor>(device);
  }
};
CudaUpload::CudaUpload(std::vector<TensorSpec> specs, int device, std::size_t slots, std::size_t budget)
    : impl_(std::make_unique<Impl>(std::move(specs),device,slots,budget)) {}
CudaUpload::~CudaUpload() { close_and_drain(); }
void CudaUpload::close_and_drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->closed = true;
  impl_->executor->close_and_drain();
}
UploadedTensors CudaUpload::submit(const std::vector<TensorBinding>& sources) {
  const auto status = validate_bindings(impl_->specs,sources);
  if (!status) throw std::invalid_argument(status.message);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed) throw std::logic_error("CUDA uploader closed");
  const auto available = std::find_if(impl_->pool.begin(),impl_->pool.end(),
    [](const auto& slot) { return slot.use_count() == 1; });
  if (available == impl_->pool.end()) throw std::length_error("CUDA upload pool exhausted");
  const auto slot = *available;
  std::vector<TensorBinding> outputs;
  for (std::size_t i = 0; i < impl_->specs.size(); ++i) {
    const auto source = std::find_if(sources.begin(),sources.end(),
      [&](const auto& value) { return value.spec.name == impl_->specs[i].name; });
    std::memcpy(static_cast<std::uint8_t*>(slot->host) + impl_->offsets[i],
      static_cast<const std::uint8_t*>(source->storage.base()) + source->storage.offset(),impl_->sizes[i]);
    auto spec = impl_->specs[i]; spec.memory = MemoryKind::cuda_device; spec.device_id = impl_->device;
    outputs.push_back({std::move(spec),BufferLease(slot,slot->gpu,impl_->bytes,
      MemoryKind::cuda_device,impl_->device,impl_->offsets[i])});
  }
  std::vector<BufferLease> leases{
    BufferLease(slot,slot->host,impl_->bytes,MemoryKind::pinned_host,-1),
    BufferLease(slot,slot->gpu,impl_->bytes,MemoryKind::cuda_device,impl_->device)};
  const auto bytes = impl_->bytes;
  auto ticket = impl_->executor->submit(std::move(leases),[slot,bytes](cudaStream_t stream) {
    const auto error = cudaMemcpyAsync(slot->gpu,slot->host,bytes,cudaMemcpyHostToDevice,stream);
    if (error != cudaSuccess) return Status{ErrorCode::execution_failed,cudaGetErrorString(error)};
    return Status{};
  });
  return {std::move(outputs),std::move(ticket)};
}
}
