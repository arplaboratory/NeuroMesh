#include "neuromesh/device/cuda_download.hpp"
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
namespace neuromesh::device {
namespace {
struct PinnedSlot {
  void* data{};
  explicit PinnedSlot(std::size_t bytes) {
    const auto error=cudaMallocHost(&data,bytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }
  ~PinnedSlot(){if(cudaFreeHost(data)!=cudaSuccess) std::terminate();}
};
}
struct CudaDownload::Impl {
  std::vector<TensorSpec> specs;
  std::vector<std::size_t> offsets,sizes;
  std::vector<std::shared_ptr<PinnedSlot>> pool;
  std::size_t bytes{};
  std::mutex mutex;
  bool closed{};
  std::unique_ptr<CudaExecutor> executor;
  Impl(std::vector<TensorSpec> s,int device,std::size_t slots,std::size_t budget):specs(std::move(s)) {
    if(specs.empty() || specs.size()>8 || !slots || slots>8 || !budget || device<0)
      throw std::invalid_argument("invalid CUDA download pool configuration");
    std::unordered_set<std::string> names;
    for(const auto& spec:specs) {
      std::size_t size=0;
      const auto status=tensor_bytes(spec,size);
      if(!status || spec.memory!=MemoryKind::cuda_device || spec.device_id!=device || !names.insert(spec.name).second)
        throw std::invalid_argument("invalid CUDA download profile");
      const auto padding=(256-bytes%256)%256;
      if(padding>budget-bytes || size>budget-bytes-padding)
        throw std::length_error("download slot budget exceeded");
      bytes+=padding; offsets.push_back(bytes); sizes.push_back(size); bytes+=size;
    }
    for(std::size_t i=0;i<slots;++i) pool.push_back(std::make_shared<PinnedSlot>(bytes));
    executor=std::make_unique<CudaExecutor>(device);
  }
};
CudaDownload::CudaDownload(std::vector<TensorSpec> s,int d,std::size_t slots,std::size_t budget)
    :impl_(std::make_unique<Impl>(std::move(s),d,slots,budget)){}
CudaDownload::~CudaDownload(){close_and_drain();}
void CudaDownload::close_and_drain(){
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->closed=true; impl_->executor->close_and_drain();
}
DownloadedTensors CudaDownload::submit(const std::vector<TensorBinding>& sources){
  const auto status=validate_bindings(impl_->specs,sources);
  if(!status) throw std::invalid_argument(status.message);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if(impl_->closed) throw std::logic_error("CUDA downloader closed");
  const auto available=std::find_if(impl_->pool.begin(),impl_->pool.end(),[](const auto& s){return s.use_count()==1;});
  if(available==impl_->pool.end()) throw std::length_error("CUDA download pool exhausted");
  const auto slot=*available;
  std::vector<TensorBinding> outputs;
  std::vector<BufferLease> leases;
  std::vector<const void*> pointers;
  for(std::size_t i=0;i<impl_->specs.size();++i){
    const auto source=std::find_if(sources.begin(),sources.end(),[&](const auto& s){return s.spec.name==impl_->specs[i].name;});
    leases.push_back(source->storage);
    pointers.push_back(static_cast<const std::uint8_t*>(source->storage.base())+source->storage.offset());
    auto spec=impl_->specs[i]; spec.memory=MemoryKind::pinned_host; spec.device_id=-1;
    outputs.push_back({std::move(spec),BufferLease(slot,slot->data,impl_->bytes,MemoryKind::pinned_host,-1,impl_->offsets[i])});
  }
  leases.emplace_back(slot,slot->data,impl_->bytes,MemoryKind::pinned_host,-1);
  auto ticket=impl_->executor->submit(std::move(leases),
    [slot,pointers=std::move(pointers),sizes=impl_->sizes,offsets=impl_->offsets](cudaStream_t stream){
      for(std::size_t i=0;i<pointers.size();++i){
        const auto error=cudaMemcpyAsync(static_cast<std::uint8_t*>(slot->data)+offsets[i],pointers[i],sizes[i],cudaMemcpyDeviceToHost,stream);
        if(error!=cudaSuccess) return Status{ErrorCode::execution_failed,cudaGetErrorString(error)};
      }
      return Status{};
    });
  return {std::move(outputs),std::move(ticket)};
}
}
