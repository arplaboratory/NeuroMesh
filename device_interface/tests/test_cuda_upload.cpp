#include "neuromesh/device/cuda_upload.hpp"
#include "neuromesh/device/cuda_download.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace neuromesh::device;
void check(bool value) { if (!value) throw std::runtime_error("CUDA upload assertion failed"); }
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
    std::cerr << "CUDA device unavailable: upload execution not validated\n"; return 77;
  }
  std::vector<TensorSpec> specs{
    {"prefix",DType::float32,{1,385,768},MemoryKind::host,-1},
    {"geometry",DType::float32,{6,16,24},MemoryKind::host,-1},
    {"utonia",DType::bfloat16,{1224,16,24},MemoryKind::host,-1}};
  std::vector<TensorBinding> inputs;
  for (const auto& spec : specs) {
    std::size_t size = 0; check(bool(tensor_bytes(spec,size)));
    auto owner = std::shared_ptr<void>(new std::uint8_t[size],[](void* p) { delete[] static_cast<std::uint8_t*>(p); });
    for (std::size_t i = 0; i < size; ++i) static_cast<std::uint8_t*>(owner.get())[i] = i % 251;
    inputs.push_back({spec,BufferLease(owner,owner.get(),size,MemoryKind::host,-1)});
  }
  CudaUpload uploader(specs,0,1,2131968);
  std::reverse(inputs.begin(),inputs.end());
  auto result = uploader.submit(inputs);
  const auto done = result.completion.wait_for(std::chrono::seconds(5)); check(done && bool(*done));
  {
    std::vector<TensorSpec> device_specs;
    for(const auto& t:result.tensors) device_specs.push_back(t.spec);
    auto retained = [&] {
      CudaDownload download(device_specs,0,1,2131968);
      auto reversed=result.tensors; std::reverse(reversed.begin(),reversed.end());
      auto host=download.submit(reversed);
      const auto ready=host.completion.wait_for(std::chrono::seconds(5)); check(ready && bool(*ready));
      bool exhausted=false;
      try {download.submit(reversed);} catch(const std::length_error&){exhausted=true;}
      check(exhausted);
      auto invalid=reversed;invalid.pop_back();
      bool rejected=false;
      try {download.submit(invalid);} catch(const std::invalid_argument&){rejected=true;}
      check(rejected);
      return std::move(host.tensors);
    }();
    for(const auto& host:retained){
      auto input=std::find_if(inputs.begin(),inputs.end(),[&](const auto& t){return t.spec.name==host.spec.name;});
      std::size_t size=0;check(bool(tensor_bytes(host.spec,size)));
      check(std::memcmp(static_cast<const std::uint8_t*>(host.storage.base())+host.storage.offset(),input->storage.base(),size)==0);
    }
    CudaDownload dropped(device_specs,0,1,2131968);
    {auto pending=dropped.submit(result.tensors);}
    dropped.close_and_drain();
    bool closed=false;
    try {dropped.submit(result.tensors);} catch(const std::logic_error&){closed=true;}
    check(closed);
  }
  for (const auto& output : result.tensors) {
    const auto input = std::find_if(inputs.begin(),inputs.end(),[&](const auto& t) { return t.spec.name == output.spec.name; });
    std::size_t size = 0; check(bool(tensor_bytes(output.spec,size)));
    std::vector<std::uint8_t> bytes(size);
    check(cudaMemcpy(bytes.data(),static_cast<std::uint8_t*>(output.storage.base()) + output.storage.offset(),
      size,cudaMemcpyDeviceToHost) == cudaSuccess);
    check(std::memcmp(bytes.data(),input->storage.base(),size) == 0);
  }
  bool exhausted = false;
  try { uploader.submit(inputs); } catch (const std::length_error&) { exhausted = true; }
  check(exhausted);
  auto invalid = inputs; invalid.pop_back();
  bool rejected = false;
  try { uploader.submit(invalid); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected);
  result.tensors.clear(); // return slot; completion ticket alone does not pin it
  { auto dropped = uploader.submit(inputs); } // registry keeps transfer alive
  uploader.close_and_drain();
  bool closed = false;
  try { uploader.submit(inputs); } catch (const std::logic_error&) { closed = true; }
  check(closed);
  auto retained = [&] {
    CudaUpload temporary(specs,0,1,2131968);
    auto upload = temporary.submit(inputs);
    auto done = upload.completion.wait_for(std::chrono::seconds(5)); check(done && bool(*done));
    return std::move(upload.tensors);
  }(); // output leases keep the slot alive after uploader destruction
  std::uint8_t byte = 255;
  check(cudaMemcpy(&byte,static_cast<std::uint8_t*>(retained[0].storage.base()) + retained[0].storage.offset(),
    1,cudaMemcpyDeviceToHost) == cudaSuccess && byte == 0);
  retained.clear();
  std::cout << "CUDA upload byte parity, bounded pool and dropped-ticket drain passed\n";
}
