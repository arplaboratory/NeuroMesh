#include <neuromesh/onnx/stage.hpp>
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace neuromesh;
void require(bool ok, const char* reason) { if(!ok) throw std::runtime_error(reason); }
device::TensorBinding allocate(const char* name) {
  void* address=nullptr;
  require(cudaMalloc(&address,6*sizeof(float))==cudaSuccess,"allocation failed");
  std::shared_ptr<void> owner(address,[](void* p){cudaFree(p);});
  return {{name,device::DType::float32,{3,2},device::MemoryKind::cuda_device,0},
          device::BufferLease(owner,address,6*sizeof(float),device::MemoryKind::cuda_device,0)};
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected symbolic identity graph");
    auto input=allocate("x"), output=allocate("y");
    const float values[]{1,2,3,4,5,6};
    require(cudaMemcpy(input.storage.base(),values,sizeof(values),cudaMemcpyHostToDevice)==cudaSuccess,"upload failed");
    auto execute=[&](onnx::StageOptions options) {
      onnx::Stage stage(argv[1],{input.spec},{output.spec},0,std::move(options));
      auto completed=stage.submit({input},{output}).wait_for(std::chrono::seconds(30));
      if(!completed){stage.close_and_drain();throw std::runtime_error("execution timeout");}
      return bool(*completed);
    };
    require(!execute({}),"unspecialized graph was accepted");
    require(execute({onnx::ArenaGrowth::requested_size,{{"rows",3}}}),"specialized graph failed");
    float actual[6]{};
    require(cudaMemcpy(actual,output.storage.base(),sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess,"download failed");
    for(int i=0;i<6;++i) require(actual[i]==values[i],"identity output changed");
    require(!execute({onnx::ArenaGrowth::requested_size,{{"rows",2}}}),"incompatible extent accepted");
    for(const auto& overrides : {
        std::vector<std::pair<std::string,std::int64_t>>{{"rows",0}},
        std::vector<std::pair<std::string,std::int64_t>>{{"rows",3},{"rows",3}}}) {
      bool rejected=false;
      try {onnx::Stage invalid(argv[1],{input.spec},{output.spec},0,{onnx::ArenaGrowth::requested_size,overrides});}
      catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"invalid dimension override accepted");
    }
    std::cout<<"Symbolic graph specialization preserves exact binding bounds\n";
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
