#include <neuromesh/onnx/stage.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
using namespace neuromesh;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void cuda_check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
std::vector<float> read(const char* path, std::size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(stream && stream.tellg() == static_cast<std::streamoff>(count * sizeof(float)), "Fixture size mismatch");
  stream.seekg(0);
  std::vector<float> values(count);
  require(static_cast<bool>(stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))), "Fixture read failed");
  return values;
}
device::TensorBinding allocate(device::TensorSpec spec) {
  std::size_t bytes{};
  require(static_cast<bool>(device::tensor_bytes(spec, bytes)), "Invalid test spec");
  void* data{};
  cuda_check(cudaMalloc(&data, bytes));
  std::shared_ptr<void> owner(data, [](void* p) { if (cudaFree(p) != cudaSuccess) std::terminate(); });
  return {std::move(spec), device::BufferLease(owner, data, bytes, device::MemoryKind::cuda_device, 0)};
}
int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: test_onnx_stage graph.onnx normalized_rgb.f32 expected_prefix.f32");
    cuda_check(cudaSetDevice(0));
    device::TensorSpec in{"normalized_rgb", device::DType::float32, {1,3,224,336}, device::MemoryKind::cuda_device, 0};
    device::TensorSpec out{"source_prefix", device::DType::float32, {1,385,768}, device::MemoryKind::cuda_device, 0};
    auto input = allocate(in), output = allocate(out);
    const auto pixels = read(argv[2], 3*224*336), expected = read(argv[3], 385*768);
    cuda_check(cudaMemcpy(input.storage.base(), pixels.data(), pixels.size()*4, cudaMemcpyHostToDevice));
    onnx::Stage stage(argv[1], {in}, {out}, 0, {onnx::ArenaGrowth::requested_size});
    bool rejected = false;
    try { stage.submit({}, {output}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Missing binding accepted");
    auto alias = input;
    alias.storage = output.storage;
    rejected = false;
    try { stage.submit({alias}, {output}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Aliased bindings accepted");
    double maximum = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto ticket = stage.submit({input}, {output}, repeat == 1
          ? onnx::ScratchPolicy::release_unused : onnx::ScratchPolicy::retain);
      auto status = ticket.wait_for(std::chrono::seconds(60));
      require(status.has_value(), "Stage completion timed out");
      if (!*status) throw std::runtime_error(status->message);
      std::vector<float> actual(expected.size());
      cuda_check(cudaMemcpy(actual.data(), output.storage.base(), actual.size()*4, cudaMemcpyDeviceToHost));
      for (std::size_t i=0; i<actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(actual[i])-expected[i]);
        maximum = std::max(maximum, error);
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]) && error <= 1e-5 + 1e-5*std::abs(expected[i]), "Native/Python ORT parity failed");
      }
    }
    stage.close_and_drain();
    rejected = false;
    try { stage.submit({input}, {output}); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Closed stage accepted work");
    auto bad = out;
    bad.name = "wrong_output";
    auto invalid_output = output;
    invalid_output.spec = bad;
    onnx::Stage mismatch(argv[1], {in}, {bad}, 0);
    auto failed = mismatch.submit({input}, {invalid_output}).wait_for(std::chrono::seconds(60));
    require(failed.has_value() && !*failed, "Graph metadata mismatch was accepted");
    auto integer_in = in;
    integer_in.dtype = device::DType::int64;
    auto integer_input = allocate(integer_in);
    cuda_check(cudaMemset(integer_input.storage.base(), 0, pixels.size()*8));
    onnx::Stage wrong_dtype(argv[1], {integer_in}, {out}, 0);
    auto dtype_status = wrong_dtype.submit({integer_input}, {output}).wait_for(std::chrono::seconds(60));
    require(dtype_status.has_value() && !*dtype_status, "Integer declaration accepted for FP32 graph input");
    auto unsupported = in;
    unsupported.dtype = device::DType::bfloat16;
    rejected = false;
    try { onnx::Stage invalid_type(argv[1], {unsupported}, {out}, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Unsupported dtype accepted by stage constructor");
    std::cout << "Native CUDA ONNX conformance passed; max_abs=" << maximum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
