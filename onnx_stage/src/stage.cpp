#include "neuromesh/onnx/stage.hpp"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <unordered_set>

namespace neuromesh::onnx {
namespace {
using namespace device;
ONNXTensorElementDataType ort_dtype(DType dtype) {
  switch(dtype) {
    case DType::float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default: throw std::invalid_argument("Stage supports FP32 and INT64 tensors only");
  }
}
void check_specs(const std::vector<TensorSpec>& specs, int device_id) {
  if (specs.empty()) throw std::invalid_argument("Stage signature cannot be empty");
  std::unordered_set<std::string> names;
  for (const auto& spec : specs) {
    std::size_t bytes{};
    auto status = tensor_bytes(spec, bytes);
    (void)ort_dtype(spec.dtype);
    if (!status || spec.memory != MemoryKind::cuda_device ||
        spec.device_id != device_id || !names.insert(spec.name).second)
      throw std::invalid_argument("Stage requires unique fixed FP32 or INT64 CUDA bindings");
  }
}
void check_graph(Ort::Session& session, const std::vector<TensorSpec>& expected, bool input,
                 const StageOptions& options) {
  const auto count = input ? session.GetInputCount() : session.GetOutputCount();
  if (count != expected.size()) throw std::invalid_argument("Graph binding count mismatch");
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t i = 0; i < count; ++i) {
    auto name = input ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
    auto type = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    auto info = type.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    std::vector<const char*> symbols(shape.size());
    info.GetSymbolicDimensions(symbols.data(), symbols.size());
    for (std::size_t axis=0;axis<shape.size();++axis)
      if (shape[axis]<0 && symbols[axis] && *symbols[axis])
        for (const auto& [symbol, extent] : options.dimension_overrides)
          if (symbol==symbols[axis]) shape[axis]=extent;
    bool found = false;
    for (const auto& spec : expected) {
      if (spec.name == name.get()) {
        found = info.GetElementType() == ort_dtype(spec.dtype) && shape == spec.shape;
        break;
      }
    }
    if (!found) {
      std::string shape;
      const auto dimensions=info.GetShape();
      for (std::size_t axis=0;axis<dimensions.size();++axis)
        shape += std::to_string(dimensions[axis])+":"+(symbols[axis] ? symbols[axis] : "")+",";
      throw std::invalid_argument("Graph binding name, shape or dtype mismatch: " +
          std::string(name.get()) + " [" + shape + "]");
    }
  }
}

}
struct Stage::State {
  std::string graph;
  std::vector<device::TensorSpec> inputs, outputs;
  int device_id;
  StageOptions options;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "neuromesh_stage"};
  std::unique_ptr<Ort::Session> session;
  void initialize(cudaStream_t stream) {
    if (session) return;
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    for (const auto& [name, extent] : this->options.dimension_overrides)
      Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, name.c_str(), extent));
    for (const auto& library : this->options.custom_operator_libraries)
      options.RegisterCustomOpsLibrary(library.c_str());
    OrtCUDAProviderOptionsV2* raw{};
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
    auto release = [](OrtCUDAProviderOptionsV2* p) { Ort::GetApi().ReleaseCUDAProviderOptions(p); };
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> provider(raw, release);
    const auto id = std::to_string(device_id);
    const char* growth = this->options.arena_growth == ArenaGrowth::requested_size
        ? "kSameAsRequested" : "kNextPowerOfTwo";
    const char* keys[]{"device_id", "use_tf32", "do_copy_in_default_stream", "arena_extend_strategy"};
    const char* values[]{id.c_str(), "0", "1", growth};
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(raw, keys, values, 4));
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(raw, "user_compute_stream", stream));
    options.AppendExecutionProvider_CUDA_V2(*raw);
    auto candidate = std::make_unique<Ort::Session>(env, graph.c_str(), options);
    check_graph(*candidate, inputs, true, this->options);
    check_graph(*candidate, outputs, false, this->options);
    session = std::move(candidate);
  }
};
Stage::Stage(std::string graph, std::vector<device::TensorSpec> inputs,
             std::vector<device::TensorSpec> outputs, int device_id, StageOptions options) {
  std::unordered_set<std::string> dimensions;
  for (const auto& [name, extent] : options.dimension_overrides)
    if (name.empty() || name.size()>128 || name.find('\0')!=std::string::npos ||
        extent<=0 || !dimensions.insert(name).second)
      throw std::invalid_argument("Invalid or duplicate symbolic dimension specialization");
  std::unordered_set<std::string> libraries;
  for (const auto& library : options.custom_operator_libraries)
    if (library.empty() || library.size()>4096 || library.find('\0')!=std::string::npos ||
        !libraries.insert(library).second)
      throw std::invalid_argument("Invalid or duplicate custom operator library");
  check_specs(inputs, device_id);
  check_specs(outputs, device_id);
  if (graph.empty()) throw std::invalid_argument("Graph path is empty");
  executor_ = std::make_unique<device::CudaExecutor>(device_id);
  state_ = std::make_shared<State>();
  state_->graph = std::move(graph);
  state_->inputs = std::move(inputs);
  state_->outputs = std::move(outputs);
  state_->device_id = device_id;
  state_->options = options;
}
Stage::~Stage() {
  close_and_drain();
  // Destroy ORT before the borrowed executor stream is destroyed.
  state_.reset();
}
void Stage::close_and_drain() { executor_->close_and_drain(); }
device::CompletionTicket Stage::submit(std::vector<device::TensorBinding> inputs,
                                       std::vector<device::TensorBinding> outputs, ScratchPolicy scratch) {
  auto status = device::validate_bindings(state_->inputs, inputs);
  if (!status) throw std::invalid_argument(status.message);
  status = device::validate_bindings(state_->outputs, outputs);
  if (!status) throw std::invalid_argument(status.message);
  auto all = inputs;
  all.insert(all.end(), outputs.begin(), outputs.end());
  status = validate_disjoint_bindings(all);
  if (!status) throw std::invalid_argument(status.message);
  std::vector<device::BufferLease> leases;
  for (const auto& binding : all) leases.push_back(binding.storage);
  return executor_->submit(std::move(leases),
      [state = state_, inputs = std::move(inputs), outputs = std::move(outputs), scratch](cudaStream_t stream) {
    state->initialize(stream);
    Ort::IoBinding binding(*state->session);
    Ort::MemoryInfo memory("Cuda", OrtDeviceAllocator, state->device_id, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    values.reserve(inputs.size() + outputs.size());
    auto bind = [&](const device::TensorBinding& tensor, bool input) {
      std::size_t bytes{};
      device::tensor_bytes(tensor.spec, bytes);
      auto* address = static_cast<unsigned char*>(tensor.storage.base()) + tensor.storage.offset();
      values.push_back(Ort::Value::CreateTensor(memory, address, bytes, tensor.spec.shape.data(),
                                               tensor.spec.shape.size(), ort_dtype(tensor.spec.dtype)));
      if (input) binding.BindInput(tensor.spec.name.c_str(), values.back());
      else binding.BindOutput(tensor.spec.name.c_str(), values.back());
    };
    for (const auto& tensor : inputs) bind(tensor, true);
    for (const auto& tensor : outputs) bind(tensor, false);
    // Keep ORT's default Run synchronization; no unsafe async-run override.
    Ort::RunOptions run;
    if (scratch == ScratchPolicy::release_unused) {
      const auto arenas = "gpu:" + std::to_string(state->device_id);
      run.AddConfigEntry("memory.enable_memory_arena_shrinkage", arenas.c_str());
    }
    state->session->Run(run, binding);
    return device::Status{};
  });
}
}
