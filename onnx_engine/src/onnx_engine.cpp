#include "onnx_engine/onnx_engine.h"

namespace engine_interface
{
ONNXEngine::ONNXEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_engine"),
      session_options_(),
      type_length_(0){}

ONNXEngine::~ONNXEngine() = default;

bool ONNXEngine::loadModel(const std::string& model_path,
                           const std::vector<std::vector<uint32_t>>& input_dims,
                           int type_length)
{
    type_length_ = type_length;
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    input_names_.clear();
    input_shapes_.clear();
    output_names_.clear();
    output_shapes_.clear();

    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = session_->GetInputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get input name for index " + std::to_string(i));
        }
        input_names_.emplace_back(name);
        allocator.Free(name);
        // input_names_.push_back(session_->GetInputName(i, allocator));

        auto type_info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_shapes_.push_back(type_info.GetShape());
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = session_->GetOutputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get output name for index " + std::to_string(i));
        }
        output_names_.emplace_back(name);
        allocator.Free(name);
        // output_names_.push_back(session_->GetOutputName(i, allocator));

        auto type_info = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        output_shapes_.push_back(type_info.GetShape());
    }

    return true;
}

void ONNXEngine::runInference(const std::vector<const void *> &inputTensors,
                                const std::vector<int> &inputSizes,
                                std::vector<void *> &outputTensors,
                                const std::vector<int> &outputSizes)
{
    if (!session_) {
      throw std::runtime_error("Session is not initialized. Call loadModel first.");
    }
    if (inputTensors.size() != input_names_.size()) {
      throw std::runtime_error(
          "Number of input tensors doesn't match number of input names");
    }
    if (outputTensors.size() != output_names_.size()) {
      throw std::runtime_error(
          "Number of output tensors doesn't match number of output names");
    }

    // if (inputTensors.size() + outputTensors.size() != bindings.size()) {
    //   throw std::runtime_error(
    //       "Number of input and output tensors doesn't match engine bindings");
    // }

    std::vector<Ort::Value> ort_inputs;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < inputTensors.size(); ++i) {
        const float* input_data = reinterpret_cast<const float*>(inputTensors[i]);
        const std::vector<int64_t>& input_shape = input_shapes_[i];
        size_t input_numel = 1;
        for (auto d : input_shape) input_numel *= d;

        ort_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(
                mem_info, const_cast<float*>(input_data), 
                input_numel, 
                input_shape.data(), 
                input_shape.size()
            )
        );
    }

    std::vector<const char*> input_names_c;
    std::vector<const char*> output_names_c;
    for (const auto& name : input_names_) input_names_c.push_back(name.c_str());
    for (const auto& name : output_names_) output_names_c.push_back(name.c_str());

    // run the inference
    auto ort_outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
        output_names_c.data(), output_names_c.size()
    );

    for (size_t i = 0; i < ort_outputs.size(); ++i)
    {
        float* output_data = ort_outputs[i].GetTensorMutableData<float>();
        size_t output_numel = 1;
        for (auto d : output_shapes_[i]) output_numel *= d;
        std::memcpy(outputTensors[i], output_data, output_numel * type_length_);
    }

}

} // namespace engine_interface

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(engine_interface::ONNXEngine, engine_interface::InferenceEngineBase)
