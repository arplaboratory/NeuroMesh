#include "onnx_engine/onnx_engine.h"
#include <core/session/onnxruntime_cxx_api.h>

namespace engine_interface
{

class ONNXEngine::Impl
{
public:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;
    int type_length_;

    Impl()
        : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_engine"),
          session_options_(),
          type_length_(0) {}
};

ONNXEngine::ONNXEngine()
    : impl_(std::make_unique<Impl>())
{}

ONNXEngine::~ONNXEngine() = default;

bool ONNXEngine::loadModel(const std::string& model_path,
                           const std::vector<std::vector<uint32_t>>& input_dims,
                           int type_length)
{
    impl_->type_length_ = type_length;
    impl_->session_ = std::make_unique<Ort::Session>(impl_->env_, 
                model_path.c_str(), 
                impl_->session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    impl_->input_names_.clear();
    impl_->input_shapes_.clear();
    impl_->output_names_.clear();
    impl_->output_shapes_.clear();

    size_t num_inputs = impl_->session_->GetInputCount();
    size_t num_outputs = impl_->session_->GetOutputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = impl_->session_->GetInputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get input name for index " + std::to_string(i));
        }
        impl_->input_names_.emplace_back(name);
        allocator.Free(name);

        auto type_info = impl_->session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();

        if (i < input_dims.size()) {
            std::vector<int64_t> new_shape(input_dims[i].begin(), input_dims[i].end());
            impl_->input_shapes_.push_back(new_shape);
        } else {
            impl_->input_shapes_.push_back(shape);
        }
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = impl_->session_->GetOutputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get output name for index " + std::to_string(i));
        }
        impl_->output_names_.emplace_back(name);
        allocator.Free(name);

        auto type_info = impl_->session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        impl_->output_shapes_.push_back(type_info.GetShape());
    }

    return true;
}

void ONNXEngine::runInference(const std::vector<const void *> &inputTensors,
                                const std::vector<int> &inputSizes,
                                std::vector<void *> &outputTensors,
                                const std::vector<int> &outputSizes)
{
    if (!impl_->session_) {
      throw std::runtime_error("Session is not initialized. Call loadModel first.");
    }
    if (inputTensors.size() != impl_->input_names_.size()) {
      throw std::runtime_error(
          "Number of input tensors doesn't match number of input names");
    }
    if (outputTensors.size() != impl_->output_names_.size()) {
      throw std::runtime_error(
          "Number of output tensors doesn't match number of output names");
    }

    // validate input/output buffer sizes
    for (size_t i = 0; i < inputSizes.size(); ++i) {
        size_t expected_size = 1;
        for (auto d : impl_->input_shapes_[i]) expected_size *= d;
        expected_size *= impl_->type_length_;
        if (inputSizes[i] < static_cast<int>(expected_size)) {
            throw std::runtime_error("Input buffer size is smaller than expected for input " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < outputSizes.size(); ++i) {
        size_t expected_size = 1;
        for (auto d : impl_->output_shapes_[i]) expected_size *= d;
        expected_size *= impl_->type_length_;
        if (outputSizes[i] < static_cast<int>(expected_size)) {
            throw std::runtime_error("Output buffer size is smaller than expected for output " + std::to_string(i));
        }
    } 

    std::vector<Ort::Value> ort_inputs;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < inputTensors.size(); ++i) {
        const float* input_data = reinterpret_cast<const float*>(inputTensors[i]);
        const std::vector<int64_t>& input_shape = impl_->input_shapes_[i];
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
    for (const auto& name : impl_->input_names_) input_names_c.push_back(name.c_str());
    for (const auto& name : impl_->output_names_) output_names_c.push_back(name.c_str());

    // run the inference
    auto ort_outputs = impl_->session_->Run(
        Ort::RunOptions{nullptr},
        input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
        output_names_c.data(), output_names_c.size()
    );

    for (size_t i = 0; i < ort_outputs.size(); ++i)
    {
        float* output_data = ort_outputs[i].GetTensorMutableData<float>();
        size_t output_numel = 1;
        for (auto d : impl_->output_shapes_[i]) output_numel *= d;
        std::memcpy(outputTensors[i], output_data, output_numel * impl_->type_length_);
    }

}

} // namespace engine_interface

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(engine_interface::ONNXEngine, engine_interface::InferenceEngineBase)
