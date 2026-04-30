#include "onnx_engine/onnx_engine.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <iostream>

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
          type_length_(0) 
          {
            session_options_.SetInterOpNumThreads(1);
            session_options_.SetIntraOpNumThreads(1);
            session_options_.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

            // TODO: If using CUDA, set up CUDA options
            // if (_UseCuda) {
            //     OrtCUDAProviderOptions cuda_options;
            //     cuda_options.device_id = 0;
            //     cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            //     cuda_options.arena_extend_strategy = 0;
            //     cuda_options.do_copy_in_default_stream = 0;
            //     session_options_.AppendExecutionProvider_CUDA(cuda_options);
            // }
          }
};

ONNXEngine::ONNXEngine()
    : impl_(std::make_unique<Impl>())
{}

ONNXEngine::~ONNXEngine() = default;

bool ONNXEngine::loadModel(const std::string& model_path,
                           const std::vector<std::vector<uint32_t>>& input_dims,
                           int type_length)
{
    // std::cout << "Loading ONNX model from: " << model_path << std::endl;
    impl_->type_length_ = type_length;
    // std::cout << "Type length: " << impl_->type_length_ << std::endl;
    try
    {
        impl_->session_ = std::make_unique<Ort::Session>(impl_->env_, 
                    model_path.c_str(), 
                    impl_->session_options_);
        // std::cout << "Session created successfully." << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Error creating ONNX session: " << e.what() << ". Code: " << e.GetOrtErrorCode() << std::endl;
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    ONNXTensorElementDataType type;
    Ort::TypeInfo* type_info;

    impl_->input_names_.clear();
    impl_->input_shapes_.clear();
    impl_->output_names_.clear();
    impl_->output_shapes_.clear();

    size_t num_inputs = impl_->session_->GetInputCount();
    size_t num_outputs = impl_->session_->GetOutputCount();

    // std::cout << "Number of inputs: " << num_inputs << std::endl;
    // std::cout << "Number of outputs: " << num_outputs << std::endl;

    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = impl_->session_->GetInputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get input name for index " + std::to_string(i));
        }
        impl_->input_names_.emplace_back(name);
        allocator.Free(name);

        type_info = new Ort::TypeInfo(impl_->session_->GetInputTypeInfo(i));
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        type = tensor_info.GetElementType();
        impl_->input_shapes_.push_back(tensor_info.GetShape());

        std::cout << "Input " << i << ": name = " << impl_->input_names_.back() 
                  << ", shape = [";
        for (const auto& dim : impl_->input_shapes_.back()) {
            std::cout << dim << " ";
        }
        std::cout << "], type = " << type << std::endl;

        delete(type_info);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = impl_->session_->GetOutputName(i, allocator);
        if (name == nullptr) {
            throw std::runtime_error("Failed to get output name for index " + std::to_string(i));
        }
        impl_->output_names_.emplace_back(name);
        allocator.Free(name);

        type_info = new Ort::TypeInfo(impl_->session_->GetOutputTypeInfo(i));
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        type = tensor_info.GetElementType();
        impl_->input_shapes_.push_back(tensor_info.GetShape());
        
        std::cout << "Output " << i << ": name = " << impl_->output_names_.back() 
                  << ", shape = [";
        for (const auto& dim : impl_->input_shapes_.back()) {
            std::cout << dim << " ";
        }
        std::cout << "], type = " << type << std::endl;

        delete(type_info);
    }

    return true;
}

void ONNXEngine::runInference(const std::vector<const void *> &inputTensors,
                                const std::vector<int> &inputSizes,
                                std::vector<void *> &outputTensors,
                                const std::vector<int> &outputSizes)
{
    // std::cout << "Running inference..." << std::endl;
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

    std::vector<Ort::Value> ort_inputs, ort_outputs;
    Ort::MemoryInfo mem_info{ nullptr };
    try
    {
        mem_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception &e)
    {
        std::cerr << "ONNX exception caught: " << e.what() << ". Code: " << e.GetOrtErrorCode() << std::endl;
    }

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

    // std::cout << "Input tensors converted to Ort::Value." << std::endl;

    std::vector<const char*> input_names_c;
    std::vector<const char*> output_names_c;
    for (const auto& name : impl_->input_names_) input_names_c.push_back(name.c_str());
    for (const auto& name : impl_->output_names_) output_names_c.push_back(name.c_str());

    // run the inference
    // std::cout << "Running ONNX session..." << std::endl;
    try
    {
        ort_outputs = impl_->session_->Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
            output_names_c.data(), output_names_c.size()
        );
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX exception caught during inference: " << e.what() << ". Code: " << e.GetOrtErrorCode() << std::endl;
        throw;
    }

    for (size_t i = 0; i < ort_outputs.size(); ++i)
    {
        auto& ort_val = ort_outputs[i];
        auto tensor_info = ort_val.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        size_t numel = 1;
        for (auto d : shape) {
            if (d <= 0) throw std::runtime_error("Invalid output shape");
            numel *= d;
        }
        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Output tensor type mismatch");
        if (outputSizes[i] < static_cast<int>(numel * sizeof(float)))
            throw std::runtime_error("Output buffer too small");
        float* output_data = ort_val.GetTensorMutableData<float>();
        std::memcpy(outputTensors[i], output_data, numel * sizeof(float));
    }
}

} // namespace engine_interface

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(engine_interface::ONNXEngine, engine_interface::BaseEngine)
