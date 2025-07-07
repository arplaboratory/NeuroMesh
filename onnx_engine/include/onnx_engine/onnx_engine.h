#pragma once

#include "engine_interface/inference_engine_base.hpp"
#include <core/session/onnxruntime_cxx_api.h>
#include <cstring>
#include <cassert>

namespace engine_interface
{

class ONNXEngine : public InferenceEngineBase
{
    public:
        ONNXEngine();
        ~ONNXEngine() override;

        bool loadModel(const std::string& model_path,
                       const std::vector<std::vector<uint32_t>>& input_dims,
                       int type_length) override;

        void runInference(const std::vector<const void *> &inputTensors,
                          const std::vector<int> &inputSizes,
                          std::vector<void *> &outputTensors,
                          const std::vector<int> &outputSizes) override;

    private:
      Ort::Env env_;
      Ort::SessionOptions session_options_;
      std::unique_ptr<Ort::Session> session_;
      std::vector<std::string> input_names_;
      std::vector<std::vector<int64_t>> input_shapes_;
      std::vector<std::string> output_names_;
      std::vector<std::vector<int64_t>> output_shapes_;
      int type_length_;
      // std::vector<void*> bindings;
};

}  // namespace engine_interface
