#pragma once

#include "engine_interface/inference_engine_base.hpp"
#include <cstring>
#include <cassert>
#include <memory>

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
      class Impl;
      std::unique_ptr<Impl> impl_;
};

}  // namespace engine_interface
