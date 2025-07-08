#pragma once

#include "engine_interface/inference_engine_base.hpp"
#include <vector>
#include <string>
#include <memory>

#include <tensorrt_engine/logger.h>

namespace engine_interface {

class TRTEngine : public InferenceEngineBase 
{
    public:
        TRTEngine();
        ~TRTEngine() override;

        bool loadModel(const std::string& model_path, 
            const std::vector<std::vector<uint32_t>>& input_dims, 
            int type_length) override;

        void runInference(const std::vector<const void*>& inputTensors, 
                          const std::vector<int>& inputSizes,
                          std::vector<void*>& outputTensors, 
                          const std::vector<int>& outputSizes) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
};

} // namespace engine_interface