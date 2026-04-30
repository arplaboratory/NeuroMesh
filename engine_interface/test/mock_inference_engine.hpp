#ifndef MOCK_INFERENCE_ENGINE_HPP
#define MOCK_INFERENCE_ENGINE_HPP

#include <gmock/gmock.h>
#include "engine_interface/inference_engine_base.hpp"

using ::testing::Return;
using ::testing::_;

namespace engine_interface {

// Mock class for BaseEngine
class MockInferenceEngine : public BaseEngine {
public:
    MOCK_METHOD(bool, loadModel, 
                (const std::string& model_path, 
                const std::vector<std::vector<uint32_t>>& input_dims, 
                int type_length), 
                (override));
    MOCK_METHOD(void, runInference, 
                (const std::vector<const void*>& inputTensors, 
                const std::vector<int>& inputSizes, 
                std::vector<void*>& outputTensors, 
                const std::vector<int>& outputSizes), 
                (override));
};

} // namespace engine_interface

#endif // MOCK_INFERENCE_ENGINE_HPP