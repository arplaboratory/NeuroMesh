#pragma once

#include <vector>
#include <string>

namespace engine_interface
{

class InferenceEngineBase
{
public:
    virtual ~InferenceEngineBase();

    // Load model (path, input/output dims, etc.)
    virtual bool loadModel(const std::string& model_path,
                        const std::vector<std::vector<uint32_t>>& input_dims,
                        int type_length) = 0;

    virtual void runInference(const std::vector<const void *> &inputTensors,
                                 const std::vector<int> &inputSizes,
                                 std::vector<void *> &outputTensors,
                                 const std::vector<int> &outputSizes) = 0;
};

}  // namespace engine_interface
