#include <gmock/gmock.h>
#include "onnx_engine/onnx_engine.h"

TEST(ONNXEngineTest, LoadModelSuccess) {
    engine_interface::ONNXEngine engine;
    EXPECT_NO_THROW({
    });

    EXPECT_NO_THROW({
        bool loaded = engine.loadModel("test_model.onnx", {{1, 3, 224, 224}}, 4);
        EXPECT_TRUE(loaded);
    });
}

TEST(ONNXEngineTest, InferenceProducesOutput) {
    engine_interface::ONNXEngine engine;
    engine.loadModel("test_model.onnx", {{1, 3, 224, 224}}, 4);

    std::vector<const void*> input_tensors(1, nullptr);
    std::vector<int> input_sizes(1, 1024);
    std::vector<void*> output_tensors(1, nullptr);
    std::vector<int> output_sizes(1, 1024);

    engine.runInference(input_tensors, input_sizes, output_tensors, output_sizes);
    EXPECT_EQ(output_tensors.size(), 1);
    EXPECT_NE(output_tensors[0], nullptr);
    // Clean up allocated memory
    delete[] static_cast<int*>(output_tensors[0]);
    output_tensors[0] = nullptr;
}