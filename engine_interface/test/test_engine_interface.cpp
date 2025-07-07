#include <gmock/gmock.h>
#include "engine_interface_test_fixture.hpp"
#include "mock_inference_engine.hpp"

TEST_F(EngineInterfaceTest, LoadModelSuccess) {
    std::vector<std::vector<uint32_t>> dims = {{1, 3, 224, 224}};
    EXPECT_CALL(mock_engine, loadModel("model.onnx", dims, 4))
        .Times(1)
        .WillOnce(Return(true));

    bool result = mock_engine.loadModel("model.onnx", dims, 4);
    EXPECT_TRUE(result);
}

TEST_F(EngineInterfaceTest, LoadModelFailure) {
    std::vector<std::vector<uint32_t>> dims = {{1, 3, 224, 224}};
    EXPECT_CALL(mock_engine, loadModel(_, _, _))
        .Times(1)
        .WillOnce(Return(false));

    bool result = mock_engine.loadModel("bad_model.onnx", dims, 4);
    EXPECT_FALSE(result);
}

TEST_F(EngineInterfaceTest, RunInference) {
    std::vector<const void*> input_tensors(1, nullptr);
    std::vector<int> input_sizes(1, 1024);
    std::vector<void*> output_tensors(1, nullptr);
    std::vector<int> output_sizes(1, 1024);

    EXPECT_CALL(mock_engine, runInference(_, _, _, _))
        .Times(1)
        .WillOnce([](const std::vector<const void*>& inputs, const std::vector<int>& input_sizes,
                     std::vector<void*>& outputs, const std::vector<int>& output_sizes) {
            // Simulate inference by populating output tensors
            for (size_t i = 0; i < outputs.size(); ++i) {
                outputs[i] = new int[output_sizes[i]];  // Allocate memory for output
            }
        });

    mock_engine.runInference(input_tensors, input_sizes, output_tensors, output_sizes);
    EXPECT_EQ(output_tensors.size(), 1);
    EXPECT_NE(output_tensors[0], nullptr);
    // Clean up allocated memory
    delete[] static_cast<int*>(output_tensors[0]);
    output_tensors[0] = nullptr;
}