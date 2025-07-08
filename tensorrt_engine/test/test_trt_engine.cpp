#include <gmock/gmock.h>
#include "tensorrt_engine/trt_engine_modified.h"
#include "tensorrt_engine/logger.h"

class TRTEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_dims = {{1, 3, 224, 224}};
        type_length = sizeof(float);
        engine_path = "test_engine.trt";
    }

    std::vector<std::vector<uint32_t>> input_dims;
    int type_length;
    std::string engine_path;
};

TEST_F(TRTEngineTest, LoadModelSuccess) {
    engine_interface::TRTEngine engine;
    EXPECT_TRUE(engine.loadModel(engine_path, input_dims, type_length));
}

TEST_F(TRTEngineTest, InferenceProducesOutput) {
    engine_interface::TRTEngine engine;
    ASSERT_TRUE(engine.loadModel(engine_path, input_dims, type_length));

    // Prepare dummy input data
    size_t num_elements = 1 * 3 * 224 * 224;
    std::vector<float> input_data(num_elements, 1.0f);

    std::vector<const void*> input_tensors = { input_data.data() };
    std::vector<int> input_sizes = { static_cast<int>(num_elements * sizeof(float)) };

    // Prepare output buffer
    std::vector<float> output_data(1000);
    std::vector<void*> output_tensors = { output_data.data() };
    std::vector<int> output_sizes = { static_cast<int>(output_data.size() * sizeof(float)) };

    engine.runInference(input_tensors, input_sizes, output_tensors, output_sizes);

    EXPECT_EQ(output_tensors.size(), 1);
    EXPECT_NE(output_tensors[0], nullptr);

    float sum = 0.0f;
    for (float val : output_data) sum += val;
    EXPECT_NE(sum, 0.0f);
}
