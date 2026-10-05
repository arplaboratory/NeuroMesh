#include <gmock/gmock.h>
#include "onnx_engine/onnx_engine.h"
#include <fstream>
#include <unistd.h>

// Real ONNX IR8/opset18 Identity model: float32 [1,3] x -> y.
// Inline protobuf keeps this CPU compatibility test independent of Python ONNX.
static const unsigned char kIdentityModel[] = {8, 8, 18, 14, 110, 101, 117, 114, 111, 109, 101, 115, 104, 95, 116, 101, 115, 116, 58, 70, 10, 16, 10, 1, 120, 18, 1, 121, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121, 18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 19, 10, 1, 120, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 1, 10, 2, 8, 3, 98, 19, 10, 1, 121, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 1, 10, 2, 8, 3, 66, 2, 16, 18};

TEST(ONNXEngineTest, PinnedRuntimeLoadsAndExecutesModel) {
    char path[] = "/tmp/neuromesh_onnx_XXXXXX";
    const int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(kIdentityModel), sizeof(kIdentityModel));
    }
    engine_interface::ONNXEngine engine;
    const bool loaded = engine.loadModel(path, {{1, 3}}, sizeof(float));
    unlink(path);
    ASSERT_TRUE(loaded);
    float input[] = {1.25f, -2.5f, 3.75f};
    float output[] = {0, 0, 0};
    std::vector<const void*> inputs{input};
    std::vector<void*> outputs{output};
    engine.runInference(inputs, {sizeof(input)}, outputs, {sizeof(output)});
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(input[i], output[i]);
}
