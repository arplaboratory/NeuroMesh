# `engine_interface`

The `engine_interface` package provides the abstract inference engine base class
and a ROS2 service node that exposes neural model inference over the
`TensorRequest` service.

## `BaseEngine`

Abstract C++ base class for all inference backends. Defined in
`include/engine_interface/inference_engine_base.hpp`.

```cpp
namespace engine_interface {

class BaseEngine {
public:
    virtual ~BaseEngine();

    /// Load a model from disk.
    /// @param model_path  Path to the model file (.engine or .onnx)
    /// @param input_dims  Per-input tensor dimensions
    /// @param type_length Byte width of the tensor element type (e.g. 4 for float32)
    virtual bool loadModel(const std::string& model_path,
                           const std::vector<std::vector<uint32_t>>& input_dims,
                           int type_length) = 0;

    /// Run synchronous inference.
    /// @param inputTensors   Raw pointers to input buffers
    /// @param inputSizes     Size in bytes of each input buffer
    /// @param outputTensors  Raw pointers to pre-allocated output buffers
    /// @param outputSizes    Size in bytes of each output buffer
    virtual void runInference(const std::vector<const void*>& inputTensors,
                              const std::vector<int>& inputSizes,
                              std::vector<void*>& outputTensors,
                              const std::vector<int>& outputSizes) = 0;
};

}  // namespace engine_interface
```

Implementations are registered as `pluginlib` plugins so that
`EngineInterfaceNode` can load them at runtime without a compile-time dependency.

## `EngineInterfaceNode`

A ROS2 composable node that:

1. Loads a `BaseEngine` plugin via `pluginlib::ClassLoader`
2. Reads model configuration from ROS parameters
3. Exposes a `neuromesh_interfaces/srv/TensorRequest` service

### ROS2 Parameters

| Parameter | Type | Description |
|---|---|---|
| `models` | `string` | Comma-separated list of model names to load |
| `<model>.path` | `string` | File path to the model |
| `<model>.input_dims` | `string` | Input dimensions string, e.g. `"[[1,3,392,518]]"` |
| `<model>.output_dims` | `string` | Output dimensions string |
| `<model>.tensor_type` | `string` | Element type (`float32`, `int8`, etc.) |
| `tensor_qos` | `string` | QoS profile for tensor topics |

### Services

| Name | Type | Description |
|---|---|---|
| `tensor_request` | `TensorRequest` | Run inference on a named model |

### Usage

```bash
ros2 run engine_interface engine_interface_node \
    --ros-args \
    -p models:=vggt_encoder \
    -p vggt_encoder.path:=/path/to/model.engine \
    -p vggt_encoder.input_dims:="[[1,3,392,518]]" \
    -p vggt_encoder.output_dims:="[[1,1036,1024]]" \
    -p vggt_encoder.tensor_type:=float32
```

## Writing a New Backend Plugin

1. Inherit from `engine_interface::BaseEngine`
2. Implement `loadModel()` and `runInference()`
3. Register with `pluginlib` in `plugin_description.xml`
4. Add the plugin package as a dependency of `engine_interface`

```cpp
#include "engine_interface/inference_engine_base.hpp"
#include <pluginlib/class_list_macros.hpp>

class MyEngine : public engine_interface::BaseEngine {
public:
    bool loadModel(const std::string& path,
                   const std::vector<std::vector<uint32_t>>& dims,
                   int type_len) override { /* ... */ }

    void runInference(const std::vector<const void*>& in,
                      const std::vector<int>& inSz,
                      std::vector<void*>& out,
                      const std::vector<int>& outSz) override { /* ... */ }
};

PLUGINLIB_EXPORT_CLASS(MyEngine, engine_interface::BaseEngine)
```

## Tests

Unit tests are in `test/test_engine_interface.cpp` using Google Test.

Key test cases:

| ID | Description |
|---|---|
| TC-EI-01 | Node initializes and advertises `TensorRequest` service |
| TC-EI-02 | Valid request returns correct output |
| TC-EI-03 | Request for unknown model name returns failure |
| TC-EI-04 | Engine plugin loads from `pluginlib` correctly |

Run:

```bash
colcon test --packages-select engine_interface
colcon test-result --verbose
```
