# `onnx_engine`

The `onnx_engine` package implements `engine_interface::BaseEngine` using
**ONNX Runtime**. It provides a portable, hardware-agnostic inference backend
useful for development and testing on machines without TensorRT.

## Overview

- Loads `.onnx` model files directly (no conversion required)
- Runs on CPU by default; CUDA execution provider available if ORT is built with CUDA support
- Supports the same `TensorRequest` service interface as `tensorrt_engine`

## Usage

Load the ONNX engine plugin by specifying its class name when launching
`EngineInterfaceNode`:

```bash
ros2 run engine_interface engine_interface_node \
    --ros-args \
    -p engine_plugin:=engine_interface::ONNXEngine \
    -p models:=vggt_encoder \
    -p vggt_encoder.path:=/path/to/vggt_image_encoder_2x.onnx \
    -p vggt_encoder.input_dims:="[[1,3,392,518]]" \
    -p vggt_encoder.output_dims:="[[1,1036,1024]]" \
    -p vggt_encoder.tensor_type:=float32
```

## Plugin Registration

Declared in `plugin_description.xml`:

```xml
<library path="onnx_engine">
  <class type="engine_interface::ONNXEngine"
         base_class_type="engine_interface::BaseEngine">
    <description>ONNX Runtime inference engine plugin</description>
  </class>
</library>
```

## Requirements

| Dependency | Notes |
|---|---|
| ONNX Runtime v1.10.0 | Must be built from source and installed to `/usr/local` - see [Build Instructions](../build.md) |
| CUDA | Required only for GPU execution provider |

## Tests

Key test cases from `test/test_onnx_engine.cpp`:

| ID | Description |
|---|---|
| TC-OE-01 | Engine loads a valid ONNX model without error |
| TC-OE-02 | Identity model produces output matching input |
| TC-OE-03 | Calling `runInference` before `loadModel` throws `std::runtime_error` |

```bash
colcon test --packages-select onnx_engine
```
