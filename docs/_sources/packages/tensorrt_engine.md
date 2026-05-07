# `tensorrt_engine`

The `tensorrt_engine` package implements `engine_interface::BaseEngine` using
the **NVIDIA TensorRT** runtime. It is loaded as a `pluginlib` plugin by
`EngineInterfaceNode`.

## Overview

TensorRT provides optimized inference for NVIDIA GPUs. The plugin:

- Loads serialized `.engine` files (produced by `trtexec` or the TensorRT API)
- Manages CUDA device memory buffers and execution contexts
- Supports FP16 precision for faster inference

## Model Files

Place converted `.engine` files in `tensorrt_engine/models/` following the
directory layout:

```
tensorrt_engine/models/
├── gat/
│   ├── encoder.trt
│   ├── gat_layer1.trt
│   └── gat_layer2.trt
└── vggt/
    ├── vggt_aggregator_2x.engine
    └── vggt_image_encoder_2x.engine
```

## Converting ONNX → TensorRT

Run `trtexec` on the target robot (engines are GPU-specific):

```bash
# VGGT image encoder
trtexec \
    --onnx=vggt_image_encoder_2x.onnx \
    --saveEngine=tensorrt_engine/models/vggt/vggt_image_encoder_2x.engine \
    --fp16

# VGGT aggregator (decoder)
trtexec \
    --onnx=vggt_aggregator_2x.onnx \
    --saveEngine=tensorrt_engine/models/vggt/vggt_aggregator_2x.engine \
    --fp16
```

:::{note}
TensorRT engine files are **hardware-specific**. An engine built on one GPU
model will not run on a different GPU architecture. Always convert on the
deployment hardware.
:::

## Plugin Registration

The plugin is declared in `plugin_description.xml`:

```xml
<library path="tensorrt_engine">
  <class type="engine_interface::TRTEngine"
         base_class_type="engine_interface::BaseEngine">
    <description>TensorRT inference engine plugin</description>
  </class>
</library>
```

## Requirements

| Dependency | Version |
|---|---|
| TensorRT | 8.6.1+ |
| CUDA | 12.1+ |
| cuDNN | compatible with TensorRT version |

## Tests

```bash
colcon test --packages-select tensorrt_engine
```

Tests in `test/test_trt_engine.cpp` verify that the engine loads a sample model
and produces correct outputs.
