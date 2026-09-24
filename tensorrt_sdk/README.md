# NeuroMesh TensorRT SDK targets

This plain CMake package is the single TensorRT/CUDA SDK discovery boundary for
NeuroMesh backends and model-owned plugins. It requires CUDA 12.8 and TensorRT
10.8.0.43, then exports `NeuroMesh::tensorrt_sdk`. Consumers receive the imported
`TensorRT::nvinfer`, `TensorRT::nvonnxparser`, `TensorRT::plugin`, and
`CUDA::cudart` dependencies transitively.

```cmake
find_package(neuromesh_tensorrt_sdk CONFIG REQUIRED)
target_link_libraries(my_backend PRIVATE NeuroMesh::tensorrt_sdk)
```

Set `TensorRT_ROOT` only for an unpacked SDK. System packages remain eligible.
Configure fails if the headers do not report exactly 10.8.0.43 or the CUDA 12.8
toolkit is unavailable. Serialized TensorRT plans are still deployment artifacts:
build them for the target GPU, plugin set, precision profile, and runtime image.
