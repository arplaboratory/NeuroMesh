# Build Instructions

## Dependencies

Install ROS2 Jazzy and CUDA 12.1+ before building.

## ONNX Runtime (required before `colcon build`)

The `onnx_engine` package requires ONNX Runtime to be built from source and
installed system-wide **before** building the ROS2 workspace. The version used
is **v1.10.0**, with a patch to fix GCC 13 build errors.

### Why the patch?

`onnxruntime_gcc13.patch` (found in `onnx_engine/`) adds suppression flags for
several GCC 13 warnings that are treated as errors in the default build:

```diff
 else()
   add_definitions(-DPLATFORM_POSIX)
+  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
+    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=stringop-overflow -Wno-error=deprecated-declarations -Wno-error=dangling-reference -Wno-error=maybe-uninitialized")
+    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wno-error=stringop-overflow -Wno-error=maybe-uninitialized")
+  endif()
   check_cxx_compiler_flag(-Wunused-but-set-parameter HAS_UNUSED_BUT_SET_PARAMETER)
```

### Installation Steps

```bash
# 1. Clone ONNX Runtime v1.10.0
git clone --recursive https://github.com/microsoft/onnxruntime.git -b v1.10.0

# 2. Prevent colcon from treating it as a ROS package
cd onnxruntime && touch COLCON_IGNORE

# 3. Apply the GCC 13 compatibility patch
git apply /path/to/neuromesh/onnx_engine/onnxruntime_gcc13.patch

# 4. Build
./build.sh --config Release --build_shared_lib --parallel

# 5. Install to /usr/local
cd build/Linux/Release
cmake --install . --prefix /usr/local
```

:::{note}
Replace `/path/to/neuromesh/` with the actual path to the `neuromesh` package,
e.g. `~/workspace/src/neuromesh/`.
:::

After installation, the ONNX Runtime shared library will be available at
`/usr/local/lib/libonnxruntime.so` and headers at `/usr/local/include/onnxruntime/`.

### TensorRT Engine Conversion

ONNX model files must be converted to TensorRT `.engine` files on each robot
before running inference. TensorRT engines are hardware-specific and cannot be
transferred between different GPU architectures.

1. Obtain the ONNX model files (encoder + decoder).
2. Convert on each robot:

   ```bash
   # Example: convert VGGT encoder
   trtexec --onnx=vggt_image_encoder_2x.onnx \
           --saveEngine=tensorrt_engine/models/vggt/vggt_image_encoder_2x.engine \
           --fp16
   ```

3. Place the converted `.engine` files in `tensorrt_engine/models/`.

See `tensorrt_engine/models/README.md` for placement conventions.

## Colcon Build

```bash
cd /path/to/workspace
colcon build --packages-up-to neuromesh_platform_r2
source install/setup.bash
```

## Troubleshooting

### TensorRT library not found

Ensure `LD_LIBRARY_PATH` includes the TensorRT library directory:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### Engine file not found at runtime

Check that `.engine` files are present in the paths specified in the launch
file parameters (`encoder_model_path`, `decoder_model_path`).
