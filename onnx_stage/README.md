# Experimental named CUDA ONNX stage

`NeuroMesh::onnx_stage` executes a fixed-shape ONNX graph with FP32 data and INT64 indices using device-resident
named inputs and caller-allocated outputs. It is a standalone C++17 CMake library,
not a ROS component or a replacement for the existing VGGT engine/plugin.

`Stage` owns one `CudaExecutor`. First submission creates and validates the ORT
session on its worker, borrowing its nonblocking CUDA stream. CUDA options disable
TF32 and CPU node fallback. ORT's default Run synchronization is retained; the
executor additionally records its completion event before publishing status.
Subsequent submissions reuse the session. Admission is bounded to one outstanding
request per stage; busy requests are rejected rather than queued indefinitely.

The constructor validates fixed FP32/INT64 CUDA signatures. Submit validates names,
shape, dtype, device metadata, capacity and offsets, and rejects overlapping
bindings before launch. Graph metadata is checked before execution. Integer bindings remain INT64
through I/O binding; no float conversion or host staging is inserted. Unsupported
dtypes and integer declarations for floating graph inputs fail closed. ORT/provider
exceptions become failed completion statuses. Input and output leases are retained
independently of the caller's ticket. Close drains before session destruction;
the session is destroyed before its borrowed CUDA stream.

For pipelines with several large sessions, `StageOptions` can select
`ArenaGrowth::requested_size` instead of the default power-of-two growth.
A submission can request `ScratchPolicy::release_unused`; ORT then shrinks
unused CUDA arena regions after that run. Session weights and live bindings stay
resident. This is allocator reclamation, not cancellation or a hard GPU memory
budget. It can increase allocation latency on the next request, so GLD requests
it at the final denoiser evaluation and after decoding, not on every Euler step.
Both choices are explicit; existing callers retain their original defaults.

```cpp
neuromesh::onnx::Stage stage(graph, inputs, outputs, device_id,
    {neuromesh::onnx::ArenaGrowth::requested_size});
auto ticket = stage.submit(input_bindings, output_bindings,
    neuromesh::onnx::ScratchPolicy::release_unused);
// Keep leases and wait for completion before reusing buffers.
```

The stage test executes retain/release/retain on the same session and verifies
numerical outputs after reuse. Rebuild dependent C++ libraries after updating
this API. Install `device_interface` and `onnx_stage` in the same development
overlay to prevent an older shared include root from shadowing the new headers.

Named symbolic dimensions can be specialized through
`StageOptions::dimension_overrides` before session creation, for example
`{{"level0_rows", 37387}, {"original_rows", 37387}}`. Names must be unique and
nonempty, with positive extents. The binding contract stays exact and immutable:
unknown symbols and shape mismatches fail closed. A different cloud hierarchy
requires a new stage; a submission never resizes a cached buffer. Some ORT output
metadata retains symbols after specialization, so validation resolves only
explicitly supplied names before comparing the exact shape.

The GLD full Utonia encoder uses one stage per original encoder level and a
sixth reconstruction stage. It releases unused scratch between levels and keeps
intermediate features on CUDA. This fits the tested Warty sweep; it is not a
hard allocator budget and still exceeds the development GPU's capacity for the
larger Wilbur sweep. The current GLD runtime and stage must be rebuilt together;
the previously built NeuroMesh SIF predates this symbolic-specialization change.

Caller obligations:

- Verify graph/bundle hashes and keep the graph immutable through lazy loading.
- Supply truthful allocation metadata and owners. Hardware pointer ownership is
  not inspected by this adapter.
- Finish producer writes before submission. Wait for the completion ticket before
  reading/reusing outputs. Arbitrary cross-stream producer dependencies are not
  implemented.
- Keep object lifetime externally synchronized; never destroy/close from executor
  callbacks or allocation deleters. The underlying executor's device-loss
  fail-stop policy remains experimental.

No normalization, RGB/LiDAR preprocessing, sampler, decoder, wire transport,
authorization, live sensor synchronization or ROS executor runs here. No hidden
host staging is performed by this API. ORT may allocate internal workspaces;
allocator budgets, latency and memory peaks have not been qualified.

## NeuroMesh deployment image

`just apptainer neuromesh hpc-x86-jazzy` installs ONNX Runtime GPU 1.23.2,
cuDNN 9.10.2.21, matching verified C++ headers, this CMake stage and
`GLD::onnx_prefix` under `/opt/neuromesh/install`. The wheel's native runtime
and CUDA provider have loader-visible links in that prefix. Image tests verify
CUDA provider availability and shared-library dependencies. The saved prefix
fixture passes on CUDA with CPU fallback disabled, and the native C++ stage
passes both fixture CTests. Accepted GLD graph bundles are not yet packaged.
The scratch GLD SIF instructions below remain useful for development.

## Validation and build

Tested with ONNX Runtime **1.23.2**, CUDA toolkit 12.8.61 and GNU C++11.4 inside the
GLD SIF. Host GNU13 builds, but that binary cannot use the SIF's older libstdc++.
Use a matching compiler/runtime pair. The SIF lacks the CUDA toolkit, so expose the
host toolkit read-only:

```sh
APPTAINER_BIND=/usr/local/cuda:/usr/local/cuda:ro just apptainer-shell gld
```

From `/workspace/meta_ws`, install `device_interface` with
`NEUROMESH_ENABLE_CUDA=ON` first. Then configure this library with:

```sh
cmake -S NeuroMesh/onnx_stage -B .gas_scratch/neuromesh_native/sif_onnx_stage -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/workspace/meta_ws/.gas_scratch/neuromesh_native/sif_install -DCMAKE_INSTALL_PREFIX=/workspace/meta_ws/.gas_scratch/neuromesh_native/sif_install -DONNXRUNTIME_INCLUDE_DIR=/workspace/meta_ws/.gas_scratch/neuromesh_ort_sdk/include -DONNXRUNTIME_LIBRARY=/workspace/meta_ws/.gas_scratch/neuromesh_ort_gpu/onnxruntime/capi/libonnxruntime.so.1.23.2 -DNEUROMESH_ONNX_TEST_GRAPH=/workspace/meta_ws/.gas_scratch/neuromesh_phase0/encoder_onnx_cuda_v2/encoder_prefix.onnx -DNEUROMESH_ONNX_FIXTURE_DIR=/workspace/meta_ws/.gas_scratch/neuromesh_phase0/native_prefix_fixture_v1
cmake --build .gas_scratch/neuromesh_native/sif_onnx_stage
ctest --test-dir .gas_scratch/neuromesh_native/sif_onnx_stage --output-on-failure
cmake --install .gas_scratch/neuromesh_native/sif_onnx_stage
```

Supply matching ORT headers from the upstream v1.23.2 session include directory.
The wheel library needs a loader-visible `libonnxruntime.so.1` link and its CUDA
and shared provider libraries alongside it. cuDNN and other CUDA libraries must
also be loader-visible. Our scratch SDK contains relative links to the wheel;
the SIF's `nvidia/*/lib` directories are added to `LD_LIBRARY_PATH`. No system
libraries are modified. Missing dependencies fail the tests, not skip them.

Generate raw little-endian FP32 fixtures with GLD's
`tools.neuromesh.native_prefix_fixture`, using the graph, its CUDA export report,
and the saved prepared batch. This pins graph and batch hashes and records raw
fixture hashes. Preserve/verify these files before testing. CTest tests are
registered only when both graph and fixture paths are supplied.

Both source fixtures pass native-vs-Python ORT with maximum error **0**, including
reused sessions. The test also exercises missing bindings, alias rejection,
closed admission and graph-output metadata mismatch. A separate installed CMake
consumer executes the graph successfully. Core tensor, completion and CUDA
executor tests pass in the same container.

These results qualify this experimental adapter's tested path only. The GLD
PyTorch-vs-ONNX numerical gate still fails. No deployable artifact, full native
GLD inference, TensorRT backend, multi-GPU behavior or live ROS2 integration is
claimed. Native runtime does not require Python; Python only generates fixtures.
