# Native device contracts

Standalone C++17 library exported as `NeuroMesh::device_interface`. It has no ROS,
CUDA, ONNX Runtime or TensorRT dependency. `gld/cpp/gld_runtime` is its first
consumer. Existing `engine_interface::BaseEngine` and VGGT plugins are unchanged.

Implemented: checked contiguous tensor byte counts, exact named binding validation,
allocation capacity/offset/alignment and declared memory/device checks, shared
buffer leases. Status codes distinguish malformed specs, overflow, binding mismatch
and invalid storage. The allocation owner must actually own the declared pointer;
this library cannot authenticate pointers or inspect device allocation properties.

Implemented completion primitives: move-only tickets with polling and bounded
waits, backend completion signals, and a bounded engine-owned in-flight registry.
The registry retains leases even when the caller drops its ticket.

An optional CUDA executor now owns a stream/event and completion worker (below).
Not implemented: model engine adapters, cross-stream event dependency wrappers,
wire codecs or the GLD request scheduler. The tests cover host ownership and rejection contracts;
they do not establish CUDA completion or backend numerical parity. Aliasing policy across input/output tensors
must be specified by each graph/backend before execution.

```sh
cmake -S NeuroMesh/device_interface -B build/device -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build/device
ctest --test-dir build/device --output-on-failure
cmake --install build/device
```

The standalone package installs a CMake config and version file. Its package.xml
uses the plain CMake build type; no ROS component or plugin is registered. Package
license metadata remains undeclared, consistent with the existing engine package.

## Completion ownership

Validate bindings and call `InFlightRegistry::reserve` **before** launch, retaining
all input/output/workspace leases. Keep the returned `CompletionSignal` in the
backend event monitor; give only the move-only `CompletionTicket` to the caller.
A pending poll or expired host wait returns an empty optional, never success.
Dropping a ticket neither cancels work nor reclaims its buffers.

Call `complete(Status)` only after every device access has stopped. On launch
failure, first establish quiescence; merely catching an exception is insufficient.
Duplicate completion returns false. Allocation deleters execute outside registry
and state locks, and must not wait for their own completion. Final status is
published only after reclamation finishes; tickets may outlive drained registries.
An execution failure is observable as `ErrorCode::execution_failed`.

`close_and_drain` rejects further reservations and waits outside callbacks for all
signals. Keep the event monitor alive until it returns, then destroy device
contexts/events. Registry destruction also drains. Abandoning a completion signal
can therefore block shutdown indefinitely: there is deliberately no timeout that
frees potentially live device buffers. Cancellation must stop future submissions;
it cannot signal completion for kernels still using allocations.

Reservations throw on invalid leases, capacity exhaustion or a closed registry.
Future plugin adapters must translate exceptions to structured backend statuses.
The bound counts incomplete submissions; completed slots are reclaimed by `reap`
or the next reservation. Callers retain their own output leases if they need data
after completion. Metadata/device truthfulness and alias policy remain the
backend's responsibility.

## Optional CUDA executor

Configure `-DNEUROMESH_ENABLE_CUDA=ON` to build `NeuroMesh::cuda_execution` using
CUDAToolkit/cudart. The default remains OFF and the core target stays CUDA-free.
CMake 3.17 or newer is required. Installed CUDA-enabled configs resolve the toolkit
transitively; a CPU-only install does not require it.

`CudaExecutor(device)` owns one nonblocking stream, one reusable event, a worker
and a one-slot in-flight registry. `submit(leases, launch)` reserves before enqueue
and returns immediately with a ticket. The launch callable runs on the worker and
must order **all** device accesses onto the supplied stream, adding dependencies
for any other streams before returning. Validate named graph bindings before submit.
The executor is not a model engine and does not infer graph signatures.

The worker records/waits for a CUDA event, then releases captured resources and
signals completion. An exception after partially enqueued work still goes through
this completion path; outputs are invalid on failure. Event errors trigger a stream
synchronization fallback. If that also fails, the experimental executor terminates
the process rather than recycle potentially live allocations. Device-loss recovery
has not been qualified; this fail-stop behavior is not a production ROS policy.

Close rejects further submissions, keeps the worker alive to finish queued work,
joins it and drains the registry. Destruction releases stream/event resources.
Do not close from launch callbacks or allocation deleters. One submission may enqueue
a whole ordered sequence of device operations; this primitive does not require a
host synchronization between every GLD denoiser/integrator operation. Actual model
adapters and scheduling still need implementation and timing validation.

```sh
cmake -S NeuroMesh/device_interface -B build/device-cuda -DNEUROMESH_ENABLE_CUDA=ON
cmake --build build/device-cuda
ctest --test-dir build/device-cuda --output-on-failure
```

The CUDA test checks real asynchronous fill/copy, pending waits, wrong-device/busy
rejection, partial-launch exceptions, dropped tickets and shutdown retention. It
returns skip code 77 without a GPU; a skipped test is not GPU validation. Tested
on RTX 3500 Ada, CUDA toolkit 12.8.61, driver 580.173.02. No ONNX/TRT or GLD graph
is exercised by this test.

## Bounded CUDA tensor upload

The optional CUDA target also provides `CudaUpload`. Configure a fixed host
tensor signature, device, slot count (1–8), and byte limit outside callbacks.
Each slot preallocates one pinned host slab and one device slab; tensor offsets
are 256-byte aligned and total byte arithmetic is checked against the budget.
Submit validates named host bindings, stages bytes into pinned memory, then
enqueues one slab H2D copy on the existing CUDA executor. No dense allocation
occurs per submission. The synchronous staging copy belongs on a worker.

Output leases pin slots through cache/request ownership; the executor retains
them through device completion even if the returned ticket is dropped. Pool
exhaustion rejects rather than allocating or waiting. The executor admits one
transfer at a time. Close drains queued work; retained output leases can outlive
the uploader. This layer is model-independent and does not implement packet
validation, GPU inference or cross-stream consumer scheduling.

`cuda_upload_contract` checks full 2,131,968-byte FP32/BF16 payload transfer,
name reordering, invalid bindings, pool exhaustion, dropped-ticket drain, closed
admission and output ownership after uploader destruction. Without CUDA it skips
with code 77; execution was verified on the RTX 3500 Ada host.
## CUDA download for source publication

The optional `NeuroMesh::cuda_execution` target includes `CudaDownload`. Configure
fixed CUDA tensor specs, a device, 1–8 pinned slots and a checked per-slot byte
budget. Call `submit` only after producer completion is established. It validates
named input bindings, issues one asynchronous D2H copy per tensor and returns
pinned-host bindings with a completion ticket. Read host bytes only after success.

The pool rejects exhaustion instead of growing. Executor leases retain input and
output allocations even if callers discard their ticket; output leases can outlive
the downloader. `close_and_drain` stops admission and joins active work. Shutdown
does not make a failed device operation successful. The existing CUDA transfer
test now covers exact download bytes, reordered bindings, pool exhaustion,
invalid inputs, dropped tickets and output lifetime after downloader destruction.
GLD packet construction, hashes and ROS messages remain outside NeuroMesh.


TensorPool provides backend-independent bounded output retention. Construct it
with a fixed signature, slot count, byte budget and allocation callback; all
storage is allocated upfront. try_acquire returns no slot under exhaustion.
Every returned tensor lease pins its entire slot, including through an
InFlightRegistry after caller tickets are dropped. The allocator must supply
exact-size standalone nonoverlapping allocations. Consumers must finish writes
before publication and treat retained tensors as immutable. The pool manages
storage, not model identity, freshness, scheduling or a semantic cache.
