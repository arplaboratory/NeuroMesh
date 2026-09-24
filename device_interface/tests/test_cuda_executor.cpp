#include "neuromesh/device/cuda_executor.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
using namespace neuromesh::device;
using namespace std::chrono_literals;
void check(bool value) { if (!value) throw std::runtime_error("CUDA executor assertion failed"); }
void cuda_check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
int main() {
  int count = 0;
  const auto available = cudaGetDeviceCount(&count);
  if (available != cudaSuccess || !count) {
    std::cerr << "CUDA unavailable: " << cudaGetErrorString(available) << '\n'; return 77;
  }
  cuda_check(cudaSetDevice(0));
  constexpr std::size_t size = 4096;
  void* device = nullptr; void* host = nullptr;
  cuda_check(cudaMalloc(&device, size)); cuda_check(cudaMallocHost(&host, size));
  auto d = std::shared_ptr<void>(device, [](void* p) { cudaFree(p); });
  auto h = std::shared_ptr<void>(host, [](void* p) { cudaFreeHost(p); });
  CudaExecutor executor(0);
  bool wrong_device = false;
  try { executor.submit({BufferLease(d, device, size, MemoryKind::cuda_device, 1)}, [](auto) { return Status{}; }); }
  catch (const std::invalid_argument&) { wrong_device = true; }
  check(wrong_device);
  std::promise<void> release, entered;
  auto permission = release.get_future().share();
  auto ticket = executor.submit({BufferLease(d, device, size, MemoryKind::cuda_device, 0),
                                 BufferLease(h, host, size, MemoryKind::pinned_host, -1)},
    [&, permission](cudaStream_t stream) {
      entered.set_value(); permission.wait();
      cuda_check(cudaMemsetAsync(device, 42, size, stream));
      cuda_check(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, stream));
      return Status{};
    });
  entered.get_future().wait();
  check(!ticket.wait_for(1ms));
  bool busy = false;
  try { executor.submit({BufferLease(d, device, size, MemoryKind::cuda_device, 0)}, [](auto) { return Status{}; }); }
  catch (const std::length_error&) { busy = true; }
  check(busy);
  release.set_value();
  auto done = ticket.wait_for(5s); check(done.has_value() && bool(*done));
  for (std::size_t i = 0; i < size; ++i) check(static_cast<unsigned char*>(host)[i] == 42);
  auto failed = executor.submit({BufferLease(d, device, size, MemoryKind::cuda_device, 0)}, [&](cudaStream_t stream) -> Status {
    cuda_check(cudaMemsetAsync(device, 0, size, stream));
    throw std::runtime_error("launch callback failure after queued work");
  });
  auto error = failed.wait_for(5s); check(error && error->code == ErrorCode::execution_failed);
  std::weak_ptr<void> retained = d;
  std::promise<void> finish, running;
  auto permit = finish.get_future().share();
  {
    auto dropped = executor.submit({BufferLease(d, device, size, MemoryKind::cuda_device, 0)}, [&, permit](cudaStream_t stream) {
      running.set_value(); permit.wait();
      cuda_check(cudaMemsetAsync(device, 7, size, stream)); return Status{};
    });
  }
  d.reset(); running.get_future().wait(); check(!retained.expired());
  auto closing = std::async(std::launch::async, [&] { executor.close_and_drain(); });
  check(closing.wait_for(10ms) == std::future_status::timeout);
  finish.set_value(); closing.get(); check(retained.expired());
  bool closed = false;
  try { executor.submit({BufferLease(h, host, size, MemoryKind::pinned_host, -1)}, [](auto) { return Status{}; }); }
  catch (const std::logic_error&) { closed = true; }
  check(closed);
  std::cout << "CUDA stream/event completion, queued failure, dropped ticket and shutdown passed\n";
}
