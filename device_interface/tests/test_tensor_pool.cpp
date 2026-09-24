#include <neuromesh/device/tensor_pool.hpp>
#include <neuromesh/device/completion.hpp>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace neuromesh::device;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("tensor pool contract failed"); }
TensorBinding allocate(const TensorSpec& spec) {
  std::size_t bytes = 0;
  check(bool(tensor_bytes(spec, bytes)));
  auto owner = std::shared_ptr<void>(new unsigned char[bytes], [](void* p) {
    delete[] static_cast<unsigned char*>(p);
  });
  return {spec, BufferLease(owner, owner.get(), bytes, MemoryKind::host, -1)};
}
}
int main() {
  try {
    const std::vector<TensorSpec> specs{
      {"latent", DType::float32, {4}, MemoryKind::host, -1},
      {"cls", DType::float32, {1}, MemoryKind::host, -1}};
    int calls = 0;
    try {
      TensorPool invalid(specs, 2, 39, [&](const auto& s) { ++calls; return allocate(s); });
      check(false);
    } catch (const std::invalid_argument&) {}
    check(calls == 0); // Reject before any allocation.
    std::optional<TensorBinding> escaped;
    {
      TensorPool pool(specs, 2, 40, allocate);
      check(pool.allocated_bytes() == 40);
      auto first = pool.try_acquire();
      auto second = pool.try_acquire();
      check(first && second && !pool.try_acquire());
      auto* active = static_cast<float*>((*first)[0].storage.base());
      active[0] = 123.0f;
      escaped = (*first)[0];
      first.reset();
      check(!pool.try_acquire()); // One tensor pins the whole slot.
      second.reset();
      auto next = pool.try_acquire();
      check(next && (*next)[0].storage.base() != active);
      check(active[0] == 123.0f);
      // Exhausted concurrent callers cannot steal either active slot.
      std::atomic<int> admitted{0};
      std::thread a([&] { if (pool.try_acquire()) ++admitted; });
      std::thread b([&] { if (pool.try_acquire()) ++admitted; });
      a.join(); b.join();
      check(admitted == 0);
    }
    check(*static_cast<float*>(escaped->storage.base()) == 123.0f);
    escaped.reset();
    TensorPool reuse(specs, 1, 20, allocate);
    auto old = reuse.try_acquire();
    auto address = (*old)[0].storage.base();
    old.reset();
    auto fresh = reuse.try_acquire();
    check(fresh && (*fresh)[0].storage.base() == address);
    TensorPool in_flight(specs, 1, 20, allocate);
    InFlightRegistry registry(1);
    std::optional<CompletionSignal> signal;
    {
      auto buffers = in_flight.try_acquire();
      auto submission = registry.reserve({(*buffers)[0].storage});
      signal.emplace(std::move(submission.signal));
    } // Both the caller's bindings and ticket are dropped while work is active.
    const bool retained_by_work = !in_flight.try_acquire();
    signal->complete({});
    check(retained_by_work && bool(in_flight.try_acquire()));
    // Cross-slot aliasing must not defeat the byte bound or corrupt live data.
    auto same = allocate(specs.front());
    try {
      TensorPool alias({specs.front()}, 2, 32, [&](const auto&) { return same; });
      check(false);
    } catch (const std::invalid_argument&) {}
    std::cout << "bounded tensor pool retention, budget, exhaustion and reuse passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
