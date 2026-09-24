#pragma once
#include "neuromesh/device/tensor.hpp"
#include <functional>
#include <mutex>
#include <optional>

namespace neuromesh::device {
// Preallocated, nonblocking storage for completed-stage outputs. All allocations
// count against max_bytes, including slots retained by downstream GPU work.
// Publication/cache identity and freshness remain the model adapter's concern.
class TensorPool {
public:
  using Allocator = std::function<TensorBinding(const TensorSpec&)>;
  TensorPool(std::vector<TensorSpec> signature, std::size_t slots,
             std::size_t max_bytes, const Allocator& allocate);
  // No eviction: exhaustion returns nullopt. Every returned binding pins its
  // entire slot, even if the vector or pool is destroyed. Producers must finish
  // writes before handing bindings to readers; readers must treat them immutable.
  std::optional<std::vector<TensorBinding>> try_acquire();
  std::size_t allocated_bytes() const noexcept { return allocated_bytes_; }
private:
  struct Retained { std::vector<TensorBinding> backing; };
  struct Slot {
    std::vector<TensorBinding> backing;
    std::weak_ptr<Retained> active;
  };
  std::mutex mutex_;
  std::vector<Slot> slots_;
  std::size_t allocated_bytes_{};
};
}  // namespace neuromesh::device
