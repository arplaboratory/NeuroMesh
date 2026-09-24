#include "neuromesh/device/tensor_pool.hpp"
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace neuromesh::device {
TensorPool::TensorPool(std::vector<TensorSpec> signature, std::size_t slots,
                      std::size_t max_bytes, const Allocator& allocate) {
  if (signature.empty() || !slots || !allocate)
    throw std::invalid_argument("tensor pool requires a signature, slots and allocator");
  std::size_t slot_bytes = 0;
  std::unordered_set<std::string> names;
  for (const auto& spec : signature) {
    std::size_t bytes = 0;
    if (!tensor_bytes(spec, bytes) || !names.insert(spec.name).second ||
        bytes > std::numeric_limits<std::size_t>::max() - slot_bytes)
      throw std::invalid_argument("invalid tensor pool signature");
    slot_bytes += bytes;
  }
  if (slot_bytes > max_bytes / slots)
    throw std::invalid_argument("tensor pool exceeds allocation budget");
  allocated_bytes_ = slot_bytes * slots;
  slots_.reserve(slots);
  std::vector<TensorBinding> all;
  for (std::size_t i = 0; i < slots; ++i) {
    Slot slot;
    for (const auto& spec : signature) {
      auto binding = allocate(spec);
      std::size_t bytes = 0;
      tensor_bytes(spec, bytes);
      if (!validate_binding(spec, binding) || binding.storage.offset() != 0 ||
          binding.storage.capacity() != bytes)
        throw std::invalid_argument("pool allocator must return exact-size standalone storage");
      all.push_back(binding);
      slot.backing.push_back(std::move(binding));
    }
    slots_.push_back(std::move(slot));
  }
  if (!validate_disjoint_bindings(all))
    throw std::invalid_argument("pool allocations overlap");
}
std::optional<std::vector<TensorBinding>> TensorPool::try_acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& slot : slots_) {
    if (!slot.active.expired()) continue;
    auto retained = std::make_shared<Retained>(Retained{slot.backing});
    std::vector<TensorBinding> result;
    result.reserve(slot.backing.size());
    for (const auto& binding : slot.backing) {
      const auto& storage = binding.storage;
      result.push_back({binding.spec, BufferLease(retained, storage.base(),
          storage.capacity(), storage.memory(), storage.device_id())});
    }
    slot.active = retained;
    return result;
  }
  return std::nullopt;
}
}  // namespace neuromesh::device
