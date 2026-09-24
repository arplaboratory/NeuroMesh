#include "neuromesh/device/tensor.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace neuromesh::device;
void check(bool value) { if (!value) throw std::runtime_error("contract assertion failed"); }
int main() {
  TensorSpec spec{"prefix", DType::float32, {1, 385, 768}, MemoryKind::host, -1};
  std::size_t bytes = 0;
  check(bool(tensor_bytes(spec, bytes)) && bytes == 1182720);
  auto indices = spec; indices.name = "indices"; indices.dtype = DType::int32;
  indices.shape = {27, 30};
  check(bool(tensor_bytes(indices, bytes)) && bytes == 27 * 30 * sizeof(std::int32_t));
  check(bool(tensor_bytes(spec, bytes)) && bytes == 1182720);
  auto owner = std::shared_ptr<void>(new std::uint8_t[bytes], [](void* p) { delete[] static_cast<std::uint8_t*>(p); });
  TensorBinding binding{spec, BufferLease(owner, owner.get(), bytes, MemoryKind::host, -1)};
  check(bool(validate_bindings({spec}, {binding})));
  auto wrong = binding; wrong.spec.dtype = DType::float16;
  check(!validate_bindings({spec}, {wrong}));
  auto short_buffer = TensorBinding{spec, BufferLease(owner, owner.get(), bytes - 1, MemoryKind::host, -1)};
  check(!validate_binding(spec, short_buffer));
  check(!validate_bindings({spec}, {binding, binding}));
  check(!validate_bindings({spec, spec}, {binding, binding}));
  auto second = spec; second.name = "other";
  auto second_binding = binding; second_binding.spec = second;
  check(bool(validate_bindings({spec, second}, {second_binding, binding})));
  check(!validate_bindings({spec, second}, {binding, binding}));
  auto invalid = spec; invalid.shape = {std::numeric_limits<std::int64_t>::max(), 8};
  check(tensor_bytes(invalid, bytes).code == ErrorCode::overflow && bytes == 0);
  invalid.shape = {-1}; check(!tensor_bytes(invalid, bytes));
  invalid = spec; invalid.device_id = 0; check(!tensor_bytes(invalid, bytes));
  wrong = binding; wrong.spec.shape = {385, 1, 768}; check(!validate_binding(spec, wrong));
  auto offset = TensorBinding{spec, BufferLease(owner, owner.get(), 1182728, MemoryKind::host, -1, 1)};
  check(!validate_binding(spec, offset));
  auto absent = TensorBinding{spec, BufferLease({}, owner.get(), 1182720, MemoryKind::host, -1)};
  check(!validate_binding(spec, absent));
  auto impossible = TensorBinding{spec, BufferLease(owner,
    reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 3), 1182720, MemoryKind::host, -1)};
  check(!validate_binding(spec, impossible));
  check(!validate_disjoint_bindings({binding, second_binding}));
  check(!validate_disjoint_bindings({impossible}));
  auto short_spec = spec; short_spec.shape = {1};
  TensorBinding first{short_spec, BufferLease(owner, owner.get(), 1182720, MemoryKind::host, -1)};
  short_spec.name = "adjacent";
  TensorBinding adjacent{short_spec, BufferLease(owner, owner.get(), 1182720, MemoryKind::host, -1, 4)};
  check(bool(validate_disjoint_bindings({first, adjacent})));
  auto pinned_spec = first.spec; pinned_spec.memory = MemoryKind::pinned_host;
  TensorBinding pinned{pinned_spec, BufferLease(owner, owner.get(), 1182720, MemoryKind::pinned_host, -1)};
  check(!validate_disjoint_bindings({first, pinned}));
  std::weak_ptr<void> isolated_lifetime;
  {
    auto allocation = std::shared_ptr<void>(new int(1), [](void* p) { delete static_cast<int*>(p); });
    isolated_lifetime = allocation;
    BufferLease retained(allocation, allocation.get(), sizeof(int), MemoryKind::host, -1);
    allocation.reset(); check(!isolated_lifetime.expired());
  }
  check(isolated_lifetime.expired());
  std::weak_ptr<void> lifetime = owner;
  owner.reset(); check(!lifetime.expired()); // Bindings retain the allocation.
  std::cout << "Native named-binding, overflow and ownership checks passed\n";
}
