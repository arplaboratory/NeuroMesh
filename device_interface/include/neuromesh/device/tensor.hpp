#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace neuromesh::device {

enum class DType { float32, float16, bfloat16, int32, int64, uint8 };
enum class MemoryKind { host, pinned_host, cuda_device };
enum class ErrorCode { ok, invalid_spec, overflow, binding_mismatch, invalid_storage, execution_failed };

struct Status {
  ErrorCode code{ErrorCode::ok};
  std::string message;
  explicit operator bool() const noexcept { return code == ErrorCode::ok; }
};

// Fixed positive dimensions and contiguous C-order only. These are in-process
// values, never a wire representation or permission to dereference a CUDA address.
struct TensorSpec {
  std::string name;
  DType dtype{DType::float32};
  std::vector<std::int64_t> shape;
  MemoryKind memory{MemoryKind::host};
  int device_id{-1};
};

// The allocation owner must actually own base through device completion. A copy
// shares ownership; dropping a binding cannot destroy another copy's allocation.
// Backends must retain leases in an in-flight registry, not only caller tickets.
class BufferLease {
public:
  BufferLease(std::shared_ptr<void> owner, void* base, std::size_t capacity,
              MemoryKind memory, int device_id, std::size_t offset = 0)
      : owner_(std::move(owner)), base_(base), capacity_(capacity), offset_(offset),
        memory_(memory), device_id_(device_id) {}
  const std::shared_ptr<void>& owner() const noexcept { return owner_; }
  void* base() const noexcept { return base_; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t offset() const noexcept { return offset_; }
  MemoryKind memory() const noexcept { return memory_; }
  int device_id() const noexcept { return device_id_; }
private:
  std::shared_ptr<void> owner_;
  void* base_;
  std::size_t capacity_, offset_;
  MemoryKind memory_;
  int device_id_;
};

struct TensorBinding { TensorSpec spec; BufferLease storage; };

// On failure bytes is zero. Overflow is checked before multiplication.
Status tensor_bytes(const TensorSpec& spec, std::size_t& bytes);
Status validate_binding(const TensorSpec& expected, const TensorBinding& actual);
// Binds by name, not vector position. Rejects duplicates, extras and omissions.
Status validate_bindings(const std::vector<TensorSpec>& expected,
                         const std::vector<TensorBinding>& actual);

// Validates storage and rejects overlap within each declared address space.
// Host/pinned-host share an address space; CUDA devices are checked separately.
Status validate_disjoint_bindings(const std::vector<TensorBinding>& bindings);

}  // namespace neuromesh::device
