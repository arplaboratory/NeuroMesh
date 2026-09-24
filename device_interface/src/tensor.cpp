#include "neuromesh/device/tensor.hpp"

#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace neuromesh::device {
namespace {
std::size_t width(DType dtype) {
  switch (dtype) {
    case DType::float32: return 4;
    case DType::float16: case DType::bfloat16: return 2;
    case DType::int32: return 4;
    case DType::int64: return 8;
    case DType::uint8: return 1;
  }
  return 0;
}
bool valid_memory(MemoryKind memory, int device) {
  switch (memory) {
    case MemoryKind::host: case MemoryKind::pinned_host: return device == -1;
    case MemoryKind::cuda_device: return device >= 0;
  }
  return false;
}
Status fail(ErrorCode code, const std::string& name, const char* reason) {
  return {code, name + ": " + reason};
}
}  // namespace

Status tensor_bytes(const TensorSpec& spec, std::size_t& bytes) {
  bytes = 0;
  auto count = width(spec.dtype);
  if (spec.name.empty() || !count || spec.shape.empty() || !valid_memory(spec.memory, spec.device_id))
    return fail(ErrorCode::invalid_spec, spec.name, "invalid name, dtype, rank or memory/device");
  for (auto dim : spec.shape) {
    if (dim <= 0) return fail(ErrorCode::invalid_spec, spec.name, "dimensions must be fixed and positive");
    const auto extent = static_cast<std::uint64_t>(dim);
    if (extent > std::numeric_limits<std::size_t>::max() / count)
      return fail(ErrorCode::overflow, spec.name, "tensor byte count overflow");
    count *= static_cast<std::size_t>(extent);
  }
  bytes = count;
  return {};
}

Status validate_binding(const TensorSpec& expected, const TensorBinding& actual) {
  std::size_t bytes = 0;
  auto status = tensor_bytes(expected, bytes);
  if (!status) return status;
  const auto& spec = actual.spec;
  if (spec.name != expected.name || spec.dtype != expected.dtype || spec.shape != expected.shape ||
      spec.memory != expected.memory || spec.device_id != expected.device_id)
    return fail(ErrorCode::binding_mismatch, expected.name, "name, dtype, shape or device mismatch");
  const auto& lease = actual.storage;
  if (!lease.owner() || !lease.base() || lease.memory() != expected.memory || lease.device_id() != expected.device_id ||
      lease.offset() > lease.capacity() || bytes > lease.capacity() - lease.offset())
    return fail(ErrorCode::invalid_storage, expected.name, "missing owner or insufficient/mismatched allocation");
  const auto address = reinterpret_cast<std::uintptr_t>(lease.base());
  if (lease.offset() > std::numeric_limits<std::uintptr_t>::max() - address ||
      (address + lease.offset()) % width(expected.dtype) != 0 ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - (address + lease.offset()))
    return fail(ErrorCode::invalid_storage, expected.name, "address overflow or unaligned tensor offset");
  return {};
}

Status validate_bindings(const std::vector<TensorSpec>& expected, const std::vector<TensorBinding>& actual) {
  std::unordered_map<std::string, const TensorSpec*> named;
  for (const auto& spec : expected) {
    std::size_t ignored = 0;
    auto status = tensor_bytes(spec, ignored);
    if (!status) return status;
    if (!named.emplace(spec.name, &spec).second)
      return fail(ErrorCode::invalid_spec, spec.name, "duplicate signature name");
  }
  if (actual.size() != expected.size()) return {ErrorCode::binding_mismatch, "binding count mismatch"};
  std::unordered_set<std::string> seen;
  for (const auto& binding : actual) {
    const auto found = named.find(binding.spec.name);
    if (found == named.end() || !seen.insert(binding.spec.name).second)
      return fail(ErrorCode::binding_mismatch, binding.spec.name, "unknown or duplicate binding");
    auto status = validate_binding(*found->second, binding);
    if (!status) return status;
  }
  return {};
}
Status validate_disjoint_bindings(const std::vector<TensorBinding>& bindings) {
  for (const auto& binding : bindings) {
    auto status = validate_binding(binding.spec, binding);
    if (!status) return status;
  }
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto& left = bindings[i];
    std::size_t a_size{};
    tensor_bytes(left.spec, a_size);
    const auto a = reinterpret_cast<std::uintptr_t>(left.storage.base()) + left.storage.offset();
    for (std::size_t j = 0; j < i; ++j) {
      const auto& right = bindings[j];
      const bool a_cuda = left.spec.memory == MemoryKind::cuda_device;
      const bool b_cuda = right.spec.memory == MemoryKind::cuda_device;
      if (a_cuda != b_cuda || (a_cuda && left.spec.device_id != right.spec.device_id)) continue;
      std::size_t b_size{};
      tensor_bytes(right.spec, b_size);
      const auto b = reinterpret_cast<std::uintptr_t>(right.storage.base()) + right.storage.offset();
      if (a < b + b_size && b < a + a_size)
        return fail(ErrorCode::invalid_storage, left.spec.name, "overlapping tensor bindings");
    }
  }
  return {};
}

}  // namespace neuromesh::device
