#pragma once
#include <neuromesh/device/cuda_executor.hpp>
#include <string>
#include <utility>

namespace neuromesh::onnx {
enum class ArenaGrowth { power_of_two, requested_size };
struct StageOptions {
  ArenaGrowth arena_growth{ArenaGrowth::power_of_two};
  // Specialize named symbolic ONNX dimensions before session creation. Each
  // stage still has one immutable, positive, exact binding signature. Rebuild
  // the stage for a different shape; submissions cannot grow its buffers.
  std::vector<std::pair<std::string, std::int64_t>> dimension_overrides;
  // Explicit caller-verified libraries only; no backend/plugin discovery or
  // fallback. ORT retains registered library lifetimes with the session.
  std::vector<std::string> custom_operator_libraries;
};
enum class ScratchPolicy { retain, release_unused };

// Experimental fixed-shape FP32/INT64 CUDA stage. Caller verifies artifact hashes and
// supplies device-ready contiguous bindings; no preprocessing or host staging.
class Stage {
public:
  Stage(std::string graph, std::vector<device::TensorSpec> inputs,
        std::vector<device::TensorSpec> outputs, int device_id,
        StageOptions options = {});
  ~Stage();
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;
  // Invalid bindings throw before launch. Busy/closed admission follows
  // CudaExecutor. Model/provider errors complete the ticket with failure.
  // release_unused requests CUDA arena shrinkage after this run; it retains
  // weights/live buffers and can cost allocation latency on the next run.
  // Producers must finish writes before submit; consumers wait for completion.
  device::CompletionTicket submit(std::vector<device::TensorBinding> inputs,
                                  std::vector<device::TensorBinding> outputs,
                                  ScratchPolicy scratch = ScratchPolicy::retain);
  void close_and_drain();
private:
  struct State;
  std::unique_ptr<device::CudaExecutor> executor_;
  std::shared_ptr<State> state_;
};
}
