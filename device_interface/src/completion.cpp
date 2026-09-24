#include "neuromesh/device/completion.hpp"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace neuromesh::device {
namespace detail {
struct CompletionState {
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::optional<Status> result;
  bool completing{false};
  std::vector<BufferLease> leases;
};
}
CompletionTicket::CompletionTicket(std::shared_ptr<detail::CompletionState> state) : state_(std::move(state)) {}
CompletionSignal::CompletionSignal(std::shared_ptr<detail::CompletionState> state) : state_(std::move(state)) {}
std::optional<Status> CompletionTicket::poll() const {
  if (!state_) throw std::logic_error("moved-from completion ticket");
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->result;
}
std::optional<Status> CompletionTicket::wait_for(std::chrono::milliseconds timeout) const {
  if (!state_) throw std::logic_error("moved-from completion ticket");
  if (timeout.count() < 0) throw std::invalid_argument("negative completion timeout");
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->changed.wait_for(lock, timeout, [&] { return state_->result.has_value(); });
  return state_->result;
}
bool CompletionSignal::complete(Status result) {
  if (!state_) throw std::logic_error("moved-from completion signal");
  std::vector<BufferLease> released;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->completing || state_->result) return false;
    state_->completing = true;
    released.swap(state_->leases);
  }
  // User-provided allocation deleters run outside the state/registry locks.
  released.clear();
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->result = std::move(result);
  }
  state_->changed.notify_all();
  return true;
}
struct InFlightRegistry::Impl {
  explicit Impl(std::size_t count) : capacity(count) {}
  std::mutex mutex;
  std::size_t capacity;
  bool closed{false};
  std::vector<std::shared_ptr<detail::CompletionState>> states;
  void reap_locked() {
    states.erase(std::remove_if(states.begin(), states.end(), [](const auto& state) {
      std::lock_guard<std::mutex> lock(state->mutex);
      return state->result.has_value();
    }), states.end());
  }
};
InFlightRegistry::InFlightRegistry(std::size_t capacity) : impl_(std::make_unique<Impl>(capacity)) {
  if (!capacity) throw std::invalid_argument("in-flight capacity must be positive");
}
InFlightRegistry::~InFlightRegistry() { close_and_drain(); }
Submission InFlightRegistry::reserve(std::vector<BufferLease> leases) {
  if (leases.empty()) throw std::invalid_argument("submission requires retained leases");
  for (const auto& lease : leases)
    if (!lease.owner() || !lease.base()) throw std::invalid_argument("submission has unowned storage");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed) throw std::logic_error("registry closed");
  impl_->reap_locked();
  if (impl_->states.size() == impl_->capacity) throw std::length_error("in-flight capacity exhausted");
  auto state = std::make_shared<detail::CompletionState>();
  state->leases = std::move(leases);
  impl_->states.push_back(state);
  return {CompletionTicket(state), CompletionSignal(state)};
}
void InFlightRegistry::reap() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->reap_locked();
}
void InFlightRegistry::close_and_drain() {
  std::vector<std::shared_ptr<detail::CompletionState>> pending;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
    pending = impl_->states;
  }
  for (const auto& state : pending) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->changed.wait(lock, [&] { return state->result.has_value(); });
  }
  reap();
}
}  // namespace neuromesh::device
