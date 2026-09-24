#pragma once
#include "neuromesh/device/tensor.hpp"
#include <chrono>
#include <memory>
#include <optional>

namespace neuromesh::device {
namespace detail { struct CompletionState; }

// A disengaged optional means pending, never successful completion.
class CompletionTicket {
public:
  CompletionTicket(CompletionTicket&&) noexcept = default;
  CompletionTicket& operator=(CompletionTicket&&) noexcept = default;
  CompletionTicket(const CompletionTicket&) = delete;
  CompletionTicket& operator=(const CompletionTicket&) = delete;
  std::optional<Status> poll() const;
  std::optional<Status> wait_for(std::chrono::milliseconds timeout) const;
private:
  explicit CompletionTicket(std::shared_ptr<detail::CompletionState> state);
  std::shared_ptr<detail::CompletionState> state_;
  friend class InFlightRegistry;
};

// Backend-only token. Signal ONLY once all device accesses to retained buffers
// have stopped, including after launch failure. An error is not proof of quiescence.
// Destroying this token does not implicitly cancel or complete device work.
class CompletionSignal {
public:
  CompletionSignal(CompletionSignal&&) noexcept = default;
  CompletionSignal& operator=(CompletionSignal&&) noexcept = default;
  CompletionSignal(const CompletionSignal&) = delete;
  CompletionSignal& operator=(const CompletionSignal&) = delete;
  bool complete(Status result);
private:
  explicit CompletionSignal(std::shared_ptr<detail::CompletionState> state);
  std::shared_ptr<detail::CompletionState> state_;
  friend class InFlightRegistry;
};

struct Submission { CompletionTicket ticket; CompletionSignal signal; };

// Reserve BEFORE launching work. The registry, not the ticket, owns in-flight
// leases. Capacity bounds incomplete submissions; reap frees completed slots.
// Host synchronization only: backend event polling is a separate responsibility.
class InFlightRegistry {
public:
  explicit InFlightRegistry(std::size_t capacity);
  ~InFlightRegistry();  // closes and drains; never abandons live buffers
  InFlightRegistry(const InFlightRegistry&) = delete;
  InFlightRegistry& operator=(const InFlightRegistry&) = delete;
  // invalid_argument: empty/missing leases; length_error: full; logic_error: closed.
  // Backend plugin boundaries must translate these exceptions to structured errors.
  Submission reserve(std::vector<BufferLease> leases);
  void reap();
  // Prevents new reservations, waits for all signals. Must run outside callbacks
  // and before backend events/contexts are destroyed. No forced unsafe timeout.
  void close_and_drain();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace neuromesh::device
