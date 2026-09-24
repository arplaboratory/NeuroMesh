#include "neuromesh/device/completion.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
#include <type_traits>
using namespace neuromesh::device;
using namespace std::chrono_literals;
void check(bool condition) { if (!condition) throw std::runtime_error("completion assertion failed"); }
BufferLease allocation(std::weak_ptr<void>& weak) {
  auto owner = std::shared_ptr<void>(new int(3), [](void* p) { delete static_cast<int*>(p); });
  weak = owner;
  return {owner, owner.get(), sizeof(int), MemoryKind::host, -1};
}
int main() {
  static_assert(!std::is_copy_constructible_v<CompletionTicket>);
  InFlightRegistry registry(1);
  std::weak_ptr<void> weak;
  auto first = registry.reserve({allocation(weak)});
  check(!first.ticket.poll());
  check(!first.ticket.wait_for(0ms));
  bool full = false;
  std::weak_ptr<void> rejected;
  try { registry.reserve({allocation(rejected)}); } catch (const std::length_error&) { full = true; }
  check(full && !weak.expired());
  check(first.signal.complete({ErrorCode::execution_failed, "simulated quiescent backend failure"}));
  check(!first.signal.complete({}));
  check(weak.expired());
  check(first.ticket.poll()->code == ErrorCode::execution_failed);
  registry.reap();
  std::weak_ptr<void> retained;
  auto dropped = registry.reserve({allocation(retained)});
  { auto discarded_ticket = std::move(dropped.ticket); }
  check(!retained.expired());
  std::promise<void> started;
  auto draining = std::async(std::launch::async, [&] { started.set_value(); registry.close_and_drain(); });
  started.get_future().wait();
  check(draining.wait_for(10ms) == std::future_status::timeout);
  check(!retained.expired());
  check(dropped.signal.complete({}));
  draining.get();
  check(retained.expired());
  bool closed = false;
  try { registry.reserve({allocation(weak)}); } catch (const std::logic_error&) { closed = true; }
  check(closed);
  auto surviving = [] {
    InFlightRegistry temporary(1);
    std::weak_ptr<void> local;
    auto submission = temporary.reserve({allocation(local)});
    submission.signal.complete({});
    return std::move(submission.ticket);
  }();
  check(bool(*surviving.poll()));
  // A completed failure remains observable after draining.
  check(first.ticket.wait_for(0ms)->code == ErrorCode::execution_failed);
  // Completion is not published until allocation reclamation finishes either.
  InFlightRegistry reclaiming(1);
  std::promise<void> entered, release;
  auto permission = release.get_future().share();
  auto gated_owner = std::shared_ptr<void>(new int(4), [&](void* p) {
    entered.set_value(); permission.wait(); delete static_cast<int*>(p);
  });
  auto gated = reclaiming.reserve({BufferLease(gated_owner, gated_owner.get(), sizeof(int), MemoryKind::host, -1)});
  gated_owner.reset();
  auto completing = std::async(std::launch::async, [&] { return gated.signal.complete({}); });
  entered.get_future().wait();
  check(!gated.ticket.poll());
  check(!gated.ticket.wait_for(1ms));
  release.set_value();
  check(completing.get());
  check(bool(*gated.ticket.wait_for(0ms)));
  reclaiming.close_and_drain();
  std::cout << "Dropped-ticket retention, bounded admission, failure and drain passed\n";
}
