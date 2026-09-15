#include "profiling/GpuEventQueue.hpp"
#include "profiling/Profiler.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hypermoe::profiling {
GpuEventQueue::GpuEventQueue(GpuEventApi api, std::shared_ptr<Profiler> profiler)
    : api_(std::move(api)), profiler_(std::move(profiler)) {
    if (!api_.create || !api_.record || !api_.ready || !api_.elapsedMs ||
        !api_.destroy || !api_.synchronize) {
        throw std::invalid_argument("GPU event queue requires a complete event API");
    }
}

GpuEventQueue::~GpuEventQueue() {
    // Teardown only: mandatory normal completion boundaries collect without
    // waiting. Keep owners through shutdown if a caller left work in flight.
    std::scoped_lock lock(mutex_);
    for (auto& record : pending_) {
        try { api_.synchronize(record.stream); } catch (...) {}
        try { api_.destroy(record.start); api_.destroy(record.end); } catch (...) {}
    }
}

GpuEventQueue::Scope::~Scope() { if (queue_) queue_->abort(*this); }
GpuEventQueue::Scope::Scope(Scope&& other) noexcept
    : queue_(std::exchange(other.queue_, nullptr)), operation_(other.operation_),
      stream_(other.stream_), start_(other.start_), end_(other.end_),
      owners_(std::move(other.owners_)) {}
void GpuEventQueue::Scope::finish() {
    if (!queue_) throw std::logic_error("GPU operation scope already finished");
    queue_->finish(*this);
}

GpuEventQueue::Scope GpuEventQueue::begin(
    GpuOperation operation, backend::StreamHandle stream,
    std::vector<std::shared_ptr<void>> owners) {
    Scope scope;
    scope.queue_ = this; scope.operation_ = operation; scope.stream_ = stream;
    scope.owners_ = std::move(owners);
    if (profiler_) {
        scope.start_ = api_.create(true);
        api_.record(scope.start_, stream);
    }
    scope.end_ = api_.create(profiler_ != nullptr);
    return scope;
}

void GpuEventQueue::finish(Scope& scope) {
    api_.record(scope.end_, scope.stream_);
    // Copy owners into the record so an allocation failure still leaves the
    // Scope able to complete its exception-only lifetime barrier.
    std::scoped_lock lock(mutex_);
    pending_.push_back({scope.operation_, scope.stream_, scope.start_, scope.end_, scope.owners_});
    scope.queue_ = nullptr;
    scope.owners_.clear();
}

void GpuEventQueue::abort(Scope& scope) noexcept {
    try { api_.synchronize(scope.stream_); } catch (...) {}
    try { api_.destroy(scope.start_); api_.destroy(scope.end_); } catch (...) {}
    scope.owners_.clear(); scope.queue_ = nullptr;
}

void GpuEventQueue::collect() {
    std::scoped_lock lock(mutex_);
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (!api_.ready(it->end)) { ++it; continue; }
        if (profiler_) {
            const auto milliseconds = api_.elapsedMs(it->start, it->end);
            if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
                throw std::runtime_error("GPU event elapsed time is invalid");
            }
            profiler_->recordGpuTime(it->operation,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::duration<double, std::milli>(milliseconds)));
        }
        api_.destroy(it->start); api_.destroy(it->end);
        it = pending_.erase(it);
    }
}

std::size_t GpuEventQueue::pending() const {
    std::scoped_lock lock(mutex_); return pending_.size();
}
} // namespace hypermoe::profiling
