#include "scheduler/ExpertState.hpp"

#include <stdexcept>

namespace hypermoe::scheduler {

std::string_view toString(ExpertLifecycleState state) noexcept {
    switch (state) {
    case ExpertLifecycleState::Requested: return "REQUESTED";
    case ExpertLifecycleState::Queued: return "QUEUED";
    case ExpertLifecycleState::Loading: return "LOADING";
    case ExpertLifecycleState::Ready: return "READY";
    case ExpertLifecycleState::InUse: return "IN_USE";
    case ExpertLifecycleState::Evicting: return "EVICTING";
    case ExpertLifecycleState::Failed: return "FAILED";
    }
    return "UNKNOWN";
}

void ExpertResidencyStateMachine::registerExpert(LayerId layerId,
                                                  ExpertId id,
                                                  MemoryTier location) {
    std::scoped_lock lock(mutex_);
    if (!states_.emplace(key(layerId, id),
                         ExpertState{layerId, id, location, location,
                                     ExpertLifecycleState::Ready,
                                     std::chrono::steady_clock::now(), 0})
             .second) {
        throw std::invalid_argument("expert state is already registered");
    }
}

void ExpertResidencyStateMachine::registerExpert(ExpertId id, MemoryTier location) {
    registerExpert(0, id, location);
}

void ExpertResidencyStateMachine::request(LayerId layerId,
                                           ExpertId id,
                                           MemoryTier target) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Ready &&
        expert.state != ExpertLifecycleState::Failed) {
        throw std::logic_error("expert cannot be requested during its current transition");
    }
    expert.targetLocation = target;
    expert.state = ExpertLifecycleState::Requested;
}

void ExpertResidencyStateMachine::request(ExpertId id, MemoryTier target) {
    request(0, id, target);
}

void ExpertResidencyStateMachine::markQueued(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Requested) {
        throw std::logic_error("only a requested expert can become queued");
    }
    expert.state = ExpertLifecycleState::Queued;
}

void ExpertResidencyStateMachine::markQueued(ExpertId id) { markQueued(0, id); }

void ExpertResidencyStateMachine::markLoading(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Queued) {
        throw std::logic_error("only a queued expert can start loading");
    }
    expert.state = ExpertLifecycleState::Loading;
}

void ExpertResidencyStateMachine::markLoading(ExpertId id) { markLoading(0, id); }

void ExpertResidencyStateMachine::markReady(LayerId layerId,
                                             ExpertId id,
                                             MemoryTier location) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Loading &&
        expert.state != ExpertLifecycleState::Evicting &&
        expert.state != ExpertLifecycleState::Requested) {
        throw std::logic_error("expert cannot become ready from its current state");
    }
    expert.currentLocation = location;
    expert.targetLocation = location;
    expert.state = ExpertLifecycleState::Ready;
    expert.lastUsed = std::chrono::steady_clock::now();
}

void ExpertResidencyStateMachine::markReady(ExpertId id, MemoryTier location) {
    markReady(0, id, location);
}

void ExpertResidencyStateMachine::acquire(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Ready) {
        throw std::logic_error("only a ready expert can enter use");
    }
    expert.state = ExpertLifecycleState::InUse;
    expert.lastUsed = std::chrono::steady_clock::now();
    ++expert.usageCount;
}

void ExpertResidencyStateMachine::acquire(ExpertId id) { acquire(0, id); }

void ExpertResidencyStateMachine::release(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::InUse) {
        throw std::logic_error("only an in-use expert can be released");
    }
    expert.state = ExpertLifecycleState::Ready;
    expert.lastUsed = std::chrono::steady_clock::now();
}

void ExpertResidencyStateMachine::release(ExpertId id) { release(0, id); }

void ExpertResidencyStateMachine::beginEviction(LayerId layerId,
                                                 ExpertId id,
                                                 MemoryTier target) {
    std::scoped_lock lock(mutex_);
    auto& expert = requireLocked(layerId, id);
    if (expert.state != ExpertLifecycleState::Ready) {
        throw std::logic_error("only a ready expert can be evicted");
    }
    expert.targetLocation = target;
    expert.state = ExpertLifecycleState::Evicting;
}

void ExpertResidencyStateMachine::beginEviction(ExpertId id, MemoryTier target) {
    beginEviction(0, id, target);
}

void ExpertResidencyStateMachine::markFailed(LayerId layerId, ExpertId id) {
    std::scoped_lock lock(mutex_);
    requireLocked(layerId, id).state = ExpertLifecycleState::Failed;
}

void ExpertResidencyStateMachine::markFailed(ExpertId id) { markFailed(0, id); }

ExpertState ExpertResidencyStateMachine::snapshot(LayerId layerId,
                                                   ExpertId id) const {
    std::scoped_lock lock(mutex_);
    const auto it = states_.find(key(layerId, id));
    if (it == states_.end()) throw std::out_of_range("expert state is not registered");
    return it->second;
}

ExpertState ExpertResidencyStateMachine::snapshot(ExpertId id) const {
    return snapshot(0, id);
}

std::vector<ExpertState> ExpertResidencyStateMachine::snapshots() const {
    std::scoped_lock lock(mutex_);
    std::vector<ExpertState> result;
    result.reserve(states_.size());
    for (const auto& [id, state] : states_) {
        (void)id;
        result.push_back(state);
    }
    return result;
}

ExpertState& ExpertResidencyStateMachine::requireLocked(LayerId layerId,
                                                        ExpertId id) {
    const auto it = states_.find(key(layerId, id));
    if (it == states_.end()) throw std::out_of_range("expert state is not registered");
    return it->second;
}

} // namespace hypermoe::scheduler
