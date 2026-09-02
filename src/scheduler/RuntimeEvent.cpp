#include "scheduler/RuntimeEvent.hpp"

#include <stdexcept>
#include <vector>

namespace hypermoe::scheduler {

std::string_view toString(RuntimeEventType type) noexcept {
    switch (type) {
    case RuntimeEventType::ExpertRequested: return "EXPERT_REQUESTED";
    case RuntimeEventType::TransferStarted: return "TRANSFER_STARTED";
    case RuntimeEventType::TransferCompleted: return "TRANSFER_COMPLETED";
    case RuntimeEventType::ExpertReady: return "EXPERT_READY";
    case RuntimeEventType::CacheEvicted: return "CACHE_EVICTED";
    case RuntimeEventType::TransferFailed: return "TRANSFER_FAILED";
    }
    return "UNKNOWN";
}

SubscriptionId RuntimeEventBus::subscribe(EventCallback callback) {
    if (!callback) throw std::invalid_argument("event callback must be callable");
    std::scoped_lock lock(mutex_);
    const auto id = nextId_++;
    subscribers_.emplace(id, std::move(callback));
    return id;
}

bool RuntimeEventBus::unsubscribe(SubscriptionId id) {
    std::scoped_lock lock(mutex_);
    return subscribers_.erase(id) != 0;
}

void RuntimeEventBus::publish(const RuntimeEvent& event) const {
    std::vector<EventCallback> callbacks;
    {
        std::scoped_lock lock(mutex_);
        callbacks.reserve(subscribers_.size());
        for (const auto& [id, callback] : subscribers_) {
            (void)id;
            callbacks.push_back(callback);
        }
    }
    for (const auto& callback : callbacks) {
        try {
            callback(event);
        } catch (...) {
            // Observers cannot fail a scheduler transition.
        }
    }
}

} // namespace hypermoe::scheduler
