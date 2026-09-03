#include "prediction/ExpertHistory.hpp"

#include <functional>
#include <stdexcept>

namespace hypermoe::prediction {

std::size_t ExpertSelectionHash::operator()(ExpertSelection selection) const noexcept {
    const auto value = (static_cast<std::uint64_t>(selection.layerId) << 32U) |
                       selection.expertId;
    return std::hash<std::uint64_t>{}(value);
}

std::size_t ExpertTransitionHash::operator()(
    const ExpertTransition& transition) const noexcept {
    const auto left = ExpertSelectionHash{}(transition.from);
    const auto right = ExpertSelectionHash{}(transition.to);
    return left ^ (right + 0x9E3779B97F4A7C15ULL + (left << 6U) + (left >> 2U));
}

void ExpertHistory::record(const router::RouterDecision& decision) {
    if (!decision.valid()) throw std::invalid_argument("cannot record invalid routing decision");
    std::vector<ExpertSelection> current;
    current.reserve(decision.selectedExpertIds.size());
    for (const auto expert : decision.selectedExpertIds) {
        current.push_back({decision.layerId, expert});
    }
    std::scoped_lock lock(mutex_);
    for (const auto selection : current) ++frequencies_[selection];
    for (const auto from : previous_) {
        for (const auto to : current) ++transitions_[{from, to}];
    }
    previous_ = std::move(current);
    ++decisions_;
}

std::uint64_t ExpertHistory::frequency(LayerId layerId, ExpertId expertId) const {
    std::scoped_lock lock(mutex_);
    const auto found = frequencies_.find({layerId, expertId});
    return found == frequencies_.end() ? 0 : found->second;
}

std::uint64_t ExpertHistory::transitionCount(ExpertSelection from,
                                             ExpertSelection to) const {
    std::scoped_lock lock(mutex_);
    const auto found = transitions_.find({from, to});
    return found == transitions_.end() ? 0 : found->second;
}

std::vector<ExpertSelection> ExpertHistory::previousSelections() const {
    std::scoped_lock lock(mutex_);
    return previous_;
}

ExpertHistorySnapshot ExpertHistory::snapshot() const {
    std::scoped_lock lock(mutex_);
    return {decisions_, frequencies_, transitions_, previous_};
}

} // namespace hypermoe::prediction
