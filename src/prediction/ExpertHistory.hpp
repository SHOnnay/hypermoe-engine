#pragma once

#include "router/RouterDecision.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hypermoe::prediction {

struct ExpertSelection {
    LayerId layerId{};
    ExpertId expertId{};

    [[nodiscard]] bool operator==(const ExpertSelection&) const noexcept = default;
};

struct ExpertTransition {
    ExpertSelection from;
    ExpertSelection to;

    [[nodiscard]] bool operator==(const ExpertTransition&) const noexcept = default;
};

struct ExpertSelectionHash {
    [[nodiscard]] std::size_t operator()(ExpertSelection selection) const noexcept;
};

struct ExpertTransitionHash {
    [[nodiscard]] std::size_t operator()(const ExpertTransition& transition) const noexcept;
};

struct ExpertHistorySnapshot {
    std::uint64_t decisions{};
    std::unordered_map<ExpertSelection, std::uint64_t, ExpertSelectionHash> usageFrequency;
    std::unordered_map<ExpertTransition, std::uint64_t, ExpertTransitionHash> layerTransitions;
    std::vector<ExpertSelection> previousSelections;
};

class ExpertHistory {
public:
    void record(const router::RouterDecision& decision);
    [[nodiscard]] std::uint64_t frequency(LayerId layerId,
                                          ExpertId expertId) const;
    [[nodiscard]] std::uint64_t transitionCount(
        ExpertSelection from, ExpertSelection to) const;
    [[nodiscard]] std::vector<ExpertSelection> previousSelections() const;
    [[nodiscard]] ExpertHistorySnapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    std::uint64_t decisions_{};
    std::unordered_map<ExpertSelection, std::uint64_t, ExpertSelectionHash> frequencies_;
    std::unordered_map<ExpertTransition, std::uint64_t, ExpertTransitionHash> transitions_;
    std::vector<ExpertSelection> previous_;
};

} // namespace hypermoe::prediction
