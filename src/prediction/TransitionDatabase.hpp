#pragma once

#include "prediction/ExpertHistory.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace hypermoe::prediction {

struct ExpertCooccurrence {
    ExpertSelection first;
    ExpertSelection second;

    [[nodiscard]] bool operator==(const ExpertCooccurrence&) const noexcept = default;
};

struct ExpertCooccurrenceHash {
    [[nodiscard]] std::size_t
    operator()(const ExpertCooccurrence& pair) const noexcept;
};

struct TransitionDatabaseSnapshot {
    std::uint64_t observations{};
    std::unordered_map<ExpertSelection, std::uint64_t, ExpertSelectionHash> frequency;
    std::unordered_map<ExpertTransition, std::uint64_t, ExpertTransitionHash> transitions;
    std::unordered_map<ExpertCooccurrence, std::uint64_t, ExpertCooccurrenceHash>
        cooccurrence;
    std::vector<ExpertSelection> previousSelections;
    std::vector<std::vector<ExpertSelection>> recentSelections;
};

struct PredictionCandidateStatistics {
    ExpertSelection expert;
    std::uint64_t frequency{};
    std::vector<std::uint64_t> incomingBySource;
    std::unordered_map<ExpertId, std::uint64_t> cooccurrence;
};

struct PredictionStatistics {
    std::uint64_t layerFrequency{};
    std::vector<std::uint64_t> outgoingBySource;
    std::vector<PredictionCandidateStatistics> candidates;
};

class TransitionDatabase {
public:
    explicit TransitionDatabase(std::size_t recentWindow = 16);

    void record(const router::RouterDecision& decision);
    void record(const router::RouterDecision& decision, std::uint64_t streamId);
    void endStream(std::uint64_t streamId);
    [[nodiscard]] TransitionDatabaseSnapshot snapshot() const;
    [[nodiscard]] std::uint64_t frequency(ExpertSelection expert) const;
    [[nodiscard]] std::uint64_t transitionCount(ExpertSelection from,
                                                ExpertSelection to) const;
    [[nodiscard]] std::uint64_t cooccurrenceCount(ExpertSelection first,
                                                  ExpertSelection second) const;
    [[nodiscard]] std::vector<ExpertSelection> expertsInLayer(LayerId layerId) const;
    [[nodiscard]] std::vector<ExpertSelection>
    previousSelections(std::uint64_t streamId) const;
    [[nodiscard]] PredictionStatistics predictionStatistics(
        LayerId targetLayer,
        std::span<const ExpertSelection> sources) const;
    [[nodiscard]] static ExpertCooccurrence orderedPair(ExpertSelection first,
                                                         ExpertSelection second) noexcept;

private:
    const std::size_t recentWindow_;
    mutable std::mutex mutex_;
    std::uint64_t observations_{};
    std::unordered_map<ExpertSelection, std::uint64_t, ExpertSelectionHash> frequency_;
    std::unordered_map<ExpertTransition, std::uint64_t, ExpertTransitionHash> transitions_;
    std::unordered_map<ExpertCooccurrence, std::uint64_t, ExpertCooccurrenceHash>
        cooccurrence_;
    std::unordered_map<std::uint64_t, std::vector<ExpertSelection>> previousByStream_;
    std::deque<std::vector<ExpertSelection>> recent_;
};

} // namespace hypermoe::prediction
