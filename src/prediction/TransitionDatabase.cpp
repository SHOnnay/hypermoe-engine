#include "prediction/TransitionDatabase.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hypermoe::prediction {
namespace {

std::size_t mix(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

bool less(ExpertSelection left, ExpertSelection right) noexcept {
    return left.layerId < right.layerId ||
           (left.layerId == right.layerId && left.expertId < right.expertId);
}

} // namespace

std::size_t ExpertCooccurrenceHash::operator()(
    const ExpertCooccurrence& pair) const noexcept {
    ExpertSelectionHash hash;
    return mix(hash(pair.first), hash(pair.second));
}

TransitionDatabase::TransitionDatabase(std::size_t recentWindow, double decayFactor)
    : recentWindow_(recentWindow), decayFactor_(decayFactor) {
    if (recentWindow == 0 || !std::isfinite(decayFactor) || decayFactor <= 0.0 ||
        decayFactor > 1.0) {
        throw std::invalid_argument("invalid transition database configuration");
    }
}

double TransitionDatabase::decayedValue(const DecayedCounter& counter) const noexcept {
    const auto age = observations_ - counter.lastObservation;
    return counter.value * std::pow(decayFactor_, static_cast<double>(age));
}

void TransitionDatabase::increment(DecayedCounter& counter) noexcept {
    counter.value = decayedValue(counter) + 1.0;
    counter.lastObservation = observations_;
}

ExpertCooccurrence TransitionDatabase::orderedPair(
    ExpertSelection first, ExpertSelection second) noexcept {
    return less(second, first) ? ExpertCooccurrence{second, first}
                               : ExpertCooccurrence{first, second};
}

void TransitionDatabase::record(const router::RouterDecision& decision) {
    record(decision, 0);
}

void TransitionDatabase::record(const router::RouterDecision& decision,
                                std::uint64_t streamId) {
    if (!decision.valid()) throw std::invalid_argument("cannot record invalid routing decision");
    std::vector<ExpertSelection> current;
    current.reserve(decision.selectedExpertIds.size());
    for (const auto expert : decision.selectedExpertIds) {
        current.push_back({decision.layerId, expert});
    }
    std::scoped_lock lock(mutex_);
    const auto previous = previousByStream_.find(streamId);
    for (const auto expert : current) {
        ++frequency_[expert];
        increment(decayedFrequency_[expert]);
    }
    if (previous != previousByStream_.end()) {
        for (const auto from : previous->second) {
            for (const auto to : current) {
                const ExpertTransition transition{from, to};
                ++transitions_[transition];
                increment(decayedTransitions_[transition]);
            }
        }
    }
    for (std::size_t left = 0; left < current.size(); ++left) {
        for (std::size_t right = left + 1; right < current.size(); ++right) {
            ++cooccurrence_[orderedPair(current[left], current[right])];
        }
    }
    previousByStream_[streamId] = current;
    recent_.push_back(std::move(current));
    while (recent_.size() > recentWindow_) recent_.pop_front();
    ++observations_;
}

void TransitionDatabase::endStream(std::uint64_t streamId) {
    std::scoped_lock lock(mutex_);
    previousByStream_.erase(streamId);
}

TransitionDatabaseSnapshot TransitionDatabase::snapshot() const {
    std::scoped_lock lock(mutex_);
    const auto previous = previousByStream_.find(0);
    return {observations_, frequency_, transitions_, cooccurrence_,
            previous == previousByStream_.end() ? std::vector<ExpertSelection>{}
                                                : previous->second,
            {recent_.begin(), recent_.end()}, decayFactor_};
}

std::uint64_t TransitionDatabase::frequency(ExpertSelection expert) const {
    std::scoped_lock lock(mutex_);
    const auto found = frequency_.find(expert);
    return found == frequency_.end() ? 0 : found->second;
}

std::uint64_t TransitionDatabase::transitionCount(
    ExpertSelection from, ExpertSelection to) const {
    std::scoped_lock lock(mutex_);
    const auto found = transitions_.find({from, to});
    return found == transitions_.end() ? 0 : found->second;
}

std::uint64_t TransitionDatabase::cooccurrenceCount(
    ExpertSelection first, ExpertSelection second) const {
    if (first == second) return 0;
    std::scoped_lock lock(mutex_);
    const auto found = cooccurrence_.find(orderedPair(first, second));
    return found == cooccurrence_.end() ? 0 : found->second;
}

std::vector<ExpertSelection> TransitionDatabase::expertsInLayer(LayerId layerId) const {
    std::scoped_lock lock(mutex_);
    std::vector<ExpertSelection> result;
    for (const auto& [expert, count] : frequency_) {
        (void)count;
        if (expert.layerId == layerId) result.push_back(expert);
    }
    std::sort(result.begin(), result.end(), [](const auto left, const auto right) {
        return left.expertId < right.expertId;
    });
    return result;
}

std::vector<ExpertSelection> TransitionDatabase::previousSelections(
    std::uint64_t streamId) const {
    std::scoped_lock lock(mutex_);
    const auto found = previousByStream_.find(streamId);
    return found == previousByStream_.end() ? std::vector<ExpertSelection>{}
                                             : found->second;
}

PredictionStatistics TransitionDatabase::predictionStatistics(
    LayerId targetLayer,
    std::span<const ExpertSelection> sources) const {
    std::scoped_lock lock(mutex_);
    PredictionStatistics result;
    result.outgoingBySource.resize(sources.size());
    for (const auto& [expert, count] : frequency_) {
        if (expert.layerId != targetLayer) continue;
        PredictionCandidateStatistics candidate;
        candidate.expert = expert;
        candidate.frequency = count;
        candidate.incomingBySource.resize(sources.size());
        candidate.decayedIncomingBySource.resize(sources.size());
        const auto decayed = decayedFrequency_.find(expert);
        if (decayed != decayedFrequency_.end()) {
            candidate.decayedFrequency = decayedValue(decayed->second);
        }
        result.layerFrequency += count;
        result.decayedLayerFrequency += candidate.decayedFrequency;
        result.candidates.push_back(std::move(candidate));
    }
    for (const auto& selectionSet : recent_) {
        for (const auto selection : selectionSet) {
            if (selection.layerId != targetLayer) continue;
            ++result.windowLayerFrequency;
            const auto found = std::find_if(
                result.candidates.begin(), result.candidates.end(),
                [selection](const auto& candidate) {
                    return candidate.expert == selection;
                });
            if (found != result.candidates.end()) ++found->windowFrequency;
        }
    }
    std::sort(result.candidates.begin(), result.candidates.end(),
              [](const auto& left, const auto& right) {
                  return left.expert.expertId < right.expert.expertId;
              });
    for (std::size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        for (auto& candidate : result.candidates) {
            const auto found = transitions_.find({sources[sourceIndex], candidate.expert});
            if (found != transitions_.end()) {
                candidate.incomingBySource[sourceIndex] = found->second;
                result.outgoingBySource[sourceIndex] += found->second;
            }
            const auto decayed = decayedTransitions_.find(
                {sources[sourceIndex], candidate.expert});
            if (decayed != decayedTransitions_.end()) {
                candidate.decayedIncomingBySource[sourceIndex] =
                    decayedValue(decayed->second);
            }
        }
    }
    for (auto& candidate : result.candidates) {
        for (const auto& peer : result.candidates) {
            if (candidate.expert == peer.expert) continue;
            const auto found = cooccurrence_.find(
                orderedPair(candidate.expert, peer.expert));
            if (found != cooccurrence_.end()) {
                candidate.cooccurrence.emplace(peer.expert.expertId, found->second);
            }
        }
    }
    return result;
}

} // namespace hypermoe::prediction
