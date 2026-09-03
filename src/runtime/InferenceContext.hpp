#pragma once

#include "router/RouterDecision.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hypermoe::runtime {

struct ExecutionMetadata {
    std::size_t expertAssignments{};
    std::size_t uniqueExperts{};
    std::uint64_t expertPayloadBytes{};
    std::string tensorBackend;
    std::chrono::nanoseconds routingTime{};
    std::chrono::nanoseconds schedulingTime{};
    std::chrono::nanoseconds expertExecutionTime{};
    std::chrono::nanoseconds expertCombinationTime{};
};

class InferenceContext {
public:
    std::size_t batchSize{};
    std::uint64_t sequencePosition{};
    std::size_t hiddenDimension{};
    LayerId layerIndex{};
    std::vector<router::RouterDecision> routingDecisions;
    ExecutionMetadata execution;

    void validate() const;
    void recordRouting(router::BatchRouterDecision routing,
                       ExecutionMetadata metadata);
    void advanceLayer(LayerId nextLayer);
};

} // namespace hypermoe::runtime
