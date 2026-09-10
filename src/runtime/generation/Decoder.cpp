#include "runtime/generation/Decoder.hpp"

#include "runtime/generation/GenerationModel.hpp"
#include "runtime/generation/InferenceSession.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace hypermoe::runtime::generation {

void Decoder::prefill(InferenceSession& session,
                      std::span<const std::uint32_t> promptTokens) const {
    if (session.generationState().active()) {
        throw std::logic_error("inference session is already active");
    }
    if (promptTokens.empty() ||
        promptTokens.size() > session.kvCache().maximumSequenceLength()) {
        throw std::invalid_argument("prefill prompt exceeds KV cache capacity");
    }
    session.generationState().begin(promptTokens);
    execute(session, promptTokens);
}

void Decoder::decode(InferenceSession& session, std::uint32_t tokenId) const {
    const auto& state = session.generationState();
    if (!state.active() || state.finished() ||
        state.currentTokenPosition() >= session.kvCache().maximumSequenceLength()) {
        throw std::logic_error("inference session cannot decode another token");
    }
    const std::array<std::uint32_t, 1> token{tokenId};
    execute(session, token);
}

void Decoder::execute(InferenceSession& session,
                      std::span<const std::uint32_t> tokenIds) {
    auto& generation = session.generationState();
    if (generation.currentTokenPosition() >
        std::numeric_limits<std::uint64_t>::max() - tokenIds.size()) {
        throw std::overflow_error("generation sequence position overflows");
    }
    InferenceContext context;
    context.batchSize = tokenIds.size();
    context.sequencePosition = generation.currentTokenPosition();
    context.hiddenDimension = session.model().hiddenDimension();
    auto pass = session.model().forward(context, tokenIds, session.kvCache());
    const auto expectedTokens = generation.currentTokenPosition() + tokenIds.size();
    for (std::size_t layer = 0; layer < session.model().layerCount(); ++layer) {
        if (session.kvCache().tokenCount(layer) != expectedTokens) {
            throw std::runtime_error("model forward did not advance every KV cache layer");
        }
    }
    session.forwardState().update(
        std::move(pass), context.sequencePosition,
        static_cast<std::size_t>(expectedTokens));
    generation.advancePosition(tokenIds.size());
}

} // namespace hypermoe::runtime::generation
