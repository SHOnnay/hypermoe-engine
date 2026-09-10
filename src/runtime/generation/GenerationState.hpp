#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace hypermoe::runtime::generation {

enum class GenerationStopReason : std::uint8_t {
    None = 0,
    MaximumTokens = 1,
    StopToken = 2,
};

class GenerationState {
public:
    explicit GenerationState(std::size_t maximumNewTokens,
                             std::span<const std::uint32_t> stopTokenIds = {});

    void begin(std::span<const std::uint32_t> promptTokens);
    void advancePosition(std::size_t tokenCount);
    [[nodiscard]] bool appendGenerated(std::uint32_t tokenId);
    void finish(GenerationStopReason reason);

    [[nodiscard]] std::uint64_t currentTokenPosition() const noexcept;
    [[nodiscard]] std::size_t maximumNewTokens() const noexcept;
    [[nodiscard]] const std::vector<std::uint32_t>& promptTokens() const noexcept;
    [[nodiscard]] const std::vector<std::uint32_t>& generatedTokens() const noexcept;
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] GenerationStopReason stopReason() const noexcept;

private:
    std::size_t maximumNewTokens_{};
    std::unordered_set<std::uint32_t> stopTokenIds_;
    std::vector<std::uint32_t> promptTokens_;
    std::vector<std::uint32_t> generatedTokens_;
    std::uint64_t currentTokenPosition_{};
    bool active_{};
    GenerationStopReason stopReason_{GenerationStopReason::None};
};

} // namespace hypermoe::runtime::generation
