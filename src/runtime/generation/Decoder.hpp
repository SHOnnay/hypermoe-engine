#pragma once

#include <cstdint>
#include <span>

namespace hypermoe::runtime::generation {

class InferenceSession;

class Decoder {
public:
    void prefill(InferenceSession& session,
                 std::span<const std::uint32_t> promptTokens) const;
    void decode(InferenceSession& session, std::uint32_t tokenId) const;

private:
    static void execute(InferenceSession& session,
                        std::span<const std::uint32_t> tokenIds);
};

} // namespace hypermoe::runtime::generation
