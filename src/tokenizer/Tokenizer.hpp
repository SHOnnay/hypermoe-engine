#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hypermoe::tokenizer {

using TokenId = std::uint32_t;

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    [[nodiscard]] virtual std::vector<TokenId> encode(std::string_view text) const = 0;
    [[nodiscard]] virtual std::string decode(std::span<const TokenId> tokens) const = 0;
    [[nodiscard]] virtual std::size_t vocabularySize() const noexcept = 0;
};

} // namespace hypermoe::tokenizer
