#pragma once

#include "tokenizer/Tokenizer.hpp"

#include <functional>

namespace hypermoe::tokenizer::qwen {

class QwenTokenizerAdapter final : public Tokenizer {
public:
    using Encoder = std::function<std::vector<TokenId>(std::string_view)>;
    using Decoder = std::function<std::string(std::span<const TokenId>)>;

    QwenTokenizerAdapter(std::size_t vocabularySize,
                         Encoder encoder,
                         Decoder decoder);

    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(std::span<const TokenId> tokens) const override;
    [[nodiscard]] std::size_t vocabularySize() const noexcept override;

private:
    void validate(std::span<const TokenId> tokens) const;

    std::size_t vocabularySize_{};
    Encoder encoder_;
    Decoder decoder_;
};

} // namespace hypermoe::tokenizer::qwen
