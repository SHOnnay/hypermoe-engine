#include "tokenizer/qwen/QwenTokenizerAdapter.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::tokenizer::qwen {

QwenTokenizerAdapter::QwenTokenizerAdapter(std::size_t vocabularySize,
                                           Encoder encoder,
                                           Decoder decoder)
    : vocabularySize_(vocabularySize),
      encoder_(std::move(encoder)),
      decoder_(std::move(decoder)) {
    if (vocabularySize_ == 0 || !encoder_ || !decoder_) {
        throw std::invalid_argument(
            "Qwen tokenizer adapter requires vocabulary metadata and callbacks");
    }
}

std::vector<TokenId> QwenTokenizerAdapter::encode(std::string_view text) const {
    auto tokens = encoder_(text);
    validate(tokens);
    return tokens;
}

std::string QwenTokenizerAdapter::decode(
    std::span<const TokenId> tokens) const {
    validate(tokens);
    return decoder_(tokens);
}

std::size_t QwenTokenizerAdapter::vocabularySize() const noexcept {
    return vocabularySize_;
}

void QwenTokenizerAdapter::validate(std::span<const TokenId> tokens) const {
    for (const auto token : tokens) {
        if (token >= vocabularySize_) {
            throw std::out_of_range(
                "Qwen tokenizer adapter produced a token outside its vocabulary");
        }
    }
}

} // namespace hypermoe::tokenizer::qwen
