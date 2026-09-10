#pragma once

#include "backend/Backend.hpp"

#include <cstddef>
#include <cstdint>

namespace hypermoe::backend::cuda::kernels {

void activation(int type, const float* input, float* output,
                std::size_t elements, StreamHandle stream);
void rmsNorm(const float* input, const float* weight, float* output,
             std::size_t rows, std::size_t width, float epsilon,
             StreamHandle stream);
void rope(float* values, std::size_t tokens, std::size_t heads,
          std::size_t headDimension, std::size_t positionOffset,
          float theta, StreamHandle stream);
void routerTopK(float* logits, std::size_t tokens, std::size_t experts,
                std::size_t topK, bool softmax, bool renormalize,
                std::uint32_t* selectedIds, float* selectedScores,
                StreamHandle stream);
void causalAttention(const float* query, const float* key, const float* value,
                     float* scores, float* probabilities, float* context,
                     std::size_t queryTokens, std::size_t keyTokens,
                     std::size_t queryHeads, std::size_t keyValueHeads,
                     std::size_t headDimension,
                     std::uint64_t queryPositionOffset,
                     std::uint64_t keyPositionOffset, bool causal,
                     StreamHandle stream);
void gatherRows(const float* input, float* output, const std::uint32_t* rows,
                std::size_t rowCount, std::size_t width, StreamHandle stream);
void scatterAddRows(const float* input, float* output,
                    const std::uint32_t* rows, const float* weights,
                    std::size_t rowCount, std::size_t width,
                    StreamHandle stream);
void zero(float* output, std::size_t elements, StreamHandle stream);

} // namespace hypermoe::backend::cuda::kernels
