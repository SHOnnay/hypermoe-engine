#include "backend/cuda/CudaKernels.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hypermoe::backend::cuda::kernels {
namespace {

constexpr unsigned threadsPerBlock = 256U;

__global__ void activationKernel(int type, const float* input, float* output,
                                 std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto value = input[index];
    if (type == 0) {
        output[index] = value / (1.0F + expf(-value));
    } else {
        constexpr float inverseSqrtTwo = 0.70710678118654752440F;
        output[index] = 0.5F * value * (1.0F + erff(value * inverseSqrtTwo));
    }
}

__global__ void rmsNormKernel(const float* input, const float* weight,
                              float* output, std::size_t rows,
                              std::size_t width, float epsilon) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double squareSum = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
        const auto value = input[row * width + column];
        squareSum += static_cast<double>(value) * value;
    }
    const auto inverse = static_cast<float>(1.0 / sqrt(
        squareSum / static_cast<double>(width) + static_cast<double>(epsilon)));
    for (std::size_t column = 0; column < width; ++column) {
        output[row * width + column] =
            input[row * width + column] * inverse * weight[column];
    }
}

__global__ void ropeKernel(float* values, std::size_t tokens,
                           std::size_t heads, std::size_t headDimension,
                           std::size_t positionOffset, float theta) {
    const auto pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto pairsPerHead = headDimension / 2U;
    const auto totalPairs = tokens * heads * pairsPerHead;
    if (pair >= totalPairs) return;
    const auto dimensionPair = pair % pairsPerHead;
    const auto headAndToken = pair / pairsPerHead;
    const auto head = headAndToken % heads;
    const auto token = headAndToken / heads;
    const auto dimension = dimensionPair * 2U;
    const auto exponent = static_cast<double>(dimension) /
                          static_cast<double>(headDimension);
    const auto angle = static_cast<double>(positionOffset + token) /
                       pow(static_cast<double>(theta), exponent);
    const auto cosine = static_cast<float>(cos(angle));
    const auto sine = static_cast<float>(sin(angle));
    const auto base = (token * heads + head) * headDimension + dimension;
    const auto first = values[base];
    const auto second = values[base + 1U];
    values[base] = first * cosine - second * sine;
    values[base + 1U] = first * sine + second * cosine;
}

__global__ void routerTopKKernel(float* logits, std::size_t tokens,
                                 std::size_t experts, std::size_t topK,
                                 bool softmax, bool renormalize,
                                 std::uint32_t* selectedIds,
                                 float* selectedScores) {
    const auto token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    auto* row = logits + token * experts;
    if (softmax) {
        auto maximum = row[0];
        for (std::size_t expert = 1; expert < experts; ++expert) {
            maximum = fmaxf(maximum, row[expert]);
        }
        double sum = 0.0;
        for (std::size_t expert = 0; expert < experts; ++expert) {
            row[expert] = expf(row[expert] - maximum);
            sum += row[expert];
        }
        for (std::size_t expert = 0; expert < experts; ++expert) {
            row[expert] = static_cast<float>(static_cast<double>(row[expert]) / sum);
        }
    }
    double selectedSum = 0.0;
    for (std::size_t rank = 0; rank < topK; ++rank) {
        std::size_t best = experts;
        float bestScore = -CUDART_INF_F;
        for (std::size_t expert = 0; expert < experts; ++expert) {
            bool alreadySelected = false;
            for (std::size_t prior = 0; prior < rank; ++prior) {
                alreadySelected = alreadySelected ||
                    selectedIds[token * topK + prior] == expert;
            }
            if (!alreadySelected &&
                (row[expert] > bestScore ||
                 (row[expert] == bestScore && expert < best))) {
                best = expert;
                bestScore = row[expert];
            }
        }
        selectedIds[token * topK + rank] = static_cast<std::uint32_t>(best);
        selectedScores[token * topK + rank] = bestScore;
        selectedSum += bestScore;
    }
    if (renormalize && isfinite(selectedSum) && selectedSum != 0.0) {
        for (std::size_t rank = 0; rank < topK; ++rank) {
            selectedScores[token * topK + rank] = static_cast<float>(
                static_cast<double>(selectedScores[token * topK + rank]) /
                selectedSum);
        }
    } else if (renormalize) {
        selectedScores[token * topK] = CUDART_NAN_F;
    }
}

__global__ void causalAttentionKernel(
    const float* query, const float* key, const float* value,
    float* scores, float* probabilities, float* context,
    std::size_t queryTokens, std::size_t keyTokens,
    std::size_t queryHeads, std::size_t keyValueHeads,
    std::size_t headDimension, std::uint64_t queryPositionOffset,
    std::uint64_t keyPositionOffset, bool causal) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    const auto head = static_cast<std::size_t>(blockIdx.y);
    if (row >= queryTokens || head >= queryHeads || threadIdx.x != 0) return;
    const auto headsPerKeyValue = queryHeads / keyValueHeads;
    const auto keyValueHead = head / headsPerKeyValue;
    const auto queryWidth = queryHeads * headDimension;
    const auto scoreBase = (head * queryTokens + row) * keyTokens;
    const auto scale = 1.0 / sqrt(static_cast<double>(headDimension));
    float maximum = -CUDART_INF_F;
    for (std::size_t column = 0; column < keyTokens; ++column) {
        const auto keyPosition = keyPositionOffset + column;
        if (causal && keyPosition > queryPositionOffset + row) {
            scores[scoreBase + column] = -CUDART_INF_F;
            continue;
        }
        double dot = 0.0;
        for (std::size_t feature = 0; feature < headDimension; ++feature) {
            const auto queryIndex = row * queryWidth + head * headDimension + feature;
            const auto keyIndex =
                (column * keyValueHeads + keyValueHead) * headDimension + feature;
            dot += static_cast<double>(query[queryIndex]) * key[keyIndex];
        }
        const auto score = static_cast<float>(dot * scale);
        scores[scoreBase + column] = score;
        maximum = fmaxf(maximum, score);
    }
    double denominator = 0.0;
    for (std::size_t column = 0; column < keyTokens; ++column) {
        const auto score = scores[scoreBase + column];
        const auto probability = isfinite(score) ? expf(score - maximum) : 0.0F;
        probabilities[scoreBase + column] = probability;
        denominator += probability;
    }
    if (!isfinite(denominator) || denominator <= 0.0) {
        probabilities[scoreBase] = CUDART_NAN_F;
        return;
    }
    for (std::size_t feature = 0; feature < headDimension; ++feature) {
        context[row * queryWidth + head * headDimension + feature] = 0.0F;
    }
    for (std::size_t column = 0; column < keyTokens; ++column) {
        const auto probability = static_cast<float>(
            static_cast<double>(probabilities[scoreBase + column]) / denominator);
        probabilities[scoreBase + column] = probability;
        for (std::size_t feature = 0; feature < headDimension; ++feature) {
            const auto valueIndex =
                (column * keyValueHeads + keyValueHead) * headDimension + feature;
            context[row * queryWidth + head * headDimension + feature] +=
                probability * value[valueIndex];
        }
    }
}

__global__ void gatherRowsKernel(const float* input, float* output,
                                 const std::uint32_t* rows,
                                 std::size_t rowCount, std::size_t width) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rowCount * width) return;
    const auto row = index / width;
    const auto column = index % width;
    output[index] = input[static_cast<std::size_t>(rows[row]) * width + column];
}

__global__ void scatterAddRowsKernel(const float* input, float* output,
                                     const std::uint32_t* rows,
                                     const float* weights,
                                     std::size_t rowCount, std::size_t width) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rowCount * width) return;
    const auto row = index / width;
    const auto column = index % width;
    atomicAdd(output + static_cast<std::size_t>(rows[row]) * width + column,
              input[index] * weights[row]);
}

__global__ void zeroKernel(float* output, std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = 0.0F;
}

void checkLaunch(const char* operation) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

unsigned blocks(std::size_t elements) {
    return static_cast<unsigned>((elements + threadsPerBlock - 1U) / threadsPerBlock);
}

} // namespace

void activation(int type, const float* input, float* output,
                std::size_t elements, StreamHandle stream) {
    activationKernel<<<blocks(elements), threadsPerBlock, 0,
                       static_cast<cudaStream_t>(stream)>>>(
        type, input, output, elements);
    checkLaunch("CUDA activation kernel");
}

void rmsNorm(const float* input, const float* weight, float* output,
             std::size_t rows, std::size_t width, float epsilon,
             StreamHandle stream) {
    rmsNormKernel<<<blocks(rows), threadsPerBlock, 0,
                    static_cast<cudaStream_t>(stream)>>>(
        input, weight, output, rows, width, epsilon);
    checkLaunch("CUDA RMSNorm kernel");
}

void rope(float* values, std::size_t tokens, std::size_t heads,
          std::size_t headDimension, std::size_t positionOffset,
          float theta, StreamHandle stream) {
    const auto pairs = tokens * heads * (headDimension / 2U);
    ropeKernel<<<blocks(pairs), threadsPerBlock, 0,
                 static_cast<cudaStream_t>(stream)>>>(
        values, tokens, heads, headDimension, positionOffset, theta);
    checkLaunch("CUDA RoPE kernel");
}

void routerTopK(float* logits, std::size_t tokens, std::size_t experts,
                std::size_t topK, bool softmax, bool renormalize,
                std::uint32_t* selectedIds, float* selectedScores,
                StreamHandle stream) {
    routerTopKKernel<<<blocks(tokens), threadsPerBlock, 0,
                       static_cast<cudaStream_t>(stream)>>>(
        logits, tokens, experts, topK, softmax, renormalize,
        selectedIds, selectedScores);
    checkLaunch("CUDA router top-k kernel");
}

void causalAttention(const float* query, const float* key, const float* value,
                     float* scores, float* probabilities, float* context,
                     std::size_t queryTokens, std::size_t keyTokens,
                     std::size_t queryHeads, std::size_t keyValueHeads,
                     std::size_t headDimension,
                     std::uint64_t queryPositionOffset,
                     std::uint64_t keyPositionOffset, bool causal,
                     StreamHandle stream) {
    causalAttentionKernel<<<dim3(static_cast<unsigned>(queryTokens),
                                 static_cast<unsigned>(queryHeads)), 1U, 0,
                              static_cast<cudaStream_t>(stream)>>>(
        query, key, value, scores, probabilities, context,
        queryTokens, keyTokens, queryHeads, keyValueHeads, headDimension,
        queryPositionOffset, keyPositionOffset, causal);
    checkLaunch("CUDA causal attention kernel");
}

void gatherRows(const float* input, float* output, const std::uint32_t* rows,
                std::size_t rowCount, std::size_t width, StreamHandle stream) {
    gatherRowsKernel<<<blocks(rowCount * width), threadsPerBlock, 0,
                       static_cast<cudaStream_t>(stream)>>>(
        input, output, rows, rowCount, width);
    checkLaunch("CUDA gather kernel");
}

void scatterAddRows(const float* input, float* output,
                    const std::uint32_t* rows, const float* weights,
                    std::size_t rowCount, std::size_t width,
                    StreamHandle stream) {
    scatterAddRowsKernel<<<blocks(rowCount * width), threadsPerBlock, 0,
                          static_cast<cudaStream_t>(stream)>>>(
        input, output, rows, weights, rowCount, width);
    checkLaunch("CUDA scatter kernel");
}

void zero(float* output, std::size_t elements, StreamHandle stream) {
    zeroKernel<<<blocks(elements), threadsPerBlock, 0,
                 static_cast<cudaStream_t>(stream)>>>(output, elements);
    checkLaunch("CUDA zero kernel");
}

} // namespace hypermoe::backend::cuda::kernels
