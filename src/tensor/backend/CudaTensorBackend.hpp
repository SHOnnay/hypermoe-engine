#pragma once

#include "tensor/backend/TensorBackend.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hypermoe {
class Profiler;
}

namespace hypermoe::tensor {

class CudaTensorBackend final : public TensorBackend {
public:
    struct Impl;
    struct RoutingSelection {
        std::vector<std::uint32_t> expertIds;
        std::vector<float> scores;
    };

    explicit CudaTensorBackend(int device = 0,
                               std::shared_ptr<Profiler> profiler = {});
    ~CudaTensorBackend() override;

    CudaTensorBackend(const CudaTensorBackend&) = delete;
    CudaTensorBackend& operator=(const CudaTensorBackend&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] Device device() const noexcept override;
    [[nodiscard]] bool available() const noexcept override;
    [[nodiscard]] Tensor allocateTensor(const Shape& shape, DType dtype) override;
    void copyTensor(TensorView source, TensorView destination) override;
    void matmul(TensorView left,
                TensorView right,
                TensorView output) override;
    void add(TensorView left,
             TensorView right,
             TensorView output) override;
    void mul(TensorView left,
             TensorView right,
             TensorView output) override;
    void rmsNorm(TensorView input,
                 TensorView weight,
                 TensorView output,
                 float epsilon);
    [[nodiscard]] bool nativeKernelsAvailable() const noexcept;
    [[nodiscard]] backend::BackendStats backendStats() const;
    void applyActivation(int type, TensorView input, TensorView output);
    void applyRoPE(TensorView values, std::size_t tokenCount,
                   std::size_t headCount, std::size_t headDimension,
                   std::size_t positionOffset, float theta);
    [[nodiscard]] RoutingSelection routeTopK(
        TensorView hiddenStates, TensorView routerWeights,
        std::size_t expertCount, std::size_t topK,
        bool softmax, bool renormalize);
    void causalAttention(TensorView query, TensorView key, TensorView value,
                         TensorView scores, TensorView probabilities,
                         TensorView context, std::size_t queryHeads,
                         std::size_t keyValueHeads, std::size_t headDimension,
                         std::uint64_t queryPositionOffset,
                         std::uint64_t keyPositionOffset, bool causal);
    void gatherRows(TensorView input, std::span<const std::size_t> rows,
                    TensorView output);
    void scatterAddRows(TensorView input, std::span<const std::size_t> rows,
                        std::span<const float> weights, TensorView output);
    void zero(TensorView output);
    [[nodiscard]] Tensor reshape(const Tensor& tensor, Shape shape) override;
    void synchronize() override;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace hypermoe::tensor
