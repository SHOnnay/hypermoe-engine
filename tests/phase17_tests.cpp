#include "experts/ExpertExecutor.hpp"
#include "generation/Generator.hpp"
#include "runtime/cache/CudaKVCache.hpp"
#include "runtime/cache/KVCacheManager.hpp"
#include "runtime/generation/Decoder.hpp"
#include "runtime/generation/GenerationModel.hpp"
#include "runtime/generation/InferenceSession.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "tokenizer/Tokenizer.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "transformer/attention/CudaAttention.hpp"
#include "transformer/norm/RMSNorm.hpp"
#include "validation/CorrectnessOracle.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

int failures{};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

hypermoe::tensor::Tensor hostTensor(
    hypermoe::tensor::CpuTensorBackend& cpu,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto result = cpu.allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (result.bytes() != values.size_bytes()) {
        throw std::invalid_argument("phase 17 fixture shape mismatch");
    }
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

hypermoe::tensor::Tensor deviceTensor(
    hypermoe::tensor::CudaTensorBackend& cuda,
    hypermoe::tensor::TensorView host) {
    auto result = cuda.allocateTensor(host.shape(), host.dtype());
    cuda.copyTensor(host, result.view());
    return result;
}

std::vector<float> hostValues(
    hypermoe::tensor::CudaTensorBackend& cuda,
    hypermoe::tensor::TensorView value) {
    hypermoe::tensor::CpuTensorBackend cpu;
    auto host = cpu.allocateTensor(value.shape(), value.dtype());
    cuda.copyTensor(value, host.view());
    const auto* values = static_cast<const float*>(host.data());
    return {values, values + host.shape().elementCount()};
}

bool matches(std::span<const float> actual, std::span<const float> expected) {
    return hypermoe::validation::CorrectnessOracle::compare(
        actual, expected,
        hypermoe::validation::CorrectnessOracle::toleranceFor(
            hypermoe::tensor::DType::FP32)).matches;
}

class CudaSessionModel final : public hypermoe::runtime::generation::GenerationModel {
public:
    explicit CudaSessionModel(
        std::shared_ptr<hypermoe::tensor::CudaTensorBackend> backend)
        : backend_(std::move(backend)) {}

    [[nodiscard]] std::size_t vocabularySize() const noexcept override { return 3; }
    [[nodiscard]] std::size_t hiddenDimension() const noexcept override { return 2; }
    [[nodiscard]] std::size_t layerCount() const noexcept override { return 1; }
    [[nodiscard]] std::size_t keyValueHeads() const noexcept override { return 1; }
    [[nodiscard]] std::size_t headDimension() const noexcept override { return 2; }
    [[nodiscard]] hypermoe::tensor::Device device() const noexcept override {
        return backend_->device();
    }
    [[nodiscard]] hypermoe::tensor::Tensor materializeHost(
        hypermoe::tensor::TensorView value) const override {
        hypermoe::tensor::CpuTensorBackend cpu;
        auto result = cpu.allocateTensor(value.shape(), value.dtype());
        backend_->copyTensor(value, result.view());
        return result;
    }

    [[nodiscard]] hypermoe::runtime::generation::ForwardPass forward(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        hypermoe::runtime::cache::KVCacheBase& cache) override {
        hypermoe::tensor::CpuTensorBackend cpu;
        std::vector<float> hiddenValues(tokenIds.size() * 2, 0.25F);
        std::vector<float> logitValues(tokenIds.size() * 3, 0.0F);
        for (std::size_t token = 0; token < tokenIds.size(); ++token) {
            logitValues[token * 3 + 2] = 1.0F;
        }
        std::vector<float> cacheValues(tokenIds.size() * 2, 0.5F);
        auto hiddenHost = hostTensor(cpu, {tokenIds.size(), 2}, hiddenValues);
        auto logitsHost = hostTensor(cpu, {tokenIds.size(), 3}, logitValues);
        auto cacheHost = hostTensor(cpu, {tokenIds.size(), 1, 2}, cacheValues);
        auto hidden = deviceTensor(*backend_, hiddenHost.view());
        auto logits = deviceTensor(*backend_, logitsHost.view());
        cache.append(0, context.sequencePosition, cacheHost.view(), cacheHost.view());
        return {std::move(hidden), std::move(logits), 0, {}};
    }

private:
    std::shared_ptr<hypermoe::tensor::CudaTensorBackend> backend_;
};

class FixtureTokenizer final : public hypermoe::tokenizer::Tokenizer {
public:
    [[nodiscard]] std::vector<hypermoe::tokenizer::TokenId> encode(
        std::string_view text) const override {
        return text.empty()
            ? std::vector<hypermoe::tokenizer::TokenId>{}
            : std::vector<hypermoe::tokenizer::TokenId>{1};
    }
    [[nodiscard]] std::string decode(
        std::span<const hypermoe::tokenizer::TokenId> tokens) const override {
        return std::string(tokens.size(), 'x');
    }
    [[nodiscard]] std::size_t vocabularySize() const noexcept override { return 3; }
};

void runCudaTests() {
    using namespace hypermoe;
    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    if (!cuda->available()) {
        std::cout << "SKIP: CUDA runtime unavailable; CPU fallback remains testable\n";
        return;
    }
    tensor::CpuTensorBackend cpu;
    auto leftHost = hostTensor(cpu, {2, 2}, std::vector<float>{1, 2, 3, 4});
    auto rightHost = hostTensor(cpu, {2, 2}, std::vector<float>{5, 6, 7, 8});
    auto left = deviceTensor(*cuda, leftHost.view());
    auto right = deviceTensor(*cuda, rightHost.view());
    auto result = cuda->allocateTensor({2, 2}, tensor::DType::FP32);

    cuda->matmul(left.view(), right.view(), result.view());
    expect(matches(hostValues(*cuda, result.view()),
                   std::vector<float>{19, 22, 43, 50}),
           "CUDA cuBLAS GEMM matches the CPU oracle");
    cuda->add(left.view(), right.view(), result.view());
    expect(matches(hostValues(*cuda, result.view()),
                   std::vector<float>{6, 8, 10, 12}),
           "CUDA residual addition matches the CPU oracle");
    cuda->mul(left.view(), right.view(), result.view());
    expect(matches(hostValues(*cuda, result.view()),
                   std::vector<float>{5, 12, 21, 32}),
           "CUDA elementwise multiply matches the CPU oracle");
    tensor::activation::apply(tensor::activation::ActivationType::SiLU,
                              *cuda, left.view(), result.view());
    const auto activated = hostValues(*cuda, result.view());
    expect(activated.size() == 4 && activated.front() > 0.73F,
           "CUDA activation dispatch produces device output");

    auto gateHost = hostTensor(cpu, {2, 2}, std::vector<float>{1, 0, 0, 1});
    auto gate = deviceTensor(*cuda, gateHost.view());
    auto expertOutput = cuda->allocateTensor({2, 2}, tensor::DType::FP32);
    ExpertMlpExecutor executor(cuda);
    executor.execute(left.view(), {gate.view(), gate.view(), gate.view()},
                     expertOutput.view());
    expect(hostValues(*cuda, expertOutput.view()).size() == 4,
           "expert MLP retains projection and output tensors on CUDA");

    auto normWeightsHost = hostTensor(cpu, {2}, std::vector<float>{1, 2});
    auto normWeights = deviceTensor(*cuda, normWeightsHost.view());
    transformer::norm::RMSNorm normalization(cuda, 2);
    auto normalized = normalization.execute(left.view(), normWeights.view());
    expect(hostValues(*cuda, normalized.view()).size() == 4,
           "CUDA RMSNorm path returns device-resident output");

    transformer::attention::CpuAttention cpuAttention(
        std::make_shared<tensor::CpuTensorBackend>());
    transformer::attention::CudaAttention cudaAttention(cuda);
    const transformer::attention::AttentionWeights cpuWeights{
        gateHost.view(), gateHost.view(), gateHost.view(), gateHost.view()};
    const transformer::attention::AttentionWeights cudaWeights{
        gate.view(), gate.view(), gate.view(), gate.view()};
    const transformer::attention::AttentionConfiguration config{1, 1, 2, true};
    auto cpuAttentionResult = cpuAttention.execute(leftHost.view(), cpuWeights, config);
    auto cudaAttentionResult = cudaAttention.execute(left.view(), cudaWeights, config);
    const auto* expected = static_cast<const float*>(cpuAttentionResult.output.data());
    expect(matches(hostValues(*cuda, cudaAttentionResult.output.view()),
                   {expected, cpuAttentionResult.output.shape().elementCount()}),
           "CUDA attention matches causal CPU reference output");

    runtime::cache::CudaKVCache cache(cuda, 1, 8, 1, 2);
    auto cacheInputHost = hostTensor(
        cpu, {2, 1, 2}, std::vector<float>{1, 2, 3, 4});
    auto cacheInput = deviceTensor(*cuda, cacheInputHost.view());
    cache.append(0, 0, cacheInput.view(), cacheInput.view());
    const auto snapshot = cache.snapshot(0);
    expect(cache.device() == tensor::Device::cuda() && snapshot.tokenCount() == 2 &&
               snapshot.keys == std::vector<float>({1, 2, 3, 4}),
           "CUDA KV cache owns device tensors and returns a correct snapshot");

    auto manager = std::make_shared<runtime::cache::KVCacheManager>(
        1, 8, 1, 2, 1024, cuda);
    auto model = std::make_shared<CudaSessionModel>(cuda);
    runtime::generation::InferenceSession session(
        model, manager, 2, {}, {tensor::Device::cuda()});
    runtime::generation::Decoder decoder;
    const std::vector<std::uint32_t> prompt{1, 2};
    decoder.prefill(session, prompt);
    expect(session.kvCache().device() == tensor::Device::cuda() &&
               session.forwardState().logits().device() == tensor::Device::cuda(),
           "inference session selects CUDA model state and CUDA KV ownership");

    auto generationManager = std::make_shared<runtime::cache::KVCacheManager>(
        1, 8, 1, 2, 1024, cuda);
    generation::Generator generator(
        std::make_shared<FixtureTokenizer>(), model, generationManager);
    generation::GenerationConfig generationConfig;
    generationConfig.maximumNewTokens = 2;
    generationConfig.inference.device = tensor::Device::cuda();
    const auto generated = generator.generate("prompt", generationConfig);
    expect(generated.generatedTokens == std::vector<std::uint32_t>({2, 2}) &&
               generated.text == "xx",
           "generation API samples device logits with CUDA session state");
}

} // namespace

int main() {
    try {
        runCudaTests();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures == 0) std::cout << "Phase 17 tests passed\n";
    return failures == 0 ? 0 : 1;
}
