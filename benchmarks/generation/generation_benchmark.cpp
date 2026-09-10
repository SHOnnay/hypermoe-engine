#include "generation/Generator.hpp"
#include "runtime/cache/KVCacheManager.hpp"
#include "runtime/generation/GenerationModel.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tokenizer/qwen/QwenTokenizerAdapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class BenchmarkModel final
    : public hypermoe::runtime::generation::GenerationModel {
public:
    BenchmarkModel()
        : backend_(std::make_shared<hypermoe::tensor::CpuTensorBackend>()) {}

    [[nodiscard]] std::size_t vocabularySize() const noexcept override { return 32; }
    [[nodiscard]] std::size_t hiddenDimension() const noexcept override { return 8; }
    [[nodiscard]] std::size_t layerCount() const noexcept override { return 2; }
    [[nodiscard]] std::size_t keyValueHeads() const noexcept override { return 2; }
    [[nodiscard]] std::size_t headDimension() const noexcept override { return 2; }
    [[nodiscard]] hypermoe::tensor::Device device() const noexcept override {
        return hypermoe::tensor::Device::cpu();
    }

    [[nodiscard]] hypermoe::runtime::generation::ForwardPass forward(
        hypermoe::runtime::InferenceContext& context,
        std::span<const std::uint32_t> tokenIds,
        hypermoe::runtime::cache::KVCacheBase& cache) override {
        auto hidden = backend_->allocateTensor(
            {tokenIds.size(), hiddenDimension()}, hypermoe::tensor::DType::FP32);
        auto logits = backend_->allocateTensor(
            {tokenIds.size(), vocabularySize()}, hypermoe::tensor::DType::FP32);
        auto* hiddenData = static_cast<float*>(hidden.data());
        auto* logitData = static_cast<float*>(logits.data());
        for (std::size_t token = 0; token < tokenIds.size(); ++token) {
            for (std::size_t feature = 0; feature < hiddenDimension(); ++feature) {
                hiddenData[token * hiddenDimension() + feature] =
                    static_cast<float>(tokenIds[token]) / 32.0F;
            }
            for (std::size_t id = 0; id < vocabularySize(); ++id) {
                logitData[token * vocabularySize() + id] = -12.0F;
            }
            logitData[token * vocabularySize() +
                      (tokenIds[token] + 1U) % vocabularySize()] = 12.0F;
        }
        for (std::size_t layer = 0; layer < layerCount(); ++layer) {
            auto keys = backend_->allocateTensor(
                {tokenIds.size(), keyValueHeads(), headDimension()},
                hypermoe::tensor::DType::FP32);
            auto values = backend_->allocateTensor(keys.shape(),
                                                    hypermoe::tensor::DType::FP32);
            cache.append(layer, context.sequencePosition, keys.view(), values.view());
        }
        return {std::move(hidden), std::move(logits), 1,
                std::vector<hypermoe::router::RouterDecision>(
                    tokenIds.size(), {1, {0}, {1.0F}})};
    }

private:
    std::shared_ptr<hypermoe::tensor::CpuTensorBackend> backend_;
};

double milliseconds(std::chrono::nanoseconds value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace hypermoe;
        constexpr std::size_t iterations = 200;
        constexpr std::size_t maximumSequence = 64;
        const std::string prompt = "HyperMoE";
        auto tokenizer =
            std::make_shared<tokenizer::qwen::QwenTokenizerAdapter>(
                32,
                [](std::string_view text) {
                    std::vector<std::uint32_t> tokens;
                    tokens.reserve(text.size());
                    for (const char character : text) {
                        const auto byte = static_cast<unsigned char>(character);
                        tokens.push_back(byte % 32U);
                    }
                    return tokens;
                },
                [](std::span<const std::uint32_t> tokens) {
                    std::string text;
                    text.reserve(tokens.size());
                    for (const auto token : tokens) {
                        text.push_back(static_cast<char>('A' + token % 26U));
                    }
                    return text;
                });
        auto model = std::make_shared<BenchmarkModel>();
        runtime::cache::KVCache sizing(
            model->layerCount(), maximumSequence, model->keyValueHeads(),
            model->headDimension());
        auto manager = std::make_shared<runtime::cache::KVCacheManager>(
            model->layerCount(), maximumSequence, model->keyValueHeads(),
            model->headDimension(), sizing.maximumMemoryUsageBytes() * 2U);
        generation::Generator generator(tokenizer, model, manager);
        generation::GenerationConfig config;
        config.maximumNewTokens = 16;

        generation::GenerationMetrics total;
        generation::GenerationResult last;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            last = generator.generate(prompt, config);
            total.tokenization += last.metrics.tokenization;
            total.prefill += last.metrics.prefill;
            total.decode += last.metrics.decode;
            total.total += last.metrics.total;
            total.decodeSteps += last.metrics.decodeSteps;
            total.peakKVCacheBytes =
                std::max(total.peakKVCacheBytes, last.metrics.peakKVCacheBytes);
        }
        const auto divisor = static_cast<double>(iterations);
        const auto decodeSeconds = std::chrono::duration<double>(total.decode).count();
        const auto throughput = decodeSeconds > 0.0
            ? static_cast<double>(total.decodeSteps) / decodeSeconds : 0.0;
        const auto perTokenPerLayer = sizeof(std::uint64_t) +
            2U * model->keyValueHeads() * model->headDimension() * sizeof(float);
        const auto prefillBytes = prompt.size() * model->layerCount() *
            perTokenPerLayer;
        const auto reportPath = std::filesystem::path(
            argc > 1 ? argv[1] : "generation_report.json.report");
        std::ofstream report(reportPath);
        if (!report) throw std::runtime_error("cannot create generation report");
        report << std::fixed << std::setprecision(6)
               << "{\n  \"benchmark\": \"phase15_16_generation\",\n"
               << "  \"fixture\": \"deterministic_cpu_incremental\",\n"
               << "  \"iterations\": " << iterations << ",\n"
               << "  \"prompt_tokens\": " << prompt.size() << ",\n"
               << "  \"generated_tokens\": " << last.generatedTokens.size() << ",\n"
               << "  \"average_tokenization_ms\": "
               << milliseconds(total.tokenization) / divisor << ",\n"
               << "  \"average_prefill_ms\": "
               << milliseconds(total.prefill) / divisor << ",\n"
               << "  \"average_decode_ms\": "
               << milliseconds(total.decode) / divisor << ",\n"
               << "  \"average_total_ms\": "
               << milliseconds(total.total) / divisor << ",\n"
               << "  \"decode_tokens_per_second\": " << throughput << ",\n"
               << "  \"kv_cache_prefill_bytes\": " << prefillBytes << ",\n"
               << "  \"kv_cache_peak_bytes\": " << total.peakKVCacheBytes << ",\n"
               << "  \"kv_cache_reserved_bytes_per_session\": "
               << manager->bytesPerSession() << "\n}\n";
        report.close();
        std::cout << "Generation benchmark wrote " << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Generation benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
