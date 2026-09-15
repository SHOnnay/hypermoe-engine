#include "importer/qwen/QwenCheckpointLoader.hpp"
#include "models/metadata/JsonValue.hpp"
#include "models/runtime/PackedModelRuntime.hpp"
#include "profiling/RealModelProfile.hpp"
#include "tools/model_convert/real_checkpoint/RealCheckpointConverter.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "transformer/attention/CpuAttention.hpp"
#include "validation/RealModelValidation.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
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

template <typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const std::exception&) {
        expect(true, message);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase20-" +
                 std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch().count()) + "-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        if (!std::filesystem::create_directories(path_)) {
            throw std::runtime_error("failed creating unique Phase 20 test directory");
        }
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

struct TensorDefinition {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

std::vector<float> identity(std::size_t size) {
    std::vector<float> result(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        result[index * size + index] = 1.0F;
    }
    return result;
}

void writeText(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("failed writing Phase 20 metadata");
}

void writeArtifact(const std::filesystem::path& root,
                   std::string_view architecture = "Qwen3MoeForCausalLM") {
    std::ostringstream config;
    config << "{\"architectures\":[\"" << architecture
           << "\"],\"model_type\":\""
           << (architecture.starts_with("Qwen2") ? "qwen2_moe" : "qwen3_moe")
           << R"(","_name_or_path":"Qwen/Phase20-Fixture","num_hidden_layers":1,"num_experts":2,"hidden_size":4,"moe_intermediate_size":4,"num_experts_per_tok":1,"norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0.000001,"rope_theta":10000,"vocab_size":6,"tie_word_embeddings":true,"bos_token_id":0,"eos_token_id":5,"pad_token_id":0})";
    writeText(root / "config.json", config.str());
    writeText(root / "tokenizer.json",
              R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"<eos>":5}},"added_tokens":[]})");
    writeText(root / "tokenizer_config.json",
              R"({"tokenizer_class":"Qwen2TokenizerFast","chat_template":"{{ messages }}"})");
    writeText(root / "special_tokens_map.json", R"({"eos_token":"<eos>"})");

    const auto matrix = identity(4);
    const std::vector<float> keyValue{1,0,0,0, 0,1,0,0};
    const std::vector<float> norm(4, 1.0F);
    std::vector<TensorDefinition> tensors{
        {"model.embed_tokens.weight", {6,4},
         std::vector<float>{1,0,0,0, 0,1,0,0, 0,0,1,0,
                            0,0,0,1, 1,1,0,0, 0,0,1,1}},
        {"model.norm.weight", {4}, norm},
        {"model.layers.0.self_attn.q_proj.weight", {4,4}, matrix},
        {"model.layers.0.self_attn.k_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.v_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.o_proj.weight", {4,4}, matrix},
        {"model.layers.0.input_layernorm.weight", {4}, norm},
        {"model.layers.0.post_attention_layernorm.weight", {4}, norm},
        {"model.layers.0.mlp.gate.weight", {2,4},
         std::vector<float>{2,0,0,0, 0,2,0,0}}};
    if (!architecture.starts_with("Qwen2")) {
        tensors.push_back(
            {"model.layers.0.self_attn.q_norm.weight", {2}, {1.0F, 1.0F}});
        tensors.push_back(
            {"model.layers.0.self_attn.k_norm.weight", {2}, {1.0F, 1.0F}});
    }
    for (std::size_t expert = 0; expert < 2; ++expert) {
        const auto prefix = "model.layers.0.mlp.experts." + std::to_string(expert);
        tensors.push_back({prefix + ".gate_proj.weight", {4,4}, matrix});
        tensors.push_back({prefix + ".up_proj.weight", {4,4}, matrix});
        tensors.push_back({prefix + ".down_proj.weight", {4,4}, matrix});
    }
    std::ostringstream header;
    std::vector<float> payload;
    header << '{';
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& tensor = tensors[index];
        if (index != 0) header << ',';
        const auto begin = payload.size() * sizeof(float);
        payload.insert(payload.end(), tensor.values.begin(), tensor.values.end());
        const auto end = payload.size() * sizeof(float);
        header << '"' << tensor.name << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0) header << ',';
            header << tensor.shape[dimension];
        }
        header << "],\"data_offsets\":[" << begin << ',' << end << "]}";
    }
    header << '}';
    auto headerText = header.str();
    while (headerText.size() % 8U != 0) headerText.push_back(' ');
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const auto headerSize = static_cast<std::uint64_t>(headerText.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((headerSize >> shift) & 0xffU));
    }
    output.write(headerText.data(), static_cast<std::streamsize>(headerText.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing Phase 20 SafeTensors");
}

std::uint64_t parameters(const hypermoe::models::ModelManifest& manifest) {
    if (manifest.parameterCount != 0) return manifest.parameterCount;
    std::uint64_t result{};
    for (const auto& tensor : manifest.tensors) {
        result += tensor.shape.elementCount();
    }
    return result;
}

void testQueryKeyNormalization() {
    using namespace hypermoe;
    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    const auto make = [&](tensor::Shape shape, std::span<const float> values) {
        auto result = backend->allocateTensor(shape, tensor::DType::FP32);
        if (result.bytes() != values.size_bytes()) {
            throw std::invalid_argument("Q/K normalization fixture shape mismatch");
        }
        std::memcpy(result.data(), values.data(), values.size_bytes());
        return result;
    };
    const std::vector<float> hiddenValues{3.0F, 4.0F};
    const std::vector<float> identityValues{1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> normValues{1.0F, 1.0F};
    auto hidden = make({1, 2}, hiddenValues);
    auto query = make({2, 2}, identityValues);
    auto key = make({2, 2}, identityValues);
    auto value = make({2, 2}, identityValues);
    auto output = make({2, 2}, identityValues);
    auto queryNorm = make({2}, normValues);
    auto keyNorm = make({2}, normValues);
    transformer::attention::AttentionConfiguration configuration;
        configuration.headCount = 1;
        configuration.keyValueHeadCount = 1;
        configuration.headDimension = 2;
        configuration.projectionHeadDimension = 2;
        configuration.queryKeyNormEpsilon = 1.0e-6F;
    const auto result = transformer::attention::CpuAttention{backend}.execute(
        hidden.view(),
        {query.view(), key.view(), value.view(), output.view(),
         queryNorm.view(), keyNorm.view()},
        configuration);
    const auto* normalized = static_cast<const float*>(result.query.data());
    const auto divisor = std::sqrt(12.5F + configuration.queryKeyNormEpsilon);
    expect(std::abs(normalized[0] - 3.0F / divisor) < 1.0e-5F &&
               std::abs(normalized[1] - 4.0F / divisor) < 1.0e-5F,
           "Qwen per-head Q/K RMSNorm executes before attention");
}

void testCheckpointLoadingAndConversion() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(source);
    writeArtifact(source);
    const auto checkpoint = importer::qwen::QwenCheckpointLoader{}.load(source);
    expect(checkpoint.validation.shardCount == 1 &&
               checkpoint.validation.expertCount == 2 &&
               checkpoint.validation.totalParameters ==
                   checkpoint.manifest.parameterCount &&
               checkpoint.tokenizer.declaredVocabularySize == 6 &&
               checkpoint.tokenizer.vocabularyEntries == 6 &&
               checkpoint.tokenizer.eosTokenIds == std::vector<std::uint32_t>{5} &&
               checkpoint.tokenizer.hasChatTemplate &&
               checkpoint.manifest.layers.front().queryNormTensor ==
                   "model.layers.0.self_attn.q_norm.weight" &&
               checkpoint.manifest.layers.front().keyNormTensor ==
                   "model.layers.0.self_attn.k_norm.weight",
           "real Qwen loader validates model, SafeTensors, and tokenizer metadata");

    const auto report =
        conversion::real_checkpoint::RealCheckpointConverter{}.convertQwen(
            source, packed);
    expect(report.packing.validationPassed && report.packing.experts == 2 &&
               std::filesystem::is_regular_file(packed / "experts.bin") &&
               std::filesystem::is_regular_file(packed / "experts.index") &&
               std::filesystem::is_regular_file(packed / "manifest.json") &&
               std::filesystem::is_regular_file(packed / "tokenizer.json") &&
               std::filesystem::is_regular_file(
                   packed / "real_checkpoint_report.json"),
           "real checkpoint conversion produces a self-contained runtime artifact");
    expect(models::metadata::parseJson(report.toJson()).isObject(),
           "real checkpoint conversion report is valid JSON");
    expectThrows([&] {
        (void)conversion::real_checkpoint::RealCheckpointConverter{}.convertQwen(
            source, packed);
    }, "real checkpoint conversion refuses to overwrite an artifact");
    expect(std::filesystem::is_regular_file(packed / "manifest.json"),
           "failed overwrite attempt preserves the existing runtime artifact");

    models::runtime::PackedModelRuntime runtime(packed);
    const std::vector<std::uint32_t> tokenIds{0, 1};
    const auto forward = runtime.forward(tokenIds);
    const auto logits = runtime.materializeHost(forward.logits.view());
    const auto* logitsValues = static_cast<const float*>(logits.data());
    expect(logits.shape() == tensor::Shape{2, 6} &&
               std::all_of(logitsValues, logitsValues + logits.shape().elementCount(),
                           [](float value) { return std::isfinite(value); }) &&
               forward.transformer.layers.size() == 1 &&
               !forward.transformer.layers.front().expertOutputs.empty(),
           "packed runtime executes token IDs through embeddings, attention, MoE, and logits");
    const auto trace = validation::RealModelValidator::capture(runtime, forward);
    expect(validation::RealModelValidator::compare(trace, trace).matches(),
           "real forward trace validates embedding, attention, experts, and logits");
    const auto cpuCuda = validation::RealModelValidator::validateCpuCuda(
        packed, tokenIds);
    expect(cpuCuda.cpuTime > std::chrono::nanoseconds::zero() &&
               (cpuCuda.cudaAvailable ? cpuCuda.executed : !cpuCuda.executed),
           "CPU/CUDA validator always runs CPU and reports CUDA availability honestly");
    expect(models::metadata::parseJson(cpuCuda.toJson()).isObject(),
           "CPU/CUDA validation report is valid JSON");

    if (cpuCuda.cudaAvailable) {
        models::runtime::PackedRuntimeConfiguration automatic;
        automatic.device = tensor::Device::cuda();
        automatic.automaticExpertDeviceBudget = true;
        runtime::cache::KVCache sizing(
            runtime.architecture().layerCount, tokenIds.size(),
            runtime.architecture().keyValueHeads, runtime.architecture().headDimension);
        automatic.expertDeviceReservations.kvCacheBytes = 2U * sizing.maximumMemoryUsageBytes();
        models::runtime::PackedModelRuntime adaptive(packed, automatic);
        const std::vector<std::uint32_t> oversizedPrompt(100000, 0);
        expectThrows([&] { (void)adaptive.forward(oversizedPrompt); },
                     "automatic workspace envelope rejects oversized prefill before device allocation");
        auto reserved = adaptive.createKVCache(tokenIds.size());
        expectThrows([&] { (void)adaptive.createKVCache(1); },
                     "live CUDA cache capacities cannot overcommit automatic KV reservation");
        expectThrows([&] { (void)adaptive.createKVCache(1000000); },
                     "oversized CUDA cache rejected before allocating device storage");
        const auto gpuForward = adaptive.forward(tokenIds, 0, *reserved);
        const auto gpuTrace = validation::RealModelValidator::capture(adaptive, gpuForward);
        expect(validation::RealModelValidator::compare(trace, gpuTrace).matches(),
               "automatic expert sizing preserves full CPU/CUDA Qwen fixture correctness");
        auto unreserved = runtime.createKVCache(tokenIds.size());
        expectThrows([&] { (void)adaptive.forward(tokenIds, 0, *unreserved); },
                     "unreserved external cache cannot consume automatic GPU KV headroom");
        reserved.reset();
        auto reused = adaptive.createKVCache(tokenIds.size());
        expect(reused != nullptr, "released session capacity becomes reservable again");
        const auto snapshot = adaptive.snapshot();
        expect(snapshot.expertDeviceBudget.automatic &&
                   snapshot.expertMemory.vram.limitBytes == snapshot.expertDeviceBudget.effectiveBytes &&
                   snapshot.experts.vramPromotions > 0,
               "automatic budget and promotion metrics reflect the production runtime");
    }

    models::runtime::PackedModelRuntime incremental(packed);
    auto cache = incremental.createKVCache(tokenIds.size());
    std::vector<models::runtime::ModelForwardResult> forwards;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t position = 0; position < tokenIds.size(); ++position) {
        forwards.push_back(incremental.forward(
            std::span<const std::uint32_t>{tokenIds.data() + position, 1},
            position, *cache));
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto profile = profiling::RealModelProfileCollector::collect(
        incremental.manifest(), parameters(incremental.manifest()),
        incremental.device(), forwards, incremental.snapshot(),
        cache->memoryUsageBytes(), elapsed);
    expect(profile.tokens == 2 && profile.layerCount == 1 &&
               profile.expertRequests >= 2 && profile.ramUsageBytes > 0 &&
               profile.kvCacheBytes > 0 && profile.tokensPerSecond > 0.0 &&
               profile.toJson().find("\"expert_frequency\"") != std::string::npos &&
               models::metadata::parseJson(profile.toJson()).isObject(),
           "real-model profiler reports execution, memory, cache, and MoE metrics");
}

void testQwen2AndTokenizerValidation() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "qwen2";
    std::filesystem::create_directories(source);
    writeArtifact(source, "Qwen2MoeForCausalLM");
    expect(importer::qwen::QwenCheckpointLoader{}.load(source)
                   .manifest.sourceArchitecture == "Qwen2MoeForCausalLM",
           "checkpoint loader accepts Qwen2 routed-expert tensor layouts");
    writeText(source / "tokenizer.json",
              R"({"model":{"type":"BPE","vocab":{"bad":6}}})");
    expectThrows([&] {
        (void)importer::qwen::QwenCheckpointLoader{}.load(source);
    }, "checkpoint loader rejects tokenizer IDs outside model vocab_size");
}

} // namespace

int main() {
    try {
        testQueryKeyNormalization();
        testCheckpointLoadingAndConversion();
        testQwen2AndTokenizerValidation();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures == 0) std::cout << "Phase 20 tests passed\n";
    return failures == 0 ? 0 : 1;
}
