#include "router/CpuRouterBackend.hpp"
#include "router/CudaRouterBackend.hpp"
#include "tensor/activation/Activation.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"
#include "transformer/position/RoPE.hpp"
#include "validation/RealModelValidation.hpp"

#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures{};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-phase18-19-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
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

void writeQwenFixture(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Phase19-Fixture","num_hidden_layers":1,"num_experts":2,"hidden_size":4,"moe_intermediate_size":4,"num_experts_per_tok":1,"norm_topk_prob":true,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0.000001,"rope_theta":10000,"vocab_size":6,"tie_word_embeddings":true})";
    const auto matrix = identity(4);
    const std::vector<float> keyValue{1,0,0,0, 0,1,0,0};
    const std::vector<float> norm(4, 1.0F);
    std::vector<TensorDefinition> tensors{
        {"model.embed_tokens.weight", {6,4}, std::vector<float>(24, 0.25F)},
        {"model.norm.weight", {4}, norm},
        {"model.layers.0.self_attn.q_proj.weight", {4,4}, matrix},
        {"model.layers.0.self_attn.k_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.v_proj.weight", {2,4}, keyValue},
        {"model.layers.0.self_attn.o_proj.weight", {4,4}, matrix},
        {"model.layers.0.input_layernorm.weight", {4}, norm},
        {"model.layers.0.post_attention_layernorm.weight", {4}, norm},
        {"model.layers.0.mlp.gate.weight", {2,4},
         std::vector<float>{2,0,0,0, 0,2,0,0}}};
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
    if (!output) throw std::runtime_error("failed writing Phase 19 fixture");
}

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

hypermoe::tensor::Tensor hostTensor(
    hypermoe::tensor::CpuTensorBackend& cpu,
    const hypermoe::tensor::Shape& shape,
    std::span<const float> values) {
    auto result = cpu.allocateTensor(shape, hypermoe::tensor::DType::FP32);
    if (result.bytes() != values.size_bytes()) {
        throw std::invalid_argument("Phase 18/19 fixture shape mismatch");
    }
    std::memcpy(result.data(), values.data(), values.size_bytes());
    return result;
}

hypermoe::tensor::Tensor toDevice(
    hypermoe::tensor::CudaTensorBackend& cuda,
    hypermoe::tensor::TensorView source) {
    auto result = cuda.allocateTensor(source.shape(), source.dtype());
    cuda.copyTensor(source, result.view());
    return result;
}

std::vector<float> toHost(hypermoe::tensor::CudaTensorBackend& cuda,
                          hypermoe::tensor::TensorView source) {
    hypermoe::tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(source.shape(), source.dtype());
    cuda.copyTensor(source, result.view());
    const auto* values = static_cast<const float*>(result.data());
    return {values, values + result.shape().elementCount()};
}

bool close(std::span<const float> actual, std::span<const float> expected) {
    return hypermoe::validation::CorrectnessOracle::compare(
        actual, expected, {1.0e-5F, 1.0e-5F}).matches;
}

void testRealModelComparison() {
    using namespace hypermoe;
    validation::RealModelTrace cpu{
        {1, 2, 3}, {{1, 2}, {3, 4}}, {{0.5F, 1.0F}}};
    const auto same = validation::RealModelValidator::compare(cpu, cpu);
    expect(same.matches() && same.toJson().find("\"matches\":true") !=
               std::string::npos,
           "real-model validator compares logits, transformer, and expert traces");
    auto different = cpu;
    different.logits[1] += 0.1F;
    expect(!validation::RealModelValidator::compare(cpu, different).matches(),
           "real-model validator detects divergent CUDA logits");
    expectThrows([] {
        (void)validation::RealModelValidator::prepareQwen(
            std::filesystem::path{"missing-phase19-checkpoint"},
            std::filesystem::path{"missing-phase19-output"});
    }, "real-model preparation rejects a missing checkpoint cleanly");

    TemporaryDirectory temporary;
    const auto artifact = temporary.path() / "artifact";
    const auto packed = temporary.path() / "packed";
    std::filesystem::create_directories(artifact);
    writeQwenFixture(artifact);
    const auto prepared = validation::RealModelValidator::prepareQwen(
        artifact, packed);
    expect(prepared.checkpoint.shardCount == 1 &&
               prepared.checkpoint.expertCount == 2 &&
               prepared.packing.validationPassed &&
               prepared.manifest.runtimeArchitecture &&
               prepared.manifest.modelIO && prepared.manifest.layers.size() == 1 &&
               prepared.packing.experts == 2,
           "real-model preparation imports, validates, and packs a complete artifact");
}

void testNativeCudaDataflow() {
    using namespace hypermoe;
    auto cuda = std::make_shared<tensor::CudaTensorBackend>();
    if (!cuda->available() || !cuda->nativeKernelsAvailable()) {
        std::cout << "SKIP: native CUDA kernels unavailable\n";
        return;
    }
    tensor::CpuTensorBackend cpu;
    const std::vector<float> inputValues{-2, -1, 0, 1, 2, 3};
    auto hostInput = hostTensor(cpu, {3, 2}, inputValues);
    auto input = toDevice(*cuda, hostInput.view());
    auto activated = cuda->allocateTensor(input.shape(), tensor::DType::FP32);
    tensor::activation::apply(tensor::activation::ActivationType::SiLU,
                              *cuda, input.view(), activated.view());
    std::vector<float> expectedActivation;
    for (const auto value : inputValues) {
        expectedActivation.push_back(tensor::activation::silu(value));
    }
    expect(close(toHost(*cuda, activated.view()), expectedActivation),
           "native CUDA SiLU matches the CPU reference");

    const std::vector<float> ropeValues{1,2,3,4, 5,6,7,8};
    auto hostRope = hostTensor(cpu, {2, 4}, ropeValues);
    auto deviceRope = toDevice(*cuda, hostRope.view());
    cuda->applyRoPE(deviceRope.view(), 2, 2, 2, 3, 10000.0F);
    auto expectedRope = ropeValues;
    transformer::position::RoPE{10000.0F}.apply(expectedRope, 2, 2, 2, 3);
    expect(close(toHost(*cuda, deviceRope.view()), expectedRope),
           "native CUDA RoPE matches the CPU reference");

    auto hostRouter = hostTensor(
        cpu, {2, 3}, std::vector<float>{2,0,1, 0,2,1});
    auto routerWeights = toDevice(*cuda, hostRouter.view());
    const router::RouterConfig config{
        3, 2, router::RoutingNormalization::Softmax, true};
    router::CpuRouterBackend cpuRouter;
    router::CudaRouterBackend cudaRouter(cuda);
    const auto cpuDecision = cpuRouter.routeBatch(
        0, hostInput.view(), hostRouter.view(), config);
    const auto cudaDecision = cudaRouter.routeBatch(
        0, input.view(), routerWeights.view(), config);
    expect(cpuDecision.tokens.size() == cudaDecision.tokens.size() &&
               cpuDecision.tokens.front().selectedExpertIds ==
                   cudaDecision.tokens.front().selectedExpertIds &&
               close(cpuDecision.tokens.front().routingScores,
                     cudaDecision.tokens.front().routingScores),
           "native CUDA router top-k matches the CPU router");

    const std::vector<std::size_t> rows{2, 0};
    auto gathered = cuda->allocateTensor({2, 2}, tensor::DType::FP32);
    cuda->gatherRows(input.view(), rows, gathered.view());
    expect(close(toHost(*cuda, gathered.view()),
                 std::vector<float>{2,3,-2,-1}),
           "native CUDA gather preserves expert token order");
    auto scattered = cuda->allocateTensor({3, 2}, tensor::DType::FP32);
    cuda->zero(scattered.view());
    const std::vector<float> routingWeights{0.5F, 2.0F};
    cuda->scatterAddRows(gathered.view(), rows, routingWeights, scattered.view());
    expect(close(toHost(*cuda, scattered.view()),
                 std::vector<float>{-4,-2,0,0,1,1.5F}),
           "native CUDA scatter applies routing weights on device");
}

} // namespace

int main() {
    try {
        testRealModelComparison();
        testNativeCudaDataflow();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures == 0) std::cout << "Phase 18/19 tests passed\n";
    return failures == 0 ? 0 : 1;
}
