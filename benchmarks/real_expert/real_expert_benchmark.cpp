#include "backend/Backend.hpp"
#include "backend/CpuBackend.hpp"
#include "experts/ExpertExecutor.hpp"
#include "importer/qwen/QwenImporter.hpp"
#include "models/ExpertWeightMap.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tools/model_convert/ExpertPacker.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
            ("hypermoe-real-expert-" + std::to_string(sequence.fetch_add(1)));
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

void append(std::vector<std::byte>& bytes, std::initializer_list<float> values) {
    const auto offset = bytes.size();
    bytes.resize(offset + values.size() * sizeof(float));
    std::memcpy(bytes.data() + offset, values.begin(), values.size() * sizeof(float));
}

void createFixture(const std::filesystem::path& root) {
    std::ofstream(root / "config.json")
        << R"({"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","_name_or_path":"Qwen/Benchmark-Fixture","num_hidden_layers":1,"num_experts":1,"hidden_size":2,"moe_intermediate_size":3,"num_experts_per_tok":1,"norm_topk_prob":true})";
    std::vector<std::byte> data;
    append(data, {0.5F, -0.25F, 1.0F, 0.5F, -0.5F, 1.0F,
                  1.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.5F});
    append(data, {1.0F, 0.5F, -0.25F, 0.0F, 1.0F, 0.5F});
    append(data, {1.0F, 1.0F});
    std::string header = R"({"model.layers.0.mlp.experts.gate_up_proj":{"dtype":"F32","shape":[1,6,2],"data_offsets":[0,48]},"model.layers.0.mlp.experts.down_proj":{"dtype":"F32","shape":[1,2,3],"data_offsets":[48,72]},"model.layers.0.mlp.gate.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[72,80]},"__metadata__":{"format":"pt"}})";
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((size >> shift) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
}

hypermoe::models::ExpertWeightMap weightsFrom(
    const hypermoe::models::ModelManifest& manifest) {
    using namespace hypermoe;
    const auto* mapping = manifest.findExpert(0, 0);
    if (!mapping) throw std::runtime_error("benchmark expert mapping missing");
    models::ExpertWeightMap weights;
    const auto add = [&](models::ExpertWeightType type,
                         const models::ProjectionLocation& projection) {
        const auto* tensor = manifest.findTensor(projection.tensorName);
        if (!tensor) throw std::runtime_error("benchmark projection missing");
        weights.add(0, 0, type,
            {tensor->name, projection.shape, tensor->dtype, std::nullopt,
             projection.offset, projection.size, 0, 0});
    };
    add(models::ExpertWeightType::GATE, mapping->gate);
    add(models::ExpertWeightType::UP, mapping->up);
    add(models::ExpertWeightType::DOWN, mapping->down);
    return weights;
}

} // namespace

int main(int argc, char** argv) {
    try {
        TemporaryDirectory temporary;
        std::filesystem::path artifact;
        std::filesystem::path reportPath = "real_expert_report.json";
        if (argc >= 2) {
            artifact = argv[1];
        } else {
            artifact = temporary.path() / "artifact";
            std::filesystem::create_directories(artifact);
            createFixture(artifact);
        }
        if (argc >= 3) reportPath = argv[2];
        const auto packed = temporary.path() / "hypermoe_model";

        const auto importStart = Clock::now();
        hypermoe::importer::qwen::QwenImporter importer;
        const auto sourceManifest = importer.inspect(artifact);
        const auto importEnd = Clock::now();
        const auto packStart = Clock::now();
        const auto packing = hypermoe::conversion::ExpertPacker{}.pack(
            sourceManifest,
            std::filesystem::is_directory(artifact) ? artifact : artifact.parent_path(),
            packed);
        const auto packEnd = Clock::now();

        auto store = std::make_shared<hypermoe::storage::ExpertStore>(packed);
        hypermoe::storage::DiskLoader loader(store);
        const auto loadStart = Clock::now();
        auto loaded = loader.load(0, 0);
        const auto loadEnd = Clock::now();
        auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
        const auto prepareStart = Clock::now();
        auto buffer = std::make_shared<hypermoe::backend::DeviceBuffer>(
            backend, loaded.bytes.size());
        backend->copyToDevice(buffer->data(), loaded.bytes.data(), loaded.bytes.size());
        const auto manifest = hypermoe::models::ModelManifest::load(
            packed / "manifest.json");
        auto payload = hypermoe::tensor::Tensor::fromDeviceBuffer(
            {loaded.bytes.size()}, hypermoe::tensor::DType::INT8,
            hypermoe::tensor::Device::cpu(), buffer);
        const auto weightViews = weightsFrom(manifest).createViews(
            0, 0, payload.view(), loaded.record.offset);
        auto tensors = std::make_shared<hypermoe::tensor::CpuTensorBackend>();
        auto input = tensors->allocateTensor({1, manifest.config.hiddenSize},
                                              hypermoe::tensor::DType::FP32);
        auto output = tensors->allocateTensor({1, manifest.config.hiddenSize},
                                               hypermoe::tensor::DType::FP32);
        std::fill_n(static_cast<float*>(input.data()), manifest.config.hiddenSize, 0.5F);
        const auto prepareEnd = Clock::now();
        hypermoe::ExpertMlpExecutor executor(tensors);
        const auto executeStart = Clock::now();
        executor.execute(input, weightViews, output);
        const auto executeEnd = Clock::now();

        const auto stats = backend->stats();
        std::ofstream report(reportPath, std::ios::trunc);
        report << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"artifact\": \"" << artifact.string() << "\",\n"
               << "  \"fixture\": " << (argc < 2 ? "true" : "false") << ",\n"
               << "  \"import_ms\": " << milliseconds(importStart, importEnd) << ",\n"
               << "  \"pack_ms\": " << milliseconds(packStart, packEnd) << ",\n"
               << "  \"expert_load_ms\": " << milliseconds(loadStart, loadEnd) << ",\n"
               << "  \"tensor_prepare_ms\": " << milliseconds(prepareStart, prepareEnd) << ",\n"
               << "  \"expert_execute_ms\": " << milliseconds(executeStart, executeEnd) << ",\n"
               << "  \"source_bytes_read\": " << packing.bytesRead << ",\n"
               << "  \"packed_bytes\": " << packing.bytesWritten << ",\n"
               << "  \"resident_expert_bytes\": " << loaded.bytes.size() << ",\n"
               << "  \"backend_peak_bytes\": " << stats.peakAllocatedBytes << "\n"
               << "}\n";
        if (!report) throw std::runtime_error("failed writing benchmark report");
        std::cout << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real expert benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
