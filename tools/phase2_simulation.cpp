#include "cache/HybridPolicy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/MemoryPressureController.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t KiB = 1024;
constexpr std::uint32_t kLayers = 8;
constexpr std::uint32_t kExpertsPerLayer = 16;
constexpr std::uint32_t kTopK = 3;
constexpr std::size_t kExpertBytes = 4 * KiB;
constexpr double kNvmeBytesPerMs = 2.5 * 1024.0 * 1024.0;
constexpr double kPcieBytesPerMs = 12.0 * 1024.0 * 1024.0;

struct Options {
    std::uint32_t tokens{100'000};
    std::uint32_t seed{20260902};
    std::filesystem::path report{"phase2_benchmark.json"};
    hypermoe::storage::DiskReadMode mode{hypermoe::storage::DiskReadMode::MemoryMap};
};

template <typename T>
T parseNumber(std::string_view value, std::string_view name) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto equals = argument.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("expected --name=value argument");
        }
        const auto name = argument.substr(0, equals);
        const auto value = argument.substr(equals + 1);
        if (name == "--tokens") options.tokens = parseNumber<std::uint32_t>(value, name);
        else if (name == "--seed") options.seed = parseNumber<std::uint32_t>(value, name);
        else if (name == "--report") options.report = value;
        else if (name == "--read-mode") {
            if (value == "mmap") options.mode = hypermoe::storage::DiskReadMode::MemoryMap;
            else if (value == "range") options.mode = hypermoe::storage::DiskReadMode::RangeRead;
            else throw std::invalid_argument("--read-mode must be mmap or range");
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    return options;
}

class TemporaryModel {
public:
    explicit TemporaryModel(std::uint32_t seed)
        : path_(std::filesystem::temp_directory_path() /
                ("hypermoe-phase2-" + std::to_string(seed) + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}

    ~TemporaryModel() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<std::byte> makeWeights(std::uint32_t layer, std::uint32_t expert) {
    std::vector<std::byte> result(kExpertBytes);
    std::uint32_t state = 0x9e3779b9U ^ (layer << 16U) ^ expert;
    for (auto& value : result) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<std::byte>(state >> 24U);
    }
    return result;
}

std::vector<hypermoe::storage::ExpertBlob> makeModel() {
    std::vector<hypermoe::storage::ExpertBlob> blobs;
    blobs.reserve(kLayers * kExpertsPerLayer);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t local = 0; local < kExpertsPerLayer; ++local) {
            const auto global = layer * kExpertsPerLayer + local;
            blobs.push_back({layer, global,
                             static_cast<std::uint32_t>(hypermoe::QuantizationType::Q4),
                             makeWeights(layer, local)});
        }
    }
    return blobs;
}

class RouterWorkload {
public:
    explicit RouterWorkload(std::uint32_t seed) : random_(seed) {
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            previous_[layer] = layer % 4U;
        }
    }

    std::array<std::uint32_t, kTopK> route(std::uint32_t layer) {
        std::bernoulli_distribution retainLocality(0.88);
        std::bernoulli_distribution chooseHot(0.72);
        std::uniform_int_distribution<std::uint32_t> hot(0, 3);
        std::uniform_int_distribution<std::uint32_t> all(0, kExpertsPerLayer - 1);
        const auto primary = retainLocality(random_)
                                 ? previous_[layer]
                                 : (chooseHot(random_) ? hot(random_) : all(random_));
        previous_[layer] = primary;
        return {primary,
                (primary + 1U + layer % 4U) % kExpertsPerLayer,
                (primary + 7U) % kExpertsPerLayer};
    }

private:
    std::mt19937 random_;
    std::array<std::uint32_t, kLayers> previous_{};
};

std::chrono::nanoseconds milliseconds(double value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double, std::milli>(value));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        TemporaryModel model(options.seed);
        const auto blobs = makeModel();
        hypermoe::storage::ExpertStore::create(
            model.path(), blobs,
            "{\"format\":\"hypermoe-expert-store\",\"version\":1,\"quantization\":\"Q4\"}");
        auto store = std::make_shared<hypermoe::storage::ExpertStore>(model.path());
        auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store, options.mode);
        auto transfers = std::make_shared<hypermoe::TransferManager>(loader, 1);

        constexpr std::size_t vramCapacity = 24 * kExpertBytes;
        constexpr std::size_t ramCapacity = 64 * kExpertBytes;
        hypermoe::MemoryManager memory(vramCapacity, ramCapacity);
        auto policy = std::make_unique<hypermoe::HybridPolicy>();
        auto* policyView = policy.get();
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            for (std::uint32_t local = 0; local < kExpertsPerLayer; ++local) {
                const auto global = layer * kExpertsPerLayer + local;
                policyView->setLayerProbability(global, local < 4 ? 0.8 : 0.2);
            }
        }
        hypermoe::ExpertManager experts(memory, std::move(policy), transfers);
        for (const auto& blob : blobs) {
            experts.registerExpert({blob.expertId, blob.layerId, blob.data.size(),
                                    hypermoe::QuantizationType::Q4,
                                    hypermoe::MemoryTier::Nvme});
        }
        hypermoe::MemoryPressureController pressure(
            memory, experts, *transfers,
            {2 * kExpertBytes, 4 * kExpertBytes, 32});
        hypermoe::Profiler profiler;
        RouterWorkload workload(options.seed);

        const auto benchmarkStart = std::chrono::steady_clock::now();
        for (std::uint32_t token = 0; token < options.tokens; ++token) {
            profiler.recordToken();
            for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
                for (const auto local : workload.route(layer)) {
                    const auto id = layer * kExpertsPerLayer + local;
                    const auto result = experts.requestExpert(id);
                    const bool hierarchyHit = result.source != hypermoe::RequestSource::NvmeLoad;
                    profiler.recordExpertRequest(hierarchyHit);
                    if (result.source == hypermoe::RequestSource::NvmeLoad) {
                        profiler.recordNvmeRead(result.expert.sizeBytes);
                        profiler.recordModeledLatency(
                            0.18 + static_cast<double>(result.expert.sizeBytes) /
                                       kNvmeBytesPerMs);
                    }
                    if (result.source != hypermoe::RequestSource::VramHit) {
                        profiler.recordRamToVram(result.expert.sizeBytes);
                        profiler.recordTransferTime(milliseconds(result.simulatedLatencyMs));
                        profiler.recordStallTime(milliseconds(result.simulatedLatencyMs));
                        profiler.recordModeledLatency(
                            0.03 + static_cast<double>(result.expert.sizeBytes) /
                                       kPcieBytesPerMs);
                    }
                }
            }
            const auto usage = memory.snapshot();
            profiler.observeMemory(usage.vram.usedBytes, usage.ram.usedBytes);
            if ((token + 1U) % 256U == 0) {
                const auto report = pressure.poll();
                profiler.recordMemoryPressureEvent(
                    static_cast<std::uint64_t>(report.vramPressure) +
                    static_cast<std::uint64_t>(report.ramPressure) +
                    static_cast<std::uint64_t>(report.storageQueuePressure));
            }
        }
        const auto elapsed = std::chrono::steady_clock::now() - benchmarkStart;
        const auto managerStats = experts.stats();
        profiler.recordVramEvictions(managerStats.vramEvictions);
        profiler.recordRamEvictions(managerStats.ramEvictions);
        profiler.exportJson(options.report);

        const auto metrics = profiler.snapshot();
        std::cout << "HyperMoE Phase 2 hierarchical storage simulation\n"
                  << "  tokens:                  " << metrics.tokensProcessed << '\n'
                  << "  layers / top-k:          " << kLayers << " / " << kTopK << '\n'
                  << "  expert requests:         " << metrics.expertRequests << '\n'
                  << std::fixed << std::setprecision(2)
                  << "  hierarchy cache hit:     " << metrics.cacheHitRate() * 100.0 << "%\n"
                  << "  VRAM hit rate:           " << managerStats.vramHitRate() * 100.0 << "%\n"
                  << "  NVMe reads:              " << metrics.nvmeReads << '\n'
                  << "  NVMe bytes read:         " << metrics.nvmeBytes / KiB << " KiB\n"
                  << "  RAM -> VRAM transferred: " << metrics.ramToVramBytes / KiB << " KiB\n"
                  << "  VRAM evictions:          " << metrics.vramEvictions << '\n'
                  << "  RAM evictions:           " << metrics.ramEvictions << '\n'
                  << "  pressure events:         " << metrics.memoryPressureEvents << '\n'
                  << "  modeled latency:         " << metrics.modeledLatencyMs << " ms\n"
                  << "  wall time:               "
                  << std::chrono::duration<double>(elapsed).count() << " s\n"
                  << "  JSON report:             " << options.report << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Phase 2 simulation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
