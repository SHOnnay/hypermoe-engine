#include "cache/HybridPolicy.hpp"
#include "backend/CpuBackend.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/TransferManager.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kLayers = 4;
constexpr std::uint32_t kExpertsPerLayer = 8;
constexpr std::uint32_t kTopK = 2;
constexpr std::uint32_t kTokens = 100'000;
constexpr std::size_t kExpertBytes = 256;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-integration-" + std::to_string(sequence.fetch_add(1)) + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

struct Workload {
    std::mt19937 random{20260902};
    std::array<std::uint32_t, kLayers> previous{};

    std::array<std::uint32_t, kTopK> route(std::uint32_t layer) {
        std::bernoulli_distribution retainLocality(0.86);
        std::uniform_int_distribution<std::uint32_t> select(0, kExpertsPerLayer - 1);
        auto primary = retainLocality(random) ? previous[layer] : select(random);
        previous[layer] = primary;
        return {primary, (primary + 1U + layer) % kExpertsPerLayer};
    }
};

std::uint64_t routeHash() {
    Workload workload;
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            for (const auto local : workload.route(layer)) {
                hash ^= static_cast<std::uint64_t>(layer * kExpertsPerLayer + local);
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

} // namespace

int main() {
    if (routeHash() != routeHash()) {
        std::cerr << "workload routing is not deterministic\n";
        return EXIT_FAILURE;
    }

    TemporaryDirectory model;
    std::vector<hypermoe::storage::ExpertBlob> blobs;
    blobs.reserve(kLayers * kExpertsPerLayer);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t local = 0; local < kExpertsPerLayer; ++local) {
            const auto id = layer * kExpertsPerLayer + local;
            std::vector<std::byte> weights(kExpertBytes, static_cast<std::byte>(id));
            blobs.push_back({layer, id,
                             static_cast<std::uint32_t>(hypermoe::QuantizationType::Q4),
                             std::move(weights)});
        }
    }
    hypermoe::storage::ExpertStore::create(model.path(), blobs, "{\"integration\":true}");
    auto store = std::make_shared<hypermoe::storage::ExpertStore>(model.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    auto backend = std::make_shared<hypermoe::backend::CpuBackend>();
    auto transfers = std::make_shared<hypermoe::TransferManager>(loader, backend);

    hypermoe::MemoryManager memory(8 * kExpertBytes, 16 * kExpertBytes);
    hypermoe::ExpertManager manager(
        memory, std::make_unique<hypermoe::HybridPolicy>(), transfers);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t local = 0; local < kExpertsPerLayer; ++local) {
            const auto id = layer * kExpertsPerLayer + local;
            manager.registerExpert({id, layer, kExpertBytes,
                                    hypermoe::QuantizationType::Q4,
                                    hypermoe::MemoryTier::Nvme});
        }
    }

    Workload workload;
    for (std::uint32_t token = 0; token < kTokens; ++token) {
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            for (const auto local : workload.route(layer)) {
                const auto id = layer * kExpertsPerLayer + local;
                (void)manager.requestExpert(id);
            }
        }
    }

    const auto stats = manager.stats();
    const auto usage = memory.snapshot();
    const auto expected = static_cast<std::uint64_t>(kTokens) * kLayers * kTopK;
    if (stats.requests != expected || stats.vramHits == 0 || stats.nvmeLoads == 0 ||
        stats.vramEvictions == 0 || usage.vram.usedBytes > usage.vram.limitBytes ||
        usage.ram.usedBytes > usage.ram.limitBytes) {
        std::cerr << "100000-token integration invariants failed\n";
        return EXIT_FAILURE;
    }
    if (backend->stats().allocatedBytes != usage.vram.usedBytes) {
        std::cerr << "physical and logical VRAM accounting diverged\n";
        return EXIT_FAILURE;
    }
    for (std::uint32_t id = 0; id < kLayers * kExpertsPerLayer; ++id) {
        const auto expert = manager.findExpert(id).value();
        const auto weights = manager.residentWeights(id);
        const auto deviceWeights = manager.residentDeviceWeights(id);
        if ((expert.location == hypermoe::MemoryTier::Nvme &&
             (weights || deviceWeights)) ||
            (expert.location == hypermoe::MemoryTier::Ram &&
             (!weights || deviceWeights || weights->size() != kExpertBytes)) ||
            (expert.location == hypermoe::MemoryTier::Vram &&
             (weights || !deviceWeights || deviceWeights->size() != kExpertBytes))) {
            std::cerr << "resident weight ownership invariant failed\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "Phase 2 integration passed: " << stats.requests
              << " expert requests, deterministic hash " << routeHash() << '\n';
    return EXIT_SUCCESS;
}
