#include "cache/HybridPolicy.hpp"
#include "cache/LFUPolicy.hpp"
#include "cache/LRUPolicy.hpp"
#include "hypermoe/experts/expert_manager.hpp"
#include "memory/MemoryPressureController.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"
#include "storage/MappedFile.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using namespace std::chrono_literals;
constexpr std::size_t KiB = 1024;
int failures = 0;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
                ("hypermoe-test-" + std::to_string(sequence.fetch_add(1)) + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::byte> bytes(std::size_t size, unsigned char seed) {
    std::vector<std::byte> result(size);
    for (std::size_t index = 0; index < size; ++index) {
        result[index] = std::byte((seed + static_cast<unsigned char>(index)) & 0xffU);
    }
    return result;
}

std::vector<hypermoe::storage::ExpertBlob> smallBlobs(bool withBlocker = false) {
    std::vector<hypermoe::storage::ExpertBlob> blobs;
    blobs.push_back({0, 0, 3, bytes(withBlocker ? 4 * 1024 * KiB : 4 * KiB, 11)});
    blobs.push_back({0, 1, 3, bytes(8 * KiB, 29)});
    blobs.push_back({1, 2, 3, bytes(2 * KiB, 71)});
    return blobs;
}

std::shared_ptr<hypermoe::storage::ExpertStore>
makeStore(const std::filesystem::path& path, bool withBlocker = false) {
    const auto blobs = smallBlobs(withBlocker);
    hypermoe::storage::ExpertStore::create(path, blobs, "{\"format\":\"test\"}");
    return std::make_shared<hypermoe::storage::ExpertStore>(path);
}

void testExpertIndexStoreAndMapping() {
    TemporaryDirectory directory;
    const auto blobs = smallBlobs();
    hypermoe::storage::ExpertStore::create(directory.path(), blobs, "{\"model\":\"unit\"}");

    hypermoe::storage::ExpertStore store(directory.path());
    expect(store.index().size() == blobs.size(), "index loads every record");
    const auto record = store.index().find(0, 1);
    expect(record.has_value() && record->offset % 4096 == 0,
           "index lookup is composite-keyed and aligned");
    expect(store.mappedExpert(0, 1).size() == blobs[1].data.size(),
           "mmap returns only the selected expert");
    expect(store.readExpert(1, 2) == blobs[2].data, "range read returns exact bytes");

    hypermoe::storage::MappedFile mapped(directory.path() / "experts.bin");
    expect(mapped.isOpen() && mapped.size() >= blobs[0].data.size(),
           "mapped file owns a read-only mapping");
    expect(mapped.view(record->offset, record->size).size() == record->size,
           "mapped range is bounds checked");
    try {
        (void)mapped.view(mapped.size(), 1);
        expect(false, "out-of-range mapping throws");
    } catch (const hypermoe::storage::StorageError&) {
        expect(true, "out-of-range mapping throws");
    }
}

void testCorruptionDetection() {
    TemporaryDirectory directory;
    auto store = makeStore(directory.path());
    const auto record = store->index().find(0, 1).value();
    store.reset();

    std::fstream data(directory.path() / "experts.bin",
                      std::ios::binary | std::ios::in | std::ios::out);
    data.seekp(static_cast<std::streamoff>(record.offset));
    data.put(static_cast<char>(0xff));
    data.close();

    hypermoe::storage::ExpertStore corrupted(directory.path());
    try {
        (void)corrupted.readExpert(0, 1);
        expect(false, "checksum mismatch throws");
    } catch (const hypermoe::storage::StorageError&) {
        expect(true, "checksum mismatch throws");
    }
}

void testDiskLoaderAndTransfers() {
    TemporaryDirectory directory;
    const auto store = makeStore(directory.path(), true);
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(
        store, hypermoe::storage::DiskReadMode::RangeRead);
    const auto loaded = loader->load(0, 1);
    expect(loaded.bytes.size() == 8 * KiB, "DiskLoader performs an expert range read");
    expect(loader->loadAsync(1, 2).get().bytes.size() == 2 * KiB,
           "DiskLoader asynchronous API returns checked bytes");

    hypermoe::TransferManager transfers(loader, 1);
    auto blocker = transfers.submit({.layerId = 0,
                                     .expertId = 0,
                                     .destination = hypermoe::MemoryTier::Ram,
                                     .priority = 0});
    auto low = transfers.submit({.layerId = 0,
                                 .expertId = 1,
                                 .destination = hypermoe::MemoryTier::Vram,
                                 .priority = 1});
    auto high = transfers.submit({.layerId = 1,
                                  .expertId = 2,
                                  .destination = hypermoe::MemoryTier::Vram,
                                  .priority = 10});
    const auto blockerResult = blocker.future().get();
    const auto highResult = high.future().get();
    const auto lowResult = low.future().get();
    expect(blockerResult.status == hypermoe::TransferStatus::Completed,
           "RAM staging transfer completes");
    expect(highResult.completionOrder < lowResult.completionOrder,
           "queued high-priority transfer runs before low priority");
    expect(highResult.ramToVramBytes == highResult.record.size,
           "VRAM staging accounts for the PCIe copy boundary");

    auto cancellationBlocker = transfers.submit({.layerId = 0,
                                                  .expertId = 0,
                                                  .destination = hypermoe::MemoryTier::Vram,
                                                  .priority = 10});
    auto cancelled = transfers.submit({.layerId = 0,
                                       .expertId = 0,
                                       .destination = hypermoe::MemoryTier::Vram,
                                       .priority = 0});
    cancelled.cancel();
    (void)cancellationBlocker.future().get();
    expect(cancelled.future().get().status == hypermoe::TransferStatus::Cancelled,
           "transfer cancellation is observable through the future");

    auto integratedTransfers = std::make_shared<hypermoe::TransferManager>(loader, 1);
    hypermoe::MemoryManager memory(16 * KiB, 16 * KiB);
    hypermoe::ExpertManager experts(
        memory, std::make_unique<hypermoe::LRUPolicy>(), integratedTransfers);
    experts.registerExpert({1, 0, 8 * KiB, hypermoe::QuantizationType::Q4,
                            hypermoe::MemoryTier::Nvme});
    (void)experts.requestExpert(1);
    const auto resident = experts.residentDeviceWeights(1);
    std::vector<std::byte> roundTrip(8 * KiB);
    if (resident) {
        resident->backend()->copyFromDevice(
            roundTrip.data(), resident->data(), resident->size());
        resident->backend()->synchronize();
    }
    expect(resident && roundTrip == smallBlobs()[1].data,
           "ExpertManager owns checked backend bytes after hierarchical movement");
}

template <typename Policy>
std::optional<hypermoe::ExpertId> victim(Policy& policy,
                                         std::span<const hypermoe::ExpertId> ids) {
    return policy.selectVictim(hypermoe::MemoryTier::Vram, ids, {});
}

void testCachePolicies() {
    const std::vector<hypermoe::ExpertId> ids{1, 2, 3};
    hypermoe::LRUPolicy lru;
    for (const auto id : ids) lru.onResident(id, hypermoe::MemoryTier::Vram);
    lru.onAccess(1);
    expect(victim(lru, ids) == 2, "LRU evicts the oldest resident");

    hypermoe::LFUPolicy lfu;
    for (const auto id : ids) lfu.onResident(id, hypermoe::MemoryTier::Vram);
    lfu.onAccess(1);
    lfu.onAccess(1);
    lfu.onAccess(2);
    expect(victim(lfu, ids) == 3, "LFU evicts the least frequently used resident");

    hypermoe::HybridPolicy hybrid;
    for (const auto id : ids) hybrid.onResident(id, hypermoe::MemoryTier::Vram);
    hybrid.onAccess(1);
    hybrid.setLayerProbability(2, 1.0);
    hybrid.setPrefetchConfidence(2, 1.0);
    expect(victim(hybrid, ids) == 3, "hybrid score preserves predicted experts");
    expect(hybrid.score(1) > hybrid.score(3), "hybrid frequency and recency affect score");
}

void testMemoryPressureController() {
    TemporaryDirectory directory;
    const auto store = makeStore(directory.path());
    auto loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
    hypermoe::TransferManager transfers(loader);
    hypermoe::MemoryManager memory(12 * KiB, 8 * KiB);
    hypermoe::ExpertManager experts(memory, std::make_unique<hypermoe::LRUPolicy>());
    for (hypermoe::ExpertId id = 0; id < 3; ++id) {
        experts.registerExpert({id, 0, 4 * KiB, hypermoe::QuantizationType::Q4,
                                hypermoe::MemoryTier::Nvme});
        (void)experts.requestExpert(id);
    }
    hypermoe::MemoryPressureController pressure(
        memory, experts, transfers, {4 * KiB, 8 * KiB, 64});
    const auto report = pressure.poll();
    expect(report.vramPressure && report.vramExpertsMoved >= 1,
           "VRAM safety margin triggers cold demotion");
    expect(report.ramPressure && report.ramExpertsMoved >= 1,
           "RAM safety margin triggers storage eviction");
    expect(memory.snapshot().vram.usedBytes <= 8 * KiB,
           "controller restores configured VRAM margin");
    expect(memory.snapshot().ram.usedBytes == 0,
           "controller restores configured RAM margin");
}

void testProfilerJson() {
    hypermoe::Profiler profiler;
    profiler.recordExpertRequest(true);
    profiler.recordExpertRequest(false);
    profiler.recordNvmeRead(4096);
    profiler.recordRamToVram(4096);
    profiler.recordTransferTime(2ms);
    profiler.recordStallTime(1ms);
    const auto json = profiler.toJson();
    expect(json.find("\"expert_requests\": 2") != std::string::npos,
           "profiler exports request count as JSON");
    expect(json.find("\"cache_hit_rate\": 0.500000") != std::string::npos,
           "profiler exports deterministic derived metrics");
}

} // namespace

int main() {
    testExpertIndexStoreAndMapping();
    testCorruptionDetection();
    testDiskLoaderAndTransfers();
    testCachePolicies();
    testMemoryPressureController();
    testProfilerJson();
    if (failures != 0) {
        std::cerr << failures << " Phase 2 test assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Phase 2 unit tests passed\n";
    return EXIT_SUCCESS;
}
