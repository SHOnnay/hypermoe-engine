#include "backend/CpuBackend.hpp"
#include "memory/TransferManager.hpp"
#include "profiling/Profiler.hpp"
#include "scheduler/ExpertState.hpp"
#include "scheduler/Prefetcher.hpp"
#include "scheduler/RuntimeEvent.hpp"
#include "scheduler/Scheduler.hpp"
#include "storage/DiskLoader.hpp"
#include "storage/ExpertStore.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace std::chrono_literals;
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("hypermoe-scheduler-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void testStateMachine() {
    hypermoe::scheduler::ExpertResidencyStateMachine states;
    states.registerExpert(9);
    states.request(9, hypermoe::MemoryTier::Vram);
    expect(states.snapshot(9).state == hypermoe::scheduler::ExpertLifecycleState::Requested,
           "state machine enters REQUESTED");
    states.markQueued(9);
    states.markLoading(9);
    states.markReady(9, hypermoe::MemoryTier::Vram);
    states.acquire(9);
    expect(states.snapshot(9).usageCount == 1 &&
               states.snapshot(9).state == hypermoe::scheduler::ExpertLifecycleState::InUse,
           "state machine tracks IN_USE and usage count");
    states.release(9);
    states.beginEviction(9, hypermoe::MemoryTier::Ram);
    expect(states.snapshot(9).state == hypermoe::scheduler::ExpertLifecycleState::Evicting,
           "state machine enters EVICTING");
    states.markReady(9, hypermoe::MemoryTier::Ram);
    expect(states.snapshot(9).currentLocation == hypermoe::MemoryTier::Ram,
           "completed eviction updates current location");
    try {
        states.markLoading(9);
        expect(false, "invalid transition throws");
    } catch (const std::logic_error&) {
        expect(true, "invalid transition throws");
    }
    states.registerExpert(1, 9);
    expect(states.snapshot(1, 9).layerId == 1 &&
               states.snapshot(1, 9).currentLocation == hypermoe::MemoryTier::Nvme,
           "state identity includes layer and permits repeated per-layer expert IDs");
}

void testPrefetchPrediction() {
    hypermoe::scheduler::LocalityPrefetcher prefetcher(3);
    hypermoe::scheduler::PredictionInput input;
    input.currentLayer = 5;
    input.recentExperts = {2, 8, 12};
    input.workloadPattern[6] = {8, 12, 30};
    const auto prediction = prefetcher.predict(input);
    expect(prediction.size() == 3, "prefetcher respects prediction budget");
    expect(prediction[0].layerId == 6 && prediction[0].expertId == 12 &&
               prediction[1].expertId == 8 && prediction[2].expertId == 30,
           "prefetcher combines locality and next-layer workload pattern");
    expect(prediction[0].confidence >= prediction[1].confidence,
           "prefetch predictions are confidence ordered");
}

struct SchedulerFixture {
    TemporaryDirectory directory;
    std::shared_ptr<hypermoe::storage::ExpertStore> store;
    std::shared_ptr<hypermoe::storage::DiskLoader> loader;
    std::shared_ptr<hypermoe::backend::CpuBackend> backend;
    std::shared_ptr<hypermoe::TransferManager> transfers;
    std::shared_ptr<hypermoe::Profiler> profiler;
    std::unique_ptr<hypermoe::scheduler::Scheduler> scheduler;

    SchedulerFixture() {
        std::vector<hypermoe::storage::ExpertBlob> blobs;
        blobs.push_back({0, 0, 3, std::vector<std::byte>(8 * 1024 * 1024,
                                                        std::byte{0x10})});
        blobs.push_back({0, 1, 3, std::vector<std::byte>(4096, std::byte{0x11})});
        blobs.push_back({0, 2, 3, std::vector<std::byte>(4096, std::byte{0x12})});
        blobs.push_back({1, 3, 3, std::vector<std::byte>(4096, std::byte{0x13})});
        hypermoe::storage::ExpertStore::create(directory.path(), blobs,
                                               "{\"scheduler_test\":true}");
        store = std::make_shared<hypermoe::storage::ExpertStore>(directory.path());
        loader = std::make_shared<hypermoe::storage::DiskLoader>(store);
        backend = std::make_shared<hypermoe::backend::CpuBackend>();
        transfers = std::make_shared<hypermoe::TransferManager>(loader, backend, 1);
        profiler = std::make_shared<hypermoe::Profiler>();
        scheduler = std::make_unique<hypermoe::scheduler::Scheduler>(transfers, profiler, 1);
        scheduler->registerExpert(0, 0);
        scheduler->registerExpert(0, 1);
        scheduler->registerExpert(0, 2);
        scheduler->registerExpert(1, 3);
    }
};

hypermoe::scheduler::ScheduleRequest request(hypermoe::ExpertId id,
                                             hypermoe::LayerId layer,
                                             hypermoe::scheduler::TransferPriority priority) {
    hypermoe::scheduler::ScheduleRequest result;
    result.layerId = layer;
    result.expertId = id;
    result.priority = priority;
    return result;
}

void testSchedulerPriorityEventsAndState() {
    SchedulerFixture fixture;
    std::mutex eventMutex;
    std::condition_variable eventReady;
    std::vector<hypermoe::ExpertId> started;
    std::vector<hypermoe::scheduler::RuntimeEventType> eventTypes;
    const auto subscription = fixture.scheduler->events().subscribe(
        [&](const hypermoe::scheduler::RuntimeEvent& event) {
            std::scoped_lock lock(eventMutex);
            eventTypes.push_back(event.type);
            if (event.type == hypermoe::scheduler::RuntimeEventType::TransferStarted) {
                started.push_back(event.expertId);
                eventReady.notify_all();
            }
        });

    auto blocker = fixture.scheduler->schedule(request(
        0, 0, hypermoe::scheduler::TransferPriority::ActiveInference));
    {
        std::unique_lock lock(eventMutex);
        eventReady.wait_for(lock, 2s, [&] { return !started.empty(); });
    }
    auto low = fixture.scheduler->schedule(request(
        1, 0, hypermoe::scheduler::TransferPriority::BackgroundMaintenance));
    auto high = fixture.scheduler->schedule(request(
        2, 0, hypermoe::scheduler::TransferPriority::ActiveInference));
    expect(blocker.future().get().success, "scheduler completes blocking request");
    expect(high.future().get().success && low.future().get().success,
           "scheduler completes prioritized requests");
    {
        std::scoped_lock lock(eventMutex);
        expect(started.size() == 3 && started[0] == 0 && started[1] == 2 && started[2] == 1,
               "active request runs before queued background maintenance");
        expect(std::find(eventTypes.begin(), eventTypes.end(),
                         hypermoe::scheduler::RuntimeEventType::ExpertReady) !=
                   eventTypes.end(),
               "scheduler emits readiness events");
    }
    fixture.scheduler->acquire(2);
    fixture.scheduler->release(2);
    expect(fixture.scheduler->state(2).usageCount == 1,
           "scheduler exposes consumer lifecycle transitions");
    expect(fixture.scheduler->events().unsubscribe(subscription),
           "event subscriptions can be removed");
}

void testSchedulerPrefetchAndCoalescing() {
    SchedulerFixture fixture;
    hypermoe::scheduler::PredictedExpertRequest prediction{1, 3, 0.9};
    auto prefetched = fixture.scheduler->prefetch(prediction);
    expect(prefetched.future().get().success, "predicted expert loads in background");

    auto active = fixture.scheduler->schedule(request(
        3, 1, hypermoe::scheduler::TransferPriority::ActiveInference));
    expect(active.future().get().success, "active consumer observes prefetched expert ready");
    const auto metrics = fixture.profiler->snapshot();
    expect(metrics.prefetchRequests == 1 && metrics.prefetchHits == 1,
           "profiler records a consumed prefetch hit");
    expect(metrics.queueWaitSamples >= 1 && metrics.averageQueueWaitMs() >= 0.0,
           "scheduler records queue wait samples");

    SchedulerFixture coalescedFixture;
    auto speculative = coalescedFixture.scheduler->prefetch({0, 0, 0.8});
    auto demanded = coalescedFixture.scheduler->schedule(request(
        0, 0, hypermoe::scheduler::TransferPriority::ActiveInference));
    const auto speculativeResult = speculative.future().get();
    const auto demandedResult = demanded.future().get();
    expect(speculativeResult.transfer.completionOrder ==
               demandedResult.transfer.completionOrder,
           "active request coalesces with an in-flight prefetch");
    expect(coalescedFixture.profiler->snapshot().prefetchMisses == 1,
           "in-flight prefetch that still stalls is counted as a miss");
}

void testSchedulerCancellationAndOverlapMetrics() {
    SchedulerFixture fixture;
    std::mutex eventMutex;
    std::condition_variable eventReady;
    bool blockerStarted = false;
    bool failureObserved = false;
    const auto subscription = fixture.scheduler->events().subscribe(
        [&](const hypermoe::scheduler::RuntimeEvent& event) {
            std::scoped_lock lock(eventMutex);
            if (event.type == hypermoe::scheduler::RuntimeEventType::TransferStarted &&
                event.expertId == 0) {
                blockerStarted = true;
                eventReady.notify_all();
            }
            if (event.type == hypermoe::scheduler::RuntimeEventType::TransferFailed &&
                event.expertId == 1) {
                failureObserved = true;
            }
        });
    auto blocker = fixture.scheduler->schedule(request(
        0, 0, hypermoe::scheduler::TransferPriority::ActiveInference));
    {
        std::unique_lock lock(eventMutex);
        eventReady.wait_for(lock, 2s, [&] { return blockerStarted; });
    }
    auto cancelled = fixture.scheduler->schedule(request(
        1, 0, hypermoe::scheduler::TransferPriority::BackgroundMaintenance));
    cancelled.cancel();
    expect(blocker.future().get().success, "leading request completes before cancellation test");
    const auto result = cancelled.future().get();
    expect(result.cancelled &&
               result.state.state == hypermoe::scheduler::ExpertLifecycleState::Failed,
           "queued cancellation resolves its future and moves state to FAILED");
    {
        std::scoped_lock lock(eventMutex);
        expect(failureObserved, "cancellation publishes TRANSFER_FAILED");
    }

    fixture.profiler->recordTransferOverlap(10ms, 4ms);
    const auto metrics = fixture.profiler->snapshot();
    expect(metrics.transferOverlapPercentage() == 40.0,
           "profiler computes hidden transfer percentage");
    const auto json = fixture.profiler->toJson();
    expect(json.find("\"average_queue_wait_ms\"") != std::string::npos &&
               json.find("\"transfer_overlap_percentage\": 40.000000") !=
                   std::string::npos,
           "profiler exports Phase 3.5 metrics");
    expect(fixture.scheduler->events().unsubscribe(subscription),
           "cancellation-test subscription can be removed");
}

} // namespace

int main() {
    testStateMachine();
    testPrefetchPrediction();
    testSchedulerPriorityEventsAndState();
    testSchedulerPrefetchAndCoalescing();
    testSchedulerCancellationAndOverlapMetrics();
    if (failures != 0) {
        std::cerr << failures << " scheduler assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All scheduler tests passed\n";
    return EXIT_SUCCESS;
}
