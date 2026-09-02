#pragma once

#include "hypermoe/memory/memory_tier.hpp"
#include "backend/Backend.hpp"
#include "memory/PinnedBuffer.hpp"
#include "storage/DiskLoader.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace hypermoe {

namespace backend {
class CudaMemoryPool;
class CudaRuntime;
class CudaStreamManager;
}

struct TransferResult;
using TransferCallback = std::function<void(const TransferResult&)>;

struct TransferRequest {
    std::uint32_t layerId{};
    std::uint32_t expertId{};
    MemoryTier destination{MemoryTier::Vram};
    int priority{};
    MemoryTier source{MemoryTier::Nvme};
    std::shared_ptr<const std::vector<std::byte>> hostBuffer;
    std::shared_ptr<PinnedBuffer> sourcePinnedBuffer;
    std::shared_ptr<backend::DeviceBuffer> sourceDeviceBuffer;
    TransferCallback callback;
};

enum class TransferStatus {
    Completed,
    Cancelled,
};

struct TransferResult {
    TransferStatus status{TransferStatus::Cancelled};
    storage::ExpertRecord record{};
    std::shared_ptr<const std::vector<std::byte>> buffer;
    std::shared_ptr<PinnedBuffer> pinnedBuffer;
    std::shared_ptr<backend::DeviceBuffer> deviceBuffer;
    MemoryTier source{MemoryTier::Nvme};
    MemoryTier destination{MemoryTier::Ram};
    bool usedPinnedMemory{};
    std::uint64_t nvmeBytes{};
    std::uint64_t ramToVramBytes{};
    std::uint64_t completionOrder{};
    std::chrono::nanoseconds elapsed{};
    std::chrono::nanoseconds nvmeReadTime{};
    std::chrono::nanoseconds ramCopyTime{};
    std::chrono::nanoseconds backendTransferTime{};
    double backendBandwidthGiBs{};
    bool cudaTransfer{};
};

class TransferHandle {
public:
    TransferHandle() = default;
    TransferHandle(std::future<TransferResult> result,
                   std::shared_ptr<std::atomic_bool> cancelled);
    TransferHandle(const TransferHandle&) = delete;
    TransferHandle& operator=(const TransferHandle&) = delete;
    TransferHandle(TransferHandle&&) noexcept = default;
    TransferHandle& operator=(TransferHandle&&) noexcept = default;

    void cancel() noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::future<TransferResult>& future() noexcept;

private:
    std::future<TransferResult> result_;
    std::shared_ptr<std::atomic_bool> cancelled_;
};

class TransferManager {
public:
    explicit TransferManager(std::shared_ptr<const storage::DiskLoader> loader,
                             std::size_t workerCount = 1);
    TransferManager(std::shared_ptr<const storage::DiskLoader> loader,
                    std::shared_ptr<backend::ComputeBackend> backend,
                    std::size_t workerCount = 1);
    ~TransferManager();

    TransferManager(const TransferManager&) = delete;
    TransferManager& operator=(const TransferManager&) = delete;

    [[nodiscard]] TransferHandle submit(TransferRequest request);
    [[nodiscard]] std::size_t pending() const;
    [[nodiscard]] backend::BackendKind backendKind() const noexcept;
    [[nodiscard]] std::shared_ptr<backend::CudaMemoryPool> memoryPool() const noexcept;
    void shutdown();

private:
    struct Task {
        TransferRequest request;
        std::uint64_t sequence{};
        std::shared_ptr<std::atomic_bool> cancelled;
        std::promise<TransferResult> promise;
    };

    struct HigherPriority {
        bool operator()(const std::shared_ptr<Task>& left,
                        const std::shared_ptr<Task>& right) const noexcept;
    };

    void workerLoop();
    static TransferResult cancelledResult(std::chrono::steady_clock::time_point start);

    std::shared_ptr<const storage::DiskLoader> loader_;
    std::shared_ptr<backend::ComputeBackend> backend_;
    std::shared_ptr<backend::CudaMemoryPool> memoryPool_;
    std::shared_ptr<backend::CudaRuntime> cudaRuntime_;
    std::unique_ptr<backend::CudaStreamManager> cudaStreams_;
    std::mutex cudaTransferStreamMutex_;
    std::mutex cudaPrefetchStreamMutex_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::priority_queue<std::shared_ptr<Task>,
                        std::vector<std::shared_ptr<Task>>,
                        HigherPriority>
        tasks_;
    std::vector<std::thread> workers_;
    std::uint64_t nextSequence_{};
    std::atomic_uint64_t nextCompletion_{0};
    bool stopping_{};
};

} // namespace hypermoe
