#include "memory/TransferManager.hpp"

#include "backend/CpuBackend.hpp"
#include "backend/cuda/CudaMemoryPool.hpp"
#include "backend/cuda/CudaRuntime.hpp"
#include "backend/cuda/CudaStreamManager.hpp"

#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hypermoe {
namespace {

double bandwidthGiBs(std::uint64_t bytes, std::chrono::nanoseconds duration) {
    if (bytes == 0 || duration.count() <= 0) return 0.0;
    const auto seconds = std::chrono::duration<double>(duration).count();
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds;
}

} // namespace

TransferHandle::TransferHandle(std::future<TransferResult> result,
                               std::shared_ptr<std::atomic_bool> cancelled)
    : result_(std::move(result)), cancelled_(std::move(cancelled)) {}

void TransferHandle::cancel() noexcept {
    if (cancelled_) {
        cancelled_->store(true, std::memory_order_relaxed);
    }
}

bool TransferHandle::valid() const noexcept {
    return result_.valid();
}

std::future<TransferResult>& TransferHandle::future() noexcept {
    return result_;
}

TransferManager::TransferManager(std::shared_ptr<const storage::DiskLoader> loader,
                                 std::size_t workerCount)
    : TransferManager(std::move(loader),
                      std::make_shared<backend::CpuBackend>(),
                      workerCount) {}

TransferManager::TransferManager(std::shared_ptr<const storage::DiskLoader> loader,
                                 std::shared_ptr<backend::ComputeBackend> backend,
                                 std::size_t workerCount)
    : loader_(std::move(loader)), backend_(std::move(backend)) {
    if (!loader_) {
        throw std::invalid_argument("TransferManager requires a DiskLoader");
    }
    if (workerCount == 0) {
        throw std::invalid_argument("TransferManager requires at least one worker");
    }
    if (!backend_ || !backend_->isAvailable()) {
        throw std::invalid_argument("TransferManager requires an available backend");
    }
    if (backend_->kind() == backend::BackendKind::Cuda) {
        memoryPool_ = std::make_shared<backend::CudaMemoryPool>(backend_);
        cudaRuntime_ =
            std::make_shared<backend::CudaRuntime>(backend_->deviceOrdinal());
        if (!cudaRuntime_->available()) {
            throw std::runtime_error("CUDA backend has no matching runtime device");
        }
        cudaStreams_ = std::make_unique<backend::CudaStreamManager>(cudaRuntime_);
    }
    workers_.reserve(workerCount);
    for (std::size_t index = 0; index < workerCount; ++index) {
        workers_.emplace_back(&TransferManager::workerLoop, this);
    }
}

TransferManager::~TransferManager() {
    shutdown();
}

TransferHandle TransferManager::submit(TransferRequest request) {
    if (request.destination == MemoryTier::Nvme) {
        throw std::invalid_argument("a load destination cannot be NVMe");
    }
    auto task = std::make_shared<Task>();
    task->request = request;
    task->cancelled = std::make_shared<std::atomic_bool>(false);
    auto future = task->promise.get_future();
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("TransferManager is shutting down");
        }
        task->sequence = nextSequence_++;
        tasks_.push(task);
    }
    ready_.notify_one();
    return {std::move(future), task->cancelled};
}

std::size_t TransferManager::pending() const {
    std::scoped_lock lock(mutex_);
    return tasks_.size();
}

backend::BackendKind TransferManager::backendKind() const noexcept {
    return backend_->kind();
}

std::shared_ptr<backend::CudaMemoryPool> TransferManager::memoryPool() const noexcept {
    return memoryPool_;
}

void TransferManager::shutdown() {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return;
        }
        stopping_ = true;
        auto pendingTasks = tasks_;
        while (!pendingTasks.empty()) {
            pendingTasks.top()->cancelled->store(true, std::memory_order_relaxed);
            pendingTasks.pop();
        }
    }
    ready_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

bool TransferManager::HigherPriority::operator()(
    const std::shared_ptr<Task>& left,
    const std::shared_ptr<Task>& right) const noexcept {
    if (left->request.priority != right->request.priority) {
        return left->request.priority < right->request.priority;
    }
    return left->sequence > right->sequence;
}

void TransferManager::workerLoop() {
    while (true) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) {
                return;
            }
            task = tasks_.top();
            tasks_.pop();
        }

        const auto start = std::chrono::steady_clock::now();
        if (task->cancelled->load(std::memory_order_relaxed)) {
            task->promise.set_value(cancelledResult(start));
            continue;
        }
        backend::StreamHandle stream = nullptr;
        backend::EventHandle completion = nullptr;
        bool ownsStream = false;
        std::unique_lock<std::mutex> cudaStreamLock;
        try {
            storage::LoadedExpert loaded;
            std::shared_ptr<const std::vector<std::byte>> sourceBuffer;
            std::shared_ptr<PinnedBuffer> sourcePinned;
            std::shared_ptr<backend::DeviceBuffer> sourceDevice;
            const auto nvmeStart = std::chrono::steady_clock::now();
            if (task->request.source == MemoryTier::Nvme) {
                loaded = loader_->load(task->request.layerId, task->request.expertId);
                sourceBuffer = std::make_shared<const std::vector<std::byte>>(
                    std::move(loaded.bytes));
            } else {
                const auto record = loader_->store().index().find(
                    task->request.layerId, task->request.expertId);
                if (!record) throw storage::StorageError("transfer expert is not indexed");
                loaded.record = *record;
                if (task->request.source == MemoryTier::PinnedRam) {
                    sourcePinned = task->request.sourcePinnedBuffer;
                    if (!sourcePinned || sourcePinned->size() != record->size) {
                        throw std::invalid_argument("pinned source buffer size mismatch");
                    }
                } else if (task->request.source == MemoryTier::Vram) {
                    sourceDevice = task->request.sourceDeviceBuffer;
                    if (!sourceDevice || sourceDevice->size() != record->size) {
                        throw std::invalid_argument("device source buffer size mismatch");
                    }
                    if (sourceDevice->backend().get() != backend_.get()) {
                        throw std::invalid_argument("device source belongs to another backend");
                    }
                } else {
                    sourceBuffer = task->request.hostBuffer;
                    if (!sourceBuffer || sourceBuffer->size() != record->size) {
                        throw std::invalid_argument("RAM source buffer size mismatch");
                    }
                }
            }
            const auto nvmeEnd = std::chrono::steady_clock::now();
            if (task->cancelled->load(std::memory_order_relaxed)) {
                task->promise.set_value(cancelledResult(start));
                continue;
            }

            TransferResult result;
            result.status = TransferStatus::Completed;
            result.record = loaded.record;
            result.source = task->request.source;
            result.destination = task->request.destination;
            result.nvmeBytes = task->request.source == MemoryTier::Nvme
                                   ? loaded.record.size
                                   : 0;
            result.nvmeReadTime = task->request.source == MemoryTier::Nvme
                                      ? nvmeEnd - nvmeStart
                                      : std::chrono::nanoseconds{};

            if (task->request.destination == MemoryTier::Ram) {
                if (sourceDevice) {
                    auto pinned = std::make_shared<PinnedBuffer>(loaded.record.size, backend_);
                    if (cudaStreams_) {
                        cudaStreamLock =
                            std::unique_lock<std::mutex>(cudaTransferStreamMutex_);
                        stream = cudaStreams_->stream(backend::CudaStreamRole::Transfer);
                    } else {
                        stream = backend_->createStream();
                        ownsStream = true;
                    }
                    completion = backend_->createEvent();
                    const auto transferStart = std::chrono::steady_clock::now();
                    backend_->copyFromDevice(pinned->data(), sourceDevice->data(),
                                             pinned->size(), stream);
                    backend_->recordEvent(completion, stream);
                    backend_->waitEvent(completion);
                    backend_->synchronize(stream);
                    result.backendTransferTime =
                        std::chrono::steady_clock::now() - transferStart;
                    result.cudaTransfer = backend_->kind() == backend::BackendKind::Cuda;
                    result.backendBandwidthGiBs =
                        bandwidthGiBs(loaded.record.size, result.backendTransferTime);
                    const auto copyStart = std::chrono::steady_clock::now();
                    result.buffer = std::make_shared<const std::vector<std::byte>>(
                        pinned->bytes().begin(), pinned->bytes().end());
                    result.ramCopyTime = std::chrono::steady_clock::now() - copyStart;
                    result.usedPinnedMemory = pinned->isPinned();
                    backend_->destroyEvent(completion);
                    completion = nullptr;
                    if (ownsStream) backend_->destroyStream(stream);
                    stream = nullptr;
                    ownsStream = false;
                    if (cudaStreamLock.owns_lock()) cudaStreamLock.unlock();
                } else if (sourceBuffer) {
                    result.buffer = sourceBuffer;
                } else {
                    result.buffer = std::make_shared<const std::vector<std::byte>>(
                        sourcePinned->bytes().begin(), sourcePinned->bytes().end());
                }
            } else if (task->request.destination == MemoryTier::PinnedRam) {
                auto pinned = std::make_shared<PinnedBuffer>(loaded.record.size, backend_);
                if (sourceDevice) {
                    if (cudaStreams_) {
                        cudaStreamLock =
                            std::unique_lock<std::mutex>(cudaTransferStreamMutex_);
                        stream = cudaStreams_->stream(backend::CudaStreamRole::Transfer);
                    } else {
                        stream = backend_->createStream();
                        ownsStream = true;
                    }
                    completion = backend_->createEvent();
                    const auto transferStart = std::chrono::steady_clock::now();
                    backend_->copyFromDevice(pinned->data(), sourceDevice->data(),
                                             pinned->size(), stream);
                    backend_->recordEvent(completion, stream);
                    backend_->waitEvent(completion);
                    backend_->synchronize(stream);
                    result.backendTransferTime =
                        std::chrono::steady_clock::now() - transferStart;
                    result.cudaTransfer = backend_->kind() == backend::BackendKind::Cuda;
                    result.backendBandwidthGiBs =
                        bandwidthGiBs(loaded.record.size, result.backendTransferTime);
                    backend_->destroyEvent(completion);
                    completion = nullptr;
                    if (ownsStream) backend_->destroyStream(stream);
                    stream = nullptr;
                    ownsStream = false;
                    if (cudaStreamLock.owns_lock()) cudaStreamLock.unlock();
                } else {
                    const auto copyStart = std::chrono::steady_clock::now();
                    if (sourcePinned) {
                        std::memcpy(pinned->data(), sourcePinned->data(), pinned->size());
                    } else {
                        std::memcpy(pinned->data(), sourceBuffer->data(), pinned->size());
                    }
                    result.ramCopyTime = std::chrono::steady_clock::now() - copyStart;
                }
                result.usedPinnedMemory = pinned->isPinned();
                result.pinnedBuffer = std::move(pinned);
            } else if (task->request.destination == MemoryTier::Vram) {
                if (sourceDevice) {
                    result.deviceBuffer = sourceDevice;
                    result.usedPinnedMemory = false;
                    result.completionOrder =
                        nextCompletion_.fetch_add(1, std::memory_order_relaxed);
                    result.elapsed = std::chrono::steady_clock::now() - start;
                    if (task->request.callback) task->request.callback(result);
                    task->promise.set_value(std::move(result));
                    continue;
                }
                auto staging = sourcePinned;
                if (!staging) {
                    staging = std::make_shared<PinnedBuffer>(loaded.record.size, backend_);
                    const auto copyStart = std::chrono::steady_clock::now();
                    std::memcpy(staging->data(), sourceBuffer->data(), staging->size());
                    result.ramCopyTime = std::chrono::steady_clock::now() - copyStart;
                }
                auto device = memoryPool_
                                  ? memoryPool_->allocateDeviceBuffer(loaded.record.size)
                                  : std::make_shared<backend::DeviceBuffer>(
                                        backend_, loaded.record.size);
                if (cudaStreams_) {
                    constexpr int kPrefetchPriority = 300;
                    const bool isPrefetch = task->request.priority == kPrefetchPriority;
                    cudaStreamLock = std::unique_lock<std::mutex>(
                        isPrefetch ? cudaPrefetchStreamMutex_
                                   : cudaTransferStreamMutex_);
                    const auto role = isPrefetch ? backend::CudaStreamRole::Prefetch
                                                 : backend::CudaStreamRole::Transfer;
                    stream = cudaStreams_->stream(role);
                } else {
                    stream = backend_->createStream();
                    ownsStream = true;
                }
                completion = backend_->createEvent();
                const auto transferStart = std::chrono::steady_clock::now();
                backend_->copyToDevice(device->data(), staging->data(), staging->size(), stream);
                backend_->recordEvent(completion, stream);
                backend_->waitEvent(completion);
                backend_->synchronize(stream);
                result.backendTransferTime =
                    std::chrono::steady_clock::now() - transferStart;
                result.ramToVramBytes = loaded.record.size;
                result.cudaTransfer = backend_->kind() == backend::BackendKind::Cuda;
                result.backendBandwidthGiBs =
                    bandwidthGiBs(result.ramToVramBytes, result.backendTransferTime);
                result.usedPinnedMemory = staging->isPinned();
                result.deviceBuffer = std::move(device);
                backend_->destroyEvent(completion);
                completion = nullptr;
                if (ownsStream) backend_->destroyStream(stream);
                stream = nullptr;
                ownsStream = false;
                if (cudaStreamLock.owns_lock()) cudaStreamLock.unlock();
            } else {
                throw std::invalid_argument("transfer destination cannot be NVMe");
            }
            if (task->cancelled->load(std::memory_order_relaxed)) {
                task->promise.set_value(cancelledResult(start));
                continue;
            }
            result.completionOrder = nextCompletion_.fetch_add(1, std::memory_order_relaxed);
            result.elapsed = std::chrono::steady_clock::now() - start;
            if (task->request.callback) task->request.callback(result);
            task->promise.set_value(std::move(result));
        } catch (...) {
            if (completion != nullptr) backend_->destroyEvent(completion);
            if (stream != nullptr && ownsStream) backend_->destroyStream(stream);
            task->promise.set_exception(std::current_exception());
        }
    }
}

TransferResult
TransferManager::cancelledResult(std::chrono::steady_clock::time_point start) {
    TransferResult result;
    result.status = TransferStatus::Cancelled;
    result.elapsed = std::chrono::steady_clock::now() - start;
    return result;
}

} // namespace hypermoe
