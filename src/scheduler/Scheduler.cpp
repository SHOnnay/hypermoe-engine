#include "scheduler/Scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hypermoe::scheduler {

ScheduleHandle::ScheduleHandle(std::shared_future<ScheduleResult> future,
                               std::shared_ptr<std::atomic_bool> cancelled)
    : future_(std::move(future)), cancelled_(std::move(cancelled)) {}

void ScheduleHandle::cancel() noexcept {
    if (cancelled_) cancelled_->store(true, std::memory_order_relaxed);
}

bool ScheduleHandle::valid() const noexcept { return future_.valid(); }

const std::shared_future<ScheduleResult>& ScheduleHandle::future() const noexcept {
    return future_;
}

Scheduler::Scheduler(std::shared_ptr<TransferManager> transfers,
                     std::shared_ptr<Profiler> profiler,
                     std::size_t workerCount)
    : transfers_(std::move(transfers)), profiler_(std::move(profiler)) {
    if (!transfers_) throw std::invalid_argument("Scheduler requires a TransferManager");
    if (workerCount == 0) throw std::invalid_argument("Scheduler requires a worker");
    workers_.reserve(workerCount);
    for (std::size_t index = 0; index < workerCount; ++index) {
        workers_.emplace_back(&Scheduler::workerLoop, this);
    }
}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::registerExpert(LayerId layerId,
                               ExpertId id,
                               MemoryTier location) {
    states_.registerExpert(layerId, id, location);
}

void Scheduler::registerExpert(ExpertId id, MemoryTier location) {
    registerExpert(0, id, location);
}

ScheduleHandle Scheduler::schedule(ScheduleRequest request, ScheduleCallback callback) {
    publish(RuntimeEventType::ExpertRequested, request);
    const bool isPrefetch = request.priority == TransferPriority::PredictedNextLayer;
    std::shared_ptr<Task> task;
    bool immediatelyReady = false;
    bool notifyWorker = false;
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) throw std::runtime_error("Scheduler is shutting down");
        const auto expertKey = key(request.layerId, request.expertId);
        const auto initialState = states_.snapshot(request.layerId, request.expertId);
        const auto existing = pendingByExpert_.find(expertKey);

        if (profiler_ && isPrefetch) profiler_->recordPrefetchRequest();
        if (profiler_ && request.priority == TransferPriority::ActiveInference) {
            if (prefetchedReady_.erase(expertKey) != 0 &&
                initialState.currentLocation == request.destination &&
                initialState.state == ExpertLifecycleState::Ready) {
                profiler_->recordPrefetchHit();
            } else if (existing != pendingByExpert_.end() &&
                       existing->second->prefetchRequest) {
                if (initialState.currentLocation == request.destination &&
                    initialState.state == ExpertLifecycleState::Ready) {
                    existing->second->activeConsumer = true;
                    profiler_->recordPrefetchHit();
                } else {
                    profiler_->recordPrefetchMiss();
                }
            }
        }

        immediatelyReady =
            !request.eviction && initialState.currentLocation == request.destination &&
            (initialState.state == ExpertLifecycleState::Ready ||
             initialState.state == ExpertLifecycleState::InUse);
        if (immediatelyReady) {
            if (isPrefetch) prefetchedReady_.insert(expertKey);
        } else if (existing != pendingByExpert_.end() &&
                   existing->second->request.destination != request.destination) {
            throw std::logic_error("expert already has a different pending target");
        } else if (existing != pendingByExpert_.end()) {
            task = existing->second;
            if (callback) task->callbacks.push_back(std::move(callback));
            if (request.priority == TransferPriority::ActiveInference) {
                task->activeConsumer = true;
            }
            if (static_cast<int>(request.priority) >
                    static_cast<int>(task->request.priority) &&
                !task->started) {
                task->request.priority = request.priority;
                ++task->generation;
                queue_.push({task, request.priority, nextSequence_++, task->generation});
                notifyWorker = true;
            }
        } else {
            if (request.eviction) {
                states_.beginEviction(request.layerId, request.expertId,
                                      request.destination);
            }
            else {
                states_.request(request.layerId, request.expertId,
                                request.destination);
                states_.markQueued(request.layerId, request.expertId);
            }
            task = std::make_shared<Task>();
            task->request = request;
            if (callback) task->callbacks.push_back(std::move(callback));
            task->cancelled = std::make_shared<std::atomic_bool>(false);
            task->future = task->promise.get_future().share();
            task->enqueuedAt = std::chrono::steady_clock::now();
            task->prefetchRequest = isPrefetch;
            task->activeConsumer =
                task->request.priority == TransferPriority::ActiveInference;
            pendingByExpert_[expertKey] = task;
            queue_.push({task, task->request.priority, nextSequence_++, task->generation});
            if (profiler_) profiler_->observeTransferQueueDepth(queue_.size());
            notifyWorker = true;
        }
    }
    if (immediatelyReady) return readyResult(request, std::move(callback));
    if (notifyWorker) ready_.notify_one();
    return {task->future, task->cancelled};
}

ScheduleHandle Scheduler::prefetch(const PredictedExpertRequest& prediction,
                                   ScheduleCallback callback) {
    ScheduleRequest request;
    request.layerId = prediction.layerId;
    request.expertId = prediction.expertId;
    request.source = MemoryTier::Nvme;
    request.destination = MemoryTier::Vram;
    request.priority = TransferPriority::PredictedNextLayer;
    return schedule(std::move(request), std::move(callback));
}

std::vector<ScheduleHandle>
Scheduler::prefetch(const Prefetcher& prefetcher,
                    const PredictionInput& input,
                    ScheduleCallback callback) {
    const auto predictions = prefetcher.predict(input);
    std::vector<ScheduleHandle> handles;
    handles.reserve(predictions.size());
    for (const auto& prediction : predictions) {
        handles.push_back(prefetch(prediction, callback));
    }
    return handles;
}

void Scheduler::acquire(LayerId layerId, ExpertId id) {
    states_.acquire(layerId, id);
}
void Scheduler::acquire(ExpertId id) { acquire(0, id); }
void Scheduler::release(LayerId layerId, ExpertId id) {
    states_.release(layerId, id);
}
void Scheduler::release(ExpertId id) { release(0, id); }
ExpertState Scheduler::state(LayerId layerId, ExpertId id) const {
    return states_.snapshot(layerId, id);
}
ExpertState Scheduler::state(ExpertId id) const { return state(0, id); }

std::size_t Scheduler::pending() const {
    std::scoped_lock lock(mutex_);
    return pendingByExpert_.size();
}

RuntimeEventBus& Scheduler::events() noexcept { return events_; }
const ExpertResidencyStateMachine& Scheduler::states() const noexcept { return states_; }

void Scheduler::shutdown() {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) return;
        stopping_ = true;
        for (const auto& [id, task] : pendingByExpert_) {
            (void)id;
            task->cancelled->store(true, std::memory_order_relaxed);
        }
    }
    ready_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
}

bool Scheduler::HigherPriority::operator()(const QueueEntry& left,
                                           const QueueEntry& right) const noexcept {
    if (left.priority != right.priority) {
        return static_cast<int>(left.priority) < static_cast<int>(right.priority);
    }
    return left.sequence > right.sequence;
}

ScheduleHandle Scheduler::readyResult(const ScheduleRequest& request,
                                      ScheduleCallback callback) {
    std::promise<ScheduleResult> promise;
    auto future = promise.get_future().share();
    ScheduleResult result;
    result.success = true;
    result.state = states_.snapshot(request.layerId, request.expertId);
    if (callback) {
        try {
            callback(result);
        } catch (...) {
            // Consumer callback failures do not change readiness.
        }
    }
    promise.set_value(result);
    publish(RuntimeEventType::ExpertReady, request);
    return {std::move(future), std::make_shared<std::atomic_bool>(false)};
}

void Scheduler::workerLoop() {
    while (true) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) return;
            while (!queue_.empty()) {
                auto entry = queue_.top();
                queue_.pop();
                if (!entry.task->completed && !entry.task->started &&
                    entry.generation == entry.task->generation) {
                    task = std::move(entry.task);
                    task->started = true;
                    break;
                }
            }
            if (!task) {
                if (stopping_) return;
                continue;
            }
        }

        if (profiler_) {
            profiler_->recordQueueWait(std::chrono::steady_clock::now() - task->enqueuedAt);
        }
        if (task->cancelled->load(std::memory_order_relaxed)) {
            ScheduleResult result;
            result.cancelled = true;
            result.error = "schedule request cancelled before transfer";
            complete(task, std::move(result));
            continue;
        }

        if (!task->request.eviction) {
            states_.markLoading(task->request.layerId, task->request.expertId);
        }
        publish(RuntimeEventType::TransferStarted, task->request);
        const bool cudaTransferExpected =
            transfers_->backendKind() == backend::BackendKind::Cuda &&
            (task->request.source == MemoryTier::Vram ||
             task->request.destination == MemoryTier::Vram);
        if (cudaTransferExpected) {
            publish(RuntimeEventType::CudaTransferStarted, task->request);
        }
        try {
            TransferRequest transfer;
            transfer.layerId = task->request.layerId;
            transfer.expertId = task->request.expertId;
            transfer.source = task->request.source;
            transfer.destination = task->request.destination;
            transfer.priority = static_cast<int>(task->request.priority);
            transfer.hostBuffer = task->request.hostBuffer;
            transfer.sourcePinnedBuffer = task->request.pinnedBuffer;
            transfer.sourceDeviceBuffer = task->request.deviceBuffer;
            auto handle = transfers_->submit(std::move(transfer));
            while (handle.future().wait_for(std::chrono::milliseconds(1)) !=
                   std::future_status::ready) {
                if (task->cancelled->load(std::memory_order_relaxed)) handle.cancel();
            }
            auto transferred = handle.future().get();
            if (transferred.status == TransferStatus::Cancelled) {
                ScheduleResult result;
                result.cancelled = true;
                result.error = "transfer cancelled";
                complete(task, std::move(result));
                continue;
            }
            states_.markReady(task->request.layerId, task->request.expertId,
                              task->request.destination);
            {
                std::scoped_lock lock(mutex_);
                if (task->prefetchRequest && !task->activeConsumer) {
                    prefetchedReady_.insert(
                        key(task->request.layerId, task->request.expertId));
                }
            }
            publish(RuntimeEventType::TransferCompleted, task->request,
                    transferred.elapsed);
            if (transferred.cudaTransfer) {
                publish(RuntimeEventType::CudaTransferCompleted, task->request,
                        transferred.backendTransferTime);
            }
            if (task->request.eviction) {
                publish(RuntimeEventType::CacheEvicted, task->request,
                        transferred.elapsed);
            }
            publish(RuntimeEventType::ExpertReady, task->request);
            ScheduleResult result;
            result.success = true;
            result.transfer = std::move(transferred);
            result.state =
                states_.snapshot(task->request.layerId, task->request.expertId);
            complete(task, std::move(result));
        } catch (const std::exception& error) {
            ScheduleResult result;
            result.error = error.what();
            complete(task, std::move(result));
        } catch (...) {
            ScheduleResult result;
            result.error = "unknown scheduler failure";
            complete(task, std::move(result));
        }
    }
}

void Scheduler::complete(const std::shared_ptr<Task>& task, ScheduleResult result) {
    std::vector<ScheduleCallback> callbacks;
    if (!result.success) {
        states_.markFailed(task->request.layerId, task->request.expertId);
        result.state =
            states_.snapshot(task->request.layerId, task->request.expertId);
    }
    {
        std::scoped_lock lock(mutex_);
        if (task->completed) return;
        task->completed = true;
        const auto pending = pendingByExpert_.find(
            key(task->request.layerId, task->request.expertId));
        if (pending != pendingByExpert_.end() && pending->second == task) {
            pendingByExpert_.erase(pending);
        }
        callbacks = task->callbacks;
        if (profiler_) profiler_->observeTransferQueueDepth(pendingByExpert_.size());
    }
    for (const auto& callback : callbacks) {
        try {
            callback(result);
        } catch (...) {
            // Consumer notification failures do not alter scheduler state.
        }
    }
    if (!result.success) {
        publish(RuntimeEventType::TransferFailed, task->request, {}, result.error);
    }
    task->promise.set_value(std::move(result));
}

void Scheduler::publish(RuntimeEventType type,
                        const ScheduleRequest& request,
                        std::chrono::nanoseconds duration,
                        std::string message) {
    events_.publish({type, request.expertId, request.layerId, request.source,
                     request.destination, std::chrono::steady_clock::now(),
                     duration, std::move(message)});
}

} // namespace hypermoe::scheduler
