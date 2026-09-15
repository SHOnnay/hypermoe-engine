#pragma once

#include "backend/Backend.hpp"
#include "profiling/GpuOperation.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace hypermoe { class Profiler; }
namespace hypermoe::profiling {

// Event operations are injected so completion/lifetime/error paths can be
// tested on CPU. Production binds these to the existing CudaRuntime, not a
// second execution or transfer backend.
struct GpuEventApi {
    std::function<backend::EventHandle(bool)> create;
    std::function<void(backend::EventHandle, backend::StreamHandle)> record;
    std::function<bool(backend::EventHandle)> ready;
    std::function<float(backend::EventHandle, backend::EventHandle)> elapsedMs;
    std::function<void(backend::EventHandle)> destroy;
    std::function<void(backend::StreamHandle)> synchronize;
};

class GpuEventQueue {
public:
    class Scope {
    public:
        Scope() = default;
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;
        void finish();
    private:
        friend class GpuEventQueue;
        GpuEventQueue* queue_{};
        GpuOperation operation_{};
        backend::StreamHandle stream_{};
        backend::EventHandle start_{};
        backend::EventHandle end_{};
        std::vector<std::shared_ptr<void>> owners_;
    };

    explicit GpuEventQueue(GpuEventApi api, std::shared_ptr<Profiler> profiler = {});
    ~GpuEventQueue();
    [[nodiscard]] Scope begin(GpuOperation operation, backend::StreamHandle stream,
                               std::vector<std::shared_ptr<void>> owners = {});
    // Never waits. Query failures propagate and retain pending storage.
    void collect();
    [[nodiscard]] std::size_t pending() const;
private:
    struct Record {
        GpuOperation operation;
        backend::StreamHandle stream;
        backend::EventHandle start;
        backend::EventHandle end;
        std::vector<std::shared_ptr<void>> owners;
    };
    void finish(Scope& scope);
    void abort(Scope& scope) noexcept;
    GpuEventApi api_;
    std::shared_ptr<Profiler> profiler_;
    mutable std::mutex mutex_;
    std::vector<Record> pending_;
};
} // namespace hypermoe::profiling
