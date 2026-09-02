#include "backend/cuda/CudaStreamManager.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::backend {

std::string_view toString(CudaStreamRole role) noexcept {
    switch (role) {
    case CudaStreamRole::Compute: return "COMPUTE";
    case CudaStreamRole::Transfer: return "TRANSFER";
    case CudaStreamRole::Prefetch: return "PREFETCH";
    }
    return "UNKNOWN";
}

CudaStreamManager::CudaStreamManager(std::shared_ptr<CudaRuntime> runtime)
    : runtime_(std::move(runtime)) {
    if (!runtime_) throw std::invalid_argument("CudaStreamManager requires a runtime");
    if (!runtime_->available()) return;
    try {
        streams_[index(CudaStreamRole::Compute)] = runtime_->createStream();
        streams_[index(CudaStreamRole::Transfer)] = runtime_->createStream();
        streams_[index(CudaStreamRole::Prefetch)] = runtime_->createStream();
    } catch (...) {
        shutdown();
        throw;
    }
}

CudaStreamManager::~CudaStreamManager() { shutdown(); }

bool CudaStreamManager::available() const noexcept {
    return runtime_ && runtime_->available() &&
           streams_[index(CudaStreamRole::Compute)] != nullptr;
}

StreamHandle CudaStreamManager::stream(CudaStreamRole role) const noexcept {
    return streams_[index(role)];
}

void CudaStreamManager::synchronize(CudaStreamRole role) {
    if (!available()) throw std::runtime_error("CUDA streams are unavailable");
    runtime_->synchronize(stream(role));
}

void CudaStreamManager::synchronizeAll() {
    if (!available()) throw std::runtime_error("CUDA streams are unavailable");
    for (const auto streamHandle : streams_) runtime_->synchronize(streamHandle);
}

void CudaStreamManager::shutdown() noexcept {
    if (!runtime_) return;
    for (auto& streamHandle : streams_) {
        runtime_->destroyStream(streamHandle);
        streamHandle = nullptr;
    }
}

} // namespace hypermoe::backend
