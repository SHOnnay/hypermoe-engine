#include "backend/Backend.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::backend {

DeviceBuffer::DeviceBuffer(std::shared_ptr<ComputeBackend> backend,
                           std::size_t sizeBytes)
    : backend_(std::move(backend)), size_(sizeBytes) {
    if (!backend_) {
        throw std::invalid_argument("DeviceBuffer requires a backend");
    }
    if (sizeBytes == 0) {
        throw std::invalid_argument("DeviceBuffer size must be nonzero");
    }
    data_ = backend_->allocate(sizeBytes);
}

DeviceBuffer::~DeviceBuffer() {
    reset();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : backend_(std::move(other.backend_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        backend_ = std::move(other.backend_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void DeviceBuffer::reset() noexcept {
    if (data_ != nullptr && backend_) {
        backend_->free(data_);
    }
    data_ = nullptr;
    size_ = 0;
    backend_.reset();
}

void* DeviceBuffer::data() noexcept { return data_; }
const void* DeviceBuffer::data() const noexcept { return data_; }
std::size_t DeviceBuffer::size() const noexcept { return size_; }
DeviceBuffer::operator bool() const noexcept { return data_ != nullptr; }
const std::shared_ptr<ComputeBackend>& DeviceBuffer::backend() const noexcept {
    return backend_;
}

} // namespace hypermoe::backend
