#include "memory/PinnedBuffer.hpp"

#include <new>
#include <stdexcept>
#include <utility>

namespace hypermoe {
namespace {
constexpr std::align_val_t kFallbackAlignment{64};
}

PinnedBuffer::PinnedBuffer(std::size_t sizeBytes,
                           std::shared_ptr<backend::ComputeBackend> backend)
    : backend_(std::move(backend)), size_(sizeBytes) {
    if (sizeBytes == 0) {
        throw std::invalid_argument("PinnedBuffer size must be nonzero");
    }
    if (backend_ && backend_->supportsPinnedMemory()) {
        data_ = static_cast<std::byte*>(backend_->allocatePinned(sizeBytes));
        pinned_ = true;
    } else {
        data_ = static_cast<std::byte*>(::operator new(sizeBytes, kFallbackAlignment));
    }
}

PinnedBuffer::~PinnedBuffer() { reset(); }

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : backend_(std::move(other.backend_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      pinned_(std::exchange(other.pinned_, false)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        backend_ = std::move(other.backend_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        pinned_ = std::exchange(other.pinned_, false);
    }
    return *this;
}

void PinnedBuffer::reset() noexcept {
    if (data_ != nullptr) {
        if (pinned_ && backend_) backend_->freePinned(data_);
        else ::operator delete(data_, kFallbackAlignment);
    }
    data_ = nullptr;
    size_ = 0;
    pinned_ = false;
    backend_.reset();
}

std::byte* PinnedBuffer::data() noexcept { return data_; }
const std::byte* PinnedBuffer::data() const noexcept { return data_; }
std::size_t PinnedBuffer::size() const noexcept { return size_; }
bool PinnedBuffer::isPinned() const noexcept { return pinned_; }
std::span<std::byte> PinnedBuffer::bytes() noexcept { return {data_, size_}; }
std::span<const std::byte> PinnedBuffer::bytes() const noexcept { return {data_, size_}; }

} // namespace hypermoe
