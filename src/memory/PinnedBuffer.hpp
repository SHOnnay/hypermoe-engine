#pragma once

#include "backend/Backend.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace hypermoe {

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t sizeBytes,
                          std::shared_ptr<backend::ComputeBackend> backend = {});
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void reset() noexcept;
    [[nodiscard]] std::byte* data() noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool isPinned() const noexcept;
    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

private:
    std::shared_ptr<backend::ComputeBackend> backend_;
    std::byte* data_{};
    std::size_t size_{};
    bool pinned_{};
};

} // namespace hypermoe
