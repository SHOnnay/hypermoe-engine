#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace hypermoe::storage {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    void open(const std::filesystem::path& path);
    void close() noexcept;

    [[nodiscard]] bool isOpen() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::span<const std::byte>
    view(std::uint64_t offset, std::uint64_t length) const;

private:
#ifdef _WIN32
    void* fileHandle_{reinterpret_cast<void*>(-1)};
    void* mappingHandle_{};
#else
    int fileDescriptor_{-1};
#endif
    const std::byte* data_{};
    std::size_t size_{};
};

} // namespace hypermoe::storage
