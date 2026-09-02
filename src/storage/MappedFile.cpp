#include "storage/MappedFile.hpp"

#include "storage/ExpertIndex.hpp"

#include <limits>
#include <string>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hypermoe::storage {

MappedFile::MappedFile(const std::filesystem::path& path) {
    open(path);
}

MappedFile::~MappedFile() {
    close();
}

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
#ifdef _WIN32
    fileHandle_ = std::exchange(other.fileHandle_, reinterpret_cast<void*>(-1));
    mappingHandle_ = std::exchange(other.mappingHandle_, nullptr);
#else
    fileDescriptor_ = std::exchange(other.fileDescriptor_, -1);
#endif
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
}

void MappedFile::open(const std::filesystem::path& path) {
    if (isOpen()) {
        throw StorageError("mapped file is already open");
    }
#ifdef _WIN32
    const auto file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw StorageError("cannot open mapped file: " + path.string());
    }
    LARGE_INTEGER fileSize{};
    if (!GetFileSizeEx(file, &fileSize) || fileSize.QuadPart <= 0 ||
        static_cast<unsigned long long>(fileSize.QuadPart) >
            static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        CloseHandle(file);
        throw StorageError("invalid mapped file size: " + path.string());
    }
    const auto mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        throw StorageError("cannot create file mapping: " + path.string());
    }
    const auto address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (address == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        throw StorageError("cannot map file: " + path.string());
    }
    fileHandle_ = file;
    mappingHandle_ = mapping;
    data_ = static_cast<const std::byte*>(address);
    size_ = static_cast<std::size_t>(fileSize.QuadPart);
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw StorageError("cannot open mapped file: " + path.string() + ": " +
                           std::strerror(errno));
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        ::close(descriptor);
        throw StorageError("invalid mapped file size: " + path.string());
    }
    void* address = ::mmap(nullptr, static_cast<std::size_t>(status.st_size), PROT_READ,
                           MAP_PRIVATE, descriptor, 0);
    if (address == MAP_FAILED) {
        const auto message = std::string(std::strerror(errno));
        ::close(descriptor);
        throw StorageError("cannot map file: " + path.string() + ": " + message);
    }
    fileDescriptor_ = descriptor;
    data_ = static_cast<const std::byte*>(address);
    size_ = static_cast<std::size_t>(status.st_size);
#endif
}

void MappedFile::close() noexcept {
#ifdef _WIN32
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
    }
    if (mappingHandle_ != nullptr) {
        CloseHandle(mappingHandle_);
    }
    if (fileHandle_ != reinterpret_cast<void*>(-1)) {
        CloseHandle(fileHandle_);
    }
    fileHandle_ = reinterpret_cast<void*>(-1);
    mappingHandle_ = nullptr;
#else
    if (data_ != nullptr) {
        (void)::munmap(const_cast<std::byte*>(data_), size_);
    }
    if (fileDescriptor_ >= 0) {
        (void)::close(fileDescriptor_);
    }
    fileDescriptor_ = -1;
#endif
    data_ = nullptr;
    size_ = 0;
}

bool MappedFile::isOpen() const noexcept {
    return data_ != nullptr;
}

std::size_t MappedFile::size() const noexcept {
    return size_;
}

std::span<const std::byte> MappedFile::bytes() const noexcept {
    return {data_, size_};
}

std::span<const std::byte>
MappedFile::view(std::uint64_t offset, std::uint64_t length) const {
    if (!isOpen()) {
        throw StorageError("mapped file is not open");
    }
    if (offset > size_ || length > static_cast<std::uint64_t>(size_) - offset) {
        throw StorageError("mapped file range is out of bounds");
    }
    return {data_ + static_cast<std::size_t>(offset), static_cast<std::size_t>(length)};
}

} // namespace hypermoe::storage
