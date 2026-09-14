#include "tensor/TensorView.hpp"
#include "tensor/TensorError.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::tensor {
namespace {

void validateDevice(Device device) {
    if (!isValid(device.type) || device.ordinal < 0) {
        throw TensorError("tensor view device metadata is invalid");
    }
    if (device.type == DeviceType::CPU && device.ordinal != 0) {
        throw TensorError("CPU tensor views must use device ordinal zero");
    }
}

void validateBufferDevice(const backend::DeviceBuffer& buffer, Device device) {
    validateDevice(device);
    if (!buffer.backend()) throw TensorError("tensor view buffer has no backend");
    const auto kind = buffer.backend()->kind();
    if ((device.type == DeviceType::CPU && kind != backend::BackendKind::Cpu) ||
        (device.type == DeviceType::CUDA &&
         (kind != backend::BackendKind::Cuda ||
          buffer.backend()->deviceOrdinal() != device.ordinal))) {
        throw TensorError("tensor view device metadata does not match its buffer");
    }
}

} // namespace

TensorView::TensorView(Tensor& tensor)
    : shape_(tensor.shape_),
      dtype_(tensor.dtype_),
      device_(tensor.device_),
      data_(tensor.data_),
      mutableData_(tensor.data_),
      bytes_(tensor.bytes_),
      storageBytes_(tensor.storageBytes_),
      lifetime_(tensor.owner_) {}

TensorView::TensorView(const Tensor& tensor)
    : shape_(tensor.shape_),
      dtype_(tensor.dtype_),
      device_(tensor.device_),
      data_(tensor.data_),
      bytes_(tensor.bytes_),
      storageBytes_(tensor.storageBytes_),
      lifetime_(tensor.owner_) {}

TensorView TensorView::fromDeviceBuffer(
    const Shape& shape,
    DType dtype,
    Device device,
    const std::shared_ptr<backend::DeviceBuffer>& buffer,
    bool writable) {
    if (!buffer || !*buffer) throw TensorError("tensor view buffer is empty");
    validateBufferDevice(*buffer, device);
    auto owner = std::static_pointer_cast<void>(buffer);
    return {shape, dtype, device, buffer->data(), writable ? buffer->data() : nullptr,
            buffer->size(), owner};
}

TensorView TensorView::fromHostBuffer(
    const Shape& shape,
    DType dtype,
    const std::shared_ptr<const std::vector<std::byte>>& buffer,
    bool writable) {
    if (!buffer || buffer->empty()) throw TensorError("tensor view host buffer is empty");
    validateDevice(Device::cpu());
    // Alias the buffer's ownership: the view must not own (and later delete)
    // the vector itself — that double-frees the lease's buffer (heap corruption).
    auto owner = std::shared_ptr<void>(buffer, const_cast<std::byte*>(buffer->data()));
    std::weak_ptr<void> lifetime(owner);
    void* mutableData = writable ? const_cast<std::byte*>(buffer->data()) : nullptr;
    return {shape, dtype, Device::cpu(), buffer->data(), mutableData, buffer->size(), lifetime};
}

TensorView::TensorView(Shape shape,
                       DType dtype,
                       Device device,
                       const void* data,
                       void* mutableData,
                       std::size_t storageBytes,
                       std::weak_ptr<void> lifetime)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      data_(data),
      mutableData_(mutableData),
      bytes_(logicalBytes(shape_, dtype_)),
      storageBytes_(storageBytes),
      lifetime_(std::move(lifetime)) {
    validateDevice(device_);
    if (data_ == nullptr || lifetime_.expired()) {
        throw TensorError("tensor view storage must be alive");
    }
    const auto alignment = alignmentOf(dtype_);
    if (alignment == 0 ||
        reinterpret_cast<std::uintptr_t>(data_) % alignment != 0) {
        throw TensorError("tensor view storage is not aligned for its dtype");
    }
    if (storageBytes_ < requiredBytes(shape_, dtype_)) {
        throw TensorError("tensor view storage is smaller than its metadata requires");
    }
}

const Shape& TensorView::shape() const noexcept { return shape_; }
DType TensorView::dtype() const noexcept { return dtype_; }
Device TensorView::device() const noexcept { return device_; }
const void* TensorView::data() const noexcept {
    return lifetime_.expired() ? nullptr : data_;
}
void* TensorView::mutableData() const noexcept {
    return lifetime_.expired() ? nullptr : mutableData_;
}
std::size_t TensorView::bytes() const noexcept { return bytes_; }
std::size_t TensorView::storageBytes() const noexcept { return storageBytes_; }
bool TensorView::isContiguous() const noexcept { return shape_.isContiguous(); }
bool TensorView::writable() const noexcept {
    return mutableData_ != nullptr && !lifetime_.expired();
}
bool TensorView::valid() const noexcept {
    return data_ != nullptr && bytes_ != 0 && !lifetime_.expired();
}
TensorView::operator bool() const noexcept { return valid(); }
std::shared_ptr<void> TensorView::lockOwner() const noexcept {
    return lifetime_.lock();
}

TensorView TensorView::reshape(Shape shape) const {
    if (!valid()) throw std::logic_error("cannot reshape an invalid tensor view");
    if (!isContiguous() || !shape.isContiguous()) {
        throw TensorError("reshape currently requires contiguous tensor views");
    }
    if (shape.elementCount() != shape_.elementCount()) {
        throw TensorError("reshape must preserve the element count");
    }
    return {std::move(shape), dtype_, device_, data_, mutableData_, storageBytes_, lifetime_};
}

TensorView TensorView::sliceBytes(std::size_t offsetBytes,
                                  Shape shape,
                                  DType dtype) const {
    if (!valid()) throw std::logic_error("cannot slice an invalid tensor view");
    const auto alignment = sizeOf(dtype);
    if (alignment == 0) throw TensorError("tensor view dtype is invalid");
    if (offsetBytes % alignment != 0) {
        throw TensorError("tensor view slice offset is not element-aligned");
    }
    if (offsetBytes > storageBytes_) {
        throw std::out_of_range("tensor view slice offset exceeds storage");
    }
    const auto remaining = storageBytes_ - offsetBytes;
    const auto sliceStorageBytes = requiredBytes(shape, dtype);
    if (remaining < sliceStorageBytes) {
        throw std::out_of_range("tensor view slice exceeds storage");
    }
    const auto* base = static_cast<const std::byte*>(data_);
    auto* mutableBase = static_cast<std::byte*>(mutableData_);
    return {std::move(shape), dtype, device_, base + offsetBytes,
            mutableBase == nullptr ? nullptr : mutableBase + offsetBytes,
            sliceStorageBytes, lifetime_};
}

std::size_t TensorView::requiredBytes(const Shape& shape, DType dtype) {
    const auto elementBytes = sizeOf(dtype);
    if (elementBytes == 0 ||
        shape.storageElementCount() >
            std::numeric_limits<std::size_t>::max() / elementBytes) {
        throw std::overflow_error("tensor view storage size overflow");
    }
    return shape.storageElementCount() * elementBytes;
}

std::size_t TensorView::logicalBytes(const Shape& shape, DType dtype) {
    const auto elementBytes = sizeOf(dtype);
    if (elementBytes == 0 ||
        shape.elementCount() > std::numeric_limits<std::size_t>::max() / elementBytes) {
        throw std::overflow_error("tensor view byte size overflow");
    }
    return shape.elementCount() * elementBytes;
}

} // namespace hypermoe::tensor
