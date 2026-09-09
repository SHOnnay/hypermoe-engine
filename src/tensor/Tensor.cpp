#include "tensor/Tensor.hpp"

#include "tensor/TensorView.hpp"
#include "tensor/TensorError.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::tensor {
namespace {

void validateStorageMetadata(DType dtype, Device device, const void* data) {
    if (!isValid(dtype)) throw TensorError("tensor dtype is invalid");
    if (!isValid(device.type) || device.ordinal < 0) {
        throw TensorError("tensor device metadata is invalid");
    }
    if (device.type == DeviceType::CPU && device.ordinal != 0) {
        throw TensorError("CPU tensors must use device ordinal zero");
    }
    const auto alignment = alignmentOf(dtype);
    if (data != nullptr &&
        reinterpret_cast<std::uintptr_t>(data) % alignment != 0) {
        throw TensorError("tensor storage is not aligned for its dtype");
    }
}

} // namespace

Tensor Tensor::fromStorage(Shape shape,
                           DType dtype,
                           Device device,
                           void* data,
                           std::size_t storageBytes,
                           std::shared_ptr<void> owner) {
    return {std::move(shape), dtype, device, data, storageBytes, std::move(owner)};
}

Tensor Tensor::fromDeviceBuffer(Shape shape,
                                DType dtype,
                                Device device,
                                std::shared_ptr<backend::DeviceBuffer> buffer) {
    if (!buffer || !*buffer || !buffer->backend()) {
        throw TensorError("tensor device buffer is empty");
    }
    validateStorageMetadata(dtype, device, buffer->data());
    const auto backendKind = buffer->backend()->kind();
    if ((device.type == DeviceType::CPU && backendKind != backend::BackendKind::Cpu) ||
        (device.type == DeviceType::CUDA &&
         (backendKind != backend::BackendKind::Cuda ||
          buffer->backend()->deviceOrdinal() != device.ordinal))) {
        throw TensorError("tensor device metadata does not match its buffer");
    }
    void* data = buffer->data();
    const auto storageBytes = buffer->size();
    return {std::move(shape), dtype, device, data, storageBytes,
            std::static_pointer_cast<void>(std::move(buffer))};
}

Tensor::Tensor(Shape shape,
               DType dtype,
               Device device,
               void* data,
               std::size_t storageBytes,
               std::shared_ptr<void> owner)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      data_(data),
      bytes_(logicalBytes(shape_, dtype_)),
      storageBytes_(storageBytes),
      owner_(std::move(owner)) {
    validateStorageMetadata(dtype_, device_, data_);
    if (data_ == nullptr || !owner_) {
        throw TensorError("tensor storage and owner must be present");
    }
    if (storageBytes_ < requiredBytes(shape_, dtype_)) {
        throw TensorError("tensor storage is smaller than its metadata requires");
    }
}

const Shape& Tensor::shape() const noexcept { return shape_; }
DType Tensor::dtype() const noexcept { return dtype_; }
Device Tensor::device() const noexcept { return device_; }
void* Tensor::data() noexcept { return data_; }
const void* Tensor::data() const noexcept { return data_; }
std::size_t Tensor::bytes() const noexcept { return bytes_; }
std::size_t Tensor::storageBytes() const noexcept { return storageBytes_; }
bool Tensor::isContiguous() const noexcept { return shape_.isContiguous(); }
bool Tensor::valid() const noexcept { return data_ != nullptr && owner_ && bytes_ != 0; }
Tensor::operator bool() const noexcept { return valid(); }

Tensor Tensor::reshape(Shape shape) const {
    if (!valid()) throw std::logic_error("cannot reshape an empty tensor");
    if (!isContiguous() || !shape.isContiguous()) {
        throw TensorError("reshape currently requires contiguous tensors");
    }
    if (shape.elementCount() != shape_.elementCount()) {
        throw TensorError("reshape must preserve the element count");
    }
    return {std::move(shape), dtype_, device_, data_, storageBytes_, owner_};
}

TensorView Tensor::view() & { return TensorView(*this); }

TensorView Tensor::view() const & { return TensorView(*this); }

std::size_t Tensor::requiredBytes(const Shape& shape, DType dtype) {
    const auto elementBytes = sizeOf(dtype);
    if (elementBytes == 0) throw TensorError("tensor dtype is invalid");
    if (shape.storageElementCount() >
        std::numeric_limits<std::size_t>::max() / elementBytes) {
        throw std::overflow_error("tensor byte size overflow");
    }
    return shape.storageElementCount() * elementBytes;
}

std::size_t Tensor::logicalBytes(const Shape& shape, DType dtype) {
    const auto elementBytes = sizeOf(dtype);
    if (elementBytes == 0) throw TensorError("tensor dtype is invalid");
    if (shape.elementCount() > std::numeric_limits<std::size_t>::max() / elementBytes) {
        throw std::overflow_error("tensor byte size overflow");
    }
    return shape.elementCount() * elementBytes;
}

} // namespace hypermoe::tensor
