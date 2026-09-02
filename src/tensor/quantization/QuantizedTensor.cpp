#include "tensor/quantization/QuantizedTensor.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hypermoe::tensor::quantization {
namespace {

void validateDevice(Device device) {
    if (device.ordinal < 0) throw std::invalid_argument("device ordinal must be nonnegative");
    if (device.type == DeviceType::CPU && device.ordinal != 0) {
        throw std::invalid_argument("CPU quantized tensors must use device ordinal zero");
    }
}

void validateBufferDevice(const backend::DeviceBuffer& buffer, Device device) {
    const auto kind = buffer.backend()->kind();
    if ((device.type == DeviceType::CPU && kind != backend::BackendKind::Cpu) ||
        (device.type == DeviceType::CUDA &&
         (kind != backend::BackendKind::Cuda ||
          buffer.backend()->deviceOrdinal() != device.ordinal))) {
        throw std::invalid_argument(
            "quantized tensor device metadata does not match its buffer");
    }
}

} // namespace

std::string QuantizedTensorMetadata::toJson() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(9)
           << "{\"version\":" << version << ",\"dtype\":\""
           << quantization::toString(dtype) << "\",\"scale\":"
           << parameters.scale << ",\"zero_point\":" << parameters.zeroPoint
           << ",\"shape\":[";
    const auto& dimensions = shape.dimensions();
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) output << ',';
        output << dimensions[index];
    }
    output << "],\"device\":\"" << tensor::toString(device.type)
           << "\",\"device_ordinal\":" << device.ordinal
           << ",\"storage_bytes\":" << storageBytes << '}';
    return output.str();
}

QuantizedTensor QuantizedTensor::fromStorage(
    Shape shape,
    QuantizedDType dtype,
    QuantizationParameters parameters,
    Device device,
    void* data,
    std::size_t storageBytes,
    std::shared_ptr<void> owner) {
    return {std::move(shape), dtype, parameters, device, data, storageBytes,
            std::move(owner)};
}

QuantizedTensor QuantizedTensor::fromDeviceBuffer(
    Shape shape,
    QuantizedDType dtype,
    QuantizationParameters parameters,
    Device device,
    std::shared_ptr<backend::DeviceBuffer> buffer) {
    if (!buffer || !*buffer) throw std::invalid_argument("quantized tensor buffer is empty");
    validateBufferDevice(*buffer, device);
    auto* data = buffer->data();
    const auto storageBytes = buffer->size();
    return {std::move(shape), dtype, parameters, device, data, storageBytes,
            std::static_pointer_cast<void>(std::move(buffer))};
}

QuantizedTensor::QuantizedTensor(Shape shape,
                                 QuantizedDType dtype,
                                 QuantizationParameters parameters,
                                 Device device,
                                 void* data,
                                 std::size_t storageBytes,
                                 std::shared_ptr<void> owner)
    : shape_(std::move(shape)),
      dtype_(dtype),
      parameters_(parameters),
      device_(device),
      data_(data),
      bytes_(storageSizeBytes(shape_, dtype_)),
      storageBytes_(storageBytes),
      owner_(std::move(owner)) {
    validateParameters(dtype_, parameters_);
    validateDevice(device_);
    if (data_ == nullptr || !owner_) {
        throw std::invalid_argument("quantized tensor storage and owner must be present");
    }
    if (storageBytes_ < bytes_) {
        throw std::invalid_argument("quantized tensor storage is too small");
    }
}

const Shape& QuantizedTensor::shape() const noexcept { return shape_; }
QuantizedDType QuantizedTensor::dtype() const noexcept { return dtype_; }
const QuantizationParameters& QuantizedTensor::parameters() const noexcept {
    return parameters_;
}
Device QuantizedTensor::device() const noexcept { return device_; }
void* QuantizedTensor::data() noexcept { return data_; }
const void* QuantizedTensor::data() const noexcept { return data_; }
std::size_t QuantizedTensor::bytes() const noexcept { return bytes_; }
std::size_t QuantizedTensor::storageBytes() const noexcept { return storageBytes_; }
bool QuantizedTensor::valid() const noexcept {
    return data_ != nullptr && owner_ && bytes_ != 0;
}
QuantizedTensor::operator bool() const noexcept { return valid(); }

QuantizedTensorMetadata QuantizedTensor::metadata() const {
    if (!valid()) throw std::logic_error("empty quantized tensor has no metadata");
    return {1, dtype_, parameters_, shape_, device_, bytes_};
}

} // namespace hypermoe::tensor::quantization
