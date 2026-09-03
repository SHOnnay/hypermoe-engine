#include "tensor/precision/DTypeConverter.hpp"

#include "tensor/backend/TensorBackend.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace hypermoe::tensor::precision {
namespace {

float fp16ToFloat(std::uint16_t value) noexcept {
    const auto sign = (value & 0x8000U) != 0 ? -1.0F : 1.0F;
    const auto exponent = static_cast<unsigned>((value >> 10U) & 0x1fU);
    const auto fraction = static_cast<unsigned>(value & 0x03ffU);
    if (exponent == 0) {
        return fraction == 0 ? std::copysign(0.0F, sign)
                             : sign * std::ldexp(static_cast<float>(fraction), -24);
    }
    if (exponent == 0x1fU) {
        if (fraction == 0) return std::copysign(std::numeric_limits<float>::infinity(), sign);
        const std::uint32_t bits = (value & 0x8000U) != 0 ? 0xffc00000U : 0x7fc00000U;
        return std::bit_cast<float>(bits);
    }
    return sign * std::ldexp(1.0F + static_cast<float>(fraction) / 1024.0F,
                             static_cast<int>(exponent) - 15);
}

float bf16ToFloat(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

} // namespace

std::vector<float> DTypeConverter::toFp32(
    std::span<const std::byte> source, DType sourceType) {
    const auto width = sizeOf(sourceType);
    if (width == 0 || source.empty() || source.size() % width != 0) {
        throw std::invalid_argument("dtype conversion source range is invalid");
    }
    if (sourceType == DType::INT8) {
        throw std::invalid_argument("INT8 conversion requires quantization policy metadata");
    }
    std::vector<float> result(source.size() / width);
    if (sourceType == DType::FP32) {
        if constexpr (std::endian::native == std::endian::little) {
            std::memcpy(result.data(), source.data(), source.size());
        } else {
            for (std::size_t index = 0; index < result.size(); ++index) {
                const auto offset = index * 4;
                const auto bits =
                    std::to_integer<std::uint32_t>(source[offset]) |
                    (std::to_integer<std::uint32_t>(source[offset + 1]) << 8U) |
                    (std::to_integer<std::uint32_t>(source[offset + 2]) << 16U) |
                    (std::to_integer<std::uint32_t>(source[offset + 3]) << 24U);
                result[index] = std::bit_cast<float>(bits);
            }
        }
        return result;
    }
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto offset = index * 2;
        const auto bits = static_cast<std::uint16_t>(
            std::to_integer<std::uint16_t>(source[offset]) |
            (std::to_integer<std::uint16_t>(source[offset + 1]) << 8U));
        result[index] = sourceType == DType::FP16 ? fp16ToFloat(bits)
                                                  : bf16ToFloat(bits);
    }
    return result;
}

Tensor DTypeConverter::toFp32Tensor(
    TensorView source, TensorBackend& destinationBackend) {
    [[maybe_unused]] const auto owner = source.lockOwner();
    if (!owner || !source || !source.isContiguous() ||
        source.device() != Device::cpu() ||
        destinationBackend.device() != Device::cpu()) {
        throw std::invalid_argument(
            "reference dtype conversion requires a live contiguous CPU tensor");
    }
    const auto bytes = std::span<const std::byte>(
        static_cast<const std::byte*>(source.data()), source.bytes());
    auto values = toFp32(bytes, source.dtype());
    if (values.size() != source.shape().elementCount()) {
        throw std::invalid_argument("dtype conversion shape does not match storage");
    }
    auto output = destinationBackend.allocateTensor(source.shape(), DType::FP32);
    std::memcpy(output.data(), values.data(), values.size() * sizeof(float));
    return output;
}

} // namespace hypermoe::tensor::precision
