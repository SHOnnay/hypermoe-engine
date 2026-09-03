#include "tools/model_convert/WeightConverter.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace hypermoe::conversion {

ConvertedWeight WeightConverter::convert(
    std::span<const std::byte> source,
    const tensor::Shape& shape,
    tensor::DType dtype,
    models::TensorLayout sourceLayout,
    models::TensorLayout targetLayout) {
    if (shape.rank() != 2) throw std::invalid_argument("weight conversion requires rank two");
    const auto width = tensor::sizeOf(dtype);
    if (width == 0 || shape.elementCount() >
                          std::numeric_limits<std::size_t>::max() / width ||
        source.size() != shape.elementCount() * width) {
        throw std::invalid_argument("weight bytes do not match shape and dtype");
    }
    ConvertedWeight result;
    result.bytes.resize(source.size());
    result.dtype = dtype;
    result.layout = targetLayout;
    if (sourceLayout == targetLayout) {
        std::memcpy(result.bytes.data(), source.data(), source.size());
        result.shape = shape;
        return result;
    }
    const auto rows = shape.dimensions()[0];
    const auto columns = shape.dimensions()[1];
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const auto sourceOffset = (row * columns + column) * width;
            const auto targetOffset = (column * rows + row) * width;
            std::memcpy(result.bytes.data() + targetOffset,
                        source.data() + sourceOffset, width);
        }
    }
    result.shape = tensor::Shape{columns, rows};
    return result;
}

} // namespace hypermoe::conversion
