#include "tensor/Shape.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace hypermoe::tensor {

Shape::Shape(std::initializer_list<std::size_t> dimensions)
    : Shape(std::vector<std::size_t>(dimensions)) {}

Shape::Shape(std::vector<std::size_t> dimensions)
    : dimensions_(std::move(dimensions)) {
    validateAndCount();
    makeContiguousStrides();
}

Shape::Shape(std::vector<std::size_t> dimensions,
             std::vector<std::size_t> strides)
    : dimensions_(std::move(dimensions)), strides_(std::move(strides)) {
    validateAndCount();
    if (strides_.size() != dimensions_.size()) {
        throw std::invalid_argument("shape dimensions and strides must have equal rank");
    }
    for (const auto stride : strides_) {
        if (stride == 0 && elementCount_ > 1) {
            throw std::invalid_argument("non-scalar tensor strides must be nonzero");
        }
    }
    storageElementCount_ = 1;
    for (std::size_t index = 0; index < dimensions_.size(); ++index) {
        const auto distance = dimensions_[index] - 1;
        if (distance != 0 &&
            strides_[index] >
                (std::numeric_limits<std::size_t>::max() - storageElementCount_) /
                    distance) {
            throw std::overflow_error("strided tensor storage span overflow");
        }
        storageElementCount_ += distance * strides_[index];
    }
}

const std::vector<std::size_t>& Shape::dimensions() const noexcept {
    return dimensions_;
}

const std::vector<std::size_t>& Shape::strides() const noexcept { return strides_; }
std::size_t Shape::rank() const noexcept { return dimensions_.size(); }
std::size_t Shape::elementCount() const noexcept { return elementCount_; }
std::size_t Shape::storageElementCount() const noexcept {
    return storageElementCount_;
}

bool Shape::isContiguous() const noexcept {
    std::size_t expected = 1;
    for (std::size_t index = dimensions_.size(); index > 0; --index) {
        if (strides_[index - 1] != expected) return false;
        expected *= dimensions_[index - 1];
    }
    return true;
}

void Shape::validateAndCount() {
    elementCount_ = 1;
    for (const auto dimension : dimensions_) {
        if (dimension == 0) throw std::invalid_argument("tensor dimensions must be nonzero");
        if (elementCount_ > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor element count overflow");
        }
        elementCount_ *= dimension;
    }
}

void Shape::makeContiguousStrides() {
    strides_.resize(dimensions_.size());
    std::size_t stride = 1;
    for (std::size_t index = dimensions_.size(); index > 0; --index) {
        strides_[index - 1] = stride;
        stride *= dimensions_[index - 1];
    }
    storageElementCount_ = elementCount_;
}

} // namespace hypermoe::tensor
