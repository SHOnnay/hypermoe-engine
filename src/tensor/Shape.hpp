#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace hypermoe::tensor {

class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<std::size_t> dimensions);
    explicit Shape(std::vector<std::size_t> dimensions);
    Shape(std::vector<std::size_t> dimensions,
          std::vector<std::size_t> strides);

    [[nodiscard]] const std::vector<std::size_t>& dimensions() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t elementCount() const noexcept;
    [[nodiscard]] std::size_t storageElementCount() const noexcept;
    [[nodiscard]] bool isContiguous() const noexcept;
    [[nodiscard]] bool operator==(const Shape& other) const noexcept = default;

private:
    void validateAndCount();
    void makeContiguousStrides();

    std::vector<std::size_t> dimensions_;
    std::vector<std::size_t> strides_;
    std::size_t elementCount_{1};
    std::size_t storageElementCount_{1};
};

} // namespace hypermoe::tensor
