#pragma once

#include <stdexcept>

namespace hypermoe::tensor {

class TensorError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

} // namespace hypermoe::tensor
