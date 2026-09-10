#pragma once

#include "tensor/Tensor.hpp"

namespace hypermoe::runtime::generation {

struct InferenceConfig {
    tensor::Device device{tensor::Device::cpu()};
};

} // namespace hypermoe::runtime::generation
