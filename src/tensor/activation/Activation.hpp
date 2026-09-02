#pragma once

#include "tensor/TensorView.hpp"

#include <memory>
#include <string_view>

namespace hypermoe {
class Profiler;
}

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::tensor::activation {

enum class ActivationType {
    SiLU,
    GELU,
};

[[nodiscard]] constexpr std::string_view toString(ActivationType type) noexcept {
    switch (type) {
    case ActivationType::SiLU: return "SiLU";
    case ActivationType::GELU: return "GELU";
    }
    return "UNKNOWN";
}

[[nodiscard]] float silu(float value) noexcept;
[[nodiscard]] float gelu(float value) noexcept;

// Dispatches through the selected tensor backend. CUDA tensors currently use a
// checked host-staged reference path, leaving a stable seam for a future kernel.
void apply(ActivationType type,
           TensorBackend& backend,
           TensorView input,
           TensorView output,
           const std::shared_ptr<Profiler>& profiler = {});

} // namespace hypermoe::tensor::activation
