#pragma once

#include "hypermoe/memory/memory_tier.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hypermoe {

using ExpertId = std::uint32_t;
using LayerId = std::uint32_t;

enum class QuantizationType {
    Fp32,
    Fp16,
    Bf16,
    Q4,
    Q5,
    Q6,
    Int8,
    Fp8,
};

[[nodiscard]] constexpr std::string_view toString(QuantizationType type) noexcept {
    switch (type) {
    case QuantizationType::Fp32: return "FP32";
    case QuantizationType::Fp16: return "FP16";
    case QuantizationType::Bf16: return "BF16";
    case QuantizationType::Q4: return "Q4";
    case QuantizationType::Q5: return "Q5";
    case QuantizationType::Q6: return "Q6";
    case QuantizationType::Int8: return "INT8";
    case QuantizationType::Fp8: return "FP8";
    }
    return "UNKNOWN";
}

struct Expert {
    ExpertId id{};
    LayerId layer{};
    std::size_t sizeBytes{};
    QuantizationType quantization{QuantizationType::Q4};
    MemoryTier location{MemoryTier::Nvme};
};

} // namespace hypermoe
