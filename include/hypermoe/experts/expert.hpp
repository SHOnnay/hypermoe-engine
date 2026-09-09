#pragma once

#include "hypermoe/memory/memory_tier.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hypermoe {

using ExpertId = std::uint32_t;
using LayerId = std::uint32_t;

enum class QuantizationType : std::uint32_t {
    Fp32 = 0,
    Fp16 = 1,
    Bf16 = 2,
    Q4 = 3,
    Q5 = 4,
    Q6 = 5,
    Int8 = 6,
    Fp8 = 7,
};

[[nodiscard]] constexpr bool isValid(QuantizationType type) noexcept {
    switch (type) {
    case QuantizationType::Fp32:
    case QuantizationType::Fp16:
    case QuantizationType::Bf16:
    case QuantizationType::Q4:
    case QuantizationType::Q5:
    case QuantizationType::Q6:
    case QuantizationType::Int8:
    case QuantizationType::Fp8: return true;
    }
    return false;
}

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
