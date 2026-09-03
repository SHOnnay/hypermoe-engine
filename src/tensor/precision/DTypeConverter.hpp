#pragma once

#include "tensor/Tensor.hpp"
#include "tensor/TensorView.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace hypermoe::tensor {
class TensorBackend;
}

namespace hypermoe::tensor::precision {

struct PrecisionExecutionPlan {
    DType storageDType{DType::FP32};
    DType executionDType{DType::FP32};
    Device executionDevice{Device::cpu()};
    bool conversionRequired{};
    bool nativeExecution{};
};

class DTypeConverter {
public:
    [[nodiscard]] static PrecisionExecutionPlan
    selectExecutionPlan(DType storageType, Device executionDevice);
    [[nodiscard]] static std::vector<float>
    toFp32(std::span<const std::byte> source, DType sourceType);
    [[nodiscard]] static Tensor
    toFp32Tensor(TensorView source, TensorBackend& destinationBackend);
};

} // namespace hypermoe::tensor::precision
