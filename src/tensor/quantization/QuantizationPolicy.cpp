#include "tensor/quantization/QuantizationPolicy.hpp"

#include <sstream>
#include <stdexcept>

namespace hypermoe::tensor::quantization {

void QuantizationPolicy::validate(const Shape& shape) const {
    validateParameters(tensorType, {scale, zeroPoint});
    if (!shape.isContiguous() || groupSize == 0 || groupSize > shape.elementCount() ||
        shape.elementCount() % groupSize != 0) {
        throw std::invalid_argument(
            "quantization group size must evenly divide a contiguous tensor");
    }
}

std::string QuantizationPolicy::toJson() const {
    std::ostringstream output;
    output << "{\"schema\":\"hypermoe.quantization-policy.v1\",\"type\":\""
           << toString(tensorType) << "\",\"scale\":" << scale
           << ",\"zero_point\":" << zeroPoint
           << ",\"group_size\":" << groupSize << '}';
    return output.str();
}

} // namespace hypermoe::tensor::quantization
