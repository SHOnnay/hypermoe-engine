#include "transformer/attention/CudaAttention.hpp"

#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/backend/TensorBackend.hpp"
#include "transformer/attention/CpuAttention.hpp"

#include <stdexcept>
#include <utility>

namespace hypermoe::transformer::attention {
namespace {

tensor::Tensor toHost(tensor::TensorBackend& backend, tensor::TensorView value) {
    [[maybe_unused]] const auto owner = value.lockOwner();
    if (!owner || !value || value.device() != backend.device() ||
        !value.isContiguous()) {
        throw std::invalid_argument("CUDA attention received an incompatible tensor");
    }
    tensor::CpuTensorBackend cpu;
    auto result = cpu.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

tensor::Tensor toDevice(tensor::TensorBackend& backend, tensor::TensorView value) {
    auto result = backend.allocateTensor(value.shape(), value.dtype());
    backend.copyTensor(value, result.view());
    return result;
}

} // namespace

CudaAttention::CudaAttention(std::shared_ptr<tensor::TensorBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_ || !backend_->available() ||
        backend_->device().type != tensor::DeviceType::CUDA) {
        throw std::invalid_argument("CUDA attention requires an available CUDA backend");
    }
}

std::string_view CudaAttention::name() const noexcept {
    return "CUDA projected reference attention";
}
tensor::Device CudaAttention::device() const noexcept { return backend_->device(); }

AttentionResult CudaAttention::execute(
    tensor::TensorView hiddenStates,
    const AttentionWeights& weights,
    const AttentionConfiguration& configuration) {
    auto hostBackend = std::make_shared<tensor::CpuTensorBackend>();
    auto hostHidden = toHost(*backend_, hiddenStates);
    auto hostQueryWeights = toHost(*backend_, weights.query);
    auto hostKeyWeights = toHost(*backend_, weights.key);
    auto hostValueWeights = toHost(*backend_, weights.value);
    auto hostOutputWeights = toHost(*backend_, weights.output);
    CpuAttention reference(hostBackend);
    auto host = reference.execute(
        hostHidden.view(),
        {hostQueryWeights.view(), hostKeyWeights.view(), hostValueWeights.view(),
         hostOutputWeights.view()},
        configuration);

    AttentionResult result;
    result.query = backend_->allocateTensor(host.query.shape(), host.query.dtype());
    result.key = backend_->allocateTensor(host.key.shape(), host.key.dtype());
    result.value = backend_->allocateTensor(host.value.shape(), host.value.dtype());
    result.output = backend_->allocateTensor(host.output.shape(), host.output.dtype());

    // Projection GEMMs execute through cuBLAS. RoPE remains on the reference path,
    // so its transformed query/key replace the raw projections when enabled.
    backend_->matmul(hiddenStates, weights.query, result.query.view());
    backend_->matmul(hiddenStates, weights.key, result.key.view());
    backend_->matmul(hiddenStates, weights.value, result.value.view());
    if (configuration.rotaryEmbedding) {
        backend_->copyTensor(host.query.view(), result.query.view());
        backend_->copyTensor(host.key.view(), result.key.view());
    }
    result.scores = toDevice(*backend_, host.scores.view());
    result.probabilities = toDevice(*backend_, host.probabilities.view());
    result.context = toDevice(*backend_, host.context.view());
    backend_->matmul(result.context.view(), weights.output, result.output.view());
    backend_->synchronize();
    return result;
}

} // namespace hypermoe::transformer::attention
