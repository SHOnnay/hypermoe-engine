#pragma once

#include "hypermoe/experts/cache_policy.hpp"
#include "hypermoe/memory/memory_manager.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hypermoe {

class TransferManager;
class ExpertManager;
namespace backend {
class DeviceBuffer;
}

namespace tensor {
class Shape;
class Tensor;
class TensorView;
enum class DType;
namespace quantization {
class QuantizedTensor;
enum class QuantizedDType : std::uint32_t;
struct QuantizationParameters;
}
}

// A residency lease keeps the manager's logical allocation and physical
// DeviceBuffer resident for the full lifetime of any TensorView made from it.
class ExpertResidencyLease {
public:
    ExpertResidencyLease() = default;
    ~ExpertResidencyLease();
    ExpertResidencyLease(const ExpertResidencyLease&) = delete;
    ExpertResidencyLease& operator=(const ExpertResidencyLease&) = delete;
    ExpertResidencyLease(ExpertResidencyLease&& other) noexcept;
    ExpertResidencyLease& operator=(ExpertResidencyLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::shared_ptr<backend::DeviceBuffer> buffer() const noexcept;
    [[nodiscard]] tensor::TensorView view(const tensor::Shape& shape,
                                          tensor::DType dtype) const;

private:
    friend class ExpertManager;
    ExpertResidencyLease(std::shared_ptr<std::atomic_size_t> leaseCount,
                         std::shared_ptr<backend::DeviceBuffer> buffer) noexcept;
    void reset() noexcept;

    std::shared_ptr<std::atomic_size_t> leaseCount_;
    std::shared_ptr<backend::DeviceBuffer> buffer_;
};

enum class RequestSource {
    VramHit,
    RamHit,
    NvmeLoad,
};

struct ExpertRequestResult {
    Expert expert;
    RequestSource source{RequestSource::NvmeLoad};
    double simulatedLatencyMs{};
};

struct ExpertManagerStats {
    std::uint64_t requests{};
    std::uint64_t vramHits{};
    std::uint64_t ramHits{};
    std::uint64_t nvmeLoads{};
    std::uint64_t vramEvictions{};
    std::uint64_t ramEvictions{};
    std::uint64_t nvmeBytesRead{};
    double simulatedLoadingLatencyMs{};

    [[nodiscard]] double vramHitRate() const noexcept;
};

class ExpertManager {
public:
    ExpertManager(MemoryManager& memory, std::unique_ptr<CachePolicy> policy);
    ExpertManager(MemoryManager& memory,
                  std::unique_ptr<CachePolicy> policy,
                  std::shared_ptr<TransferManager> transfers);

    void registerExpert(Expert expert);
    [[nodiscard]] ExpertRequestResult requestExpert(ExpertId id);
    [[nodiscard]] ExpertRequestResult requestExpert(LayerId layerId, ExpertId id);
    void moveExpert(ExpertId id, MemoryTier destination);
    void moveExpert(LayerId layerId, ExpertId id, MemoryTier destination);
    [[nodiscard]] std::size_t evictUntilWithin(MemoryTier tier,
                                               std::size_t maximumUsedBytes);

    [[nodiscard]] std::optional<Expert> findExpert(ExpertId id) const;
    [[nodiscard]] std::optional<Expert> findExpert(LayerId layerId, ExpertId id) const;
    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>>
    residentWeights(ExpertId id) const;
    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>>
    residentWeights(LayerId layerId, ExpertId id) const;
    [[nodiscard]] std::shared_ptr<backend::DeviceBuffer>
    residentDeviceWeights(ExpertId id) const;
    [[nodiscard]] std::shared_ptr<backend::DeviceBuffer>
    residentDeviceWeights(LayerId layerId, ExpertId id) const;
    [[nodiscard]] tensor::Tensor residentDeviceTensor(
        ExpertId id, const tensor::Shape& shape, tensor::DType dtype) const;
    [[nodiscard]] tensor::Tensor residentDeviceTensor(
        LayerId layerId,
        ExpertId id,
        const tensor::Shape& shape,
        tensor::DType dtype) const;
    [[nodiscard]] tensor::TensorView residentDeviceTensorView(
        ExpertId id, const tensor::Shape& shape, tensor::DType dtype) const;
    [[nodiscard]] tensor::TensorView residentDeviceTensorView(
        LayerId layerId,
        ExpertId id,
        const tensor::Shape& shape,
        tensor::DType dtype) const;
    [[nodiscard]] tensor::quantization::QuantizedTensor residentQuantizedTensor(
        ExpertId id,
        const tensor::Shape& shape,
        tensor::quantization::QuantizedDType dtype,
        tensor::quantization::QuantizationParameters parameters) const;
    [[nodiscard]] tensor::quantization::QuantizedTensor residentQuantizedTensor(
        LayerId layerId,
        ExpertId id,
        const tensor::Shape& shape,
        tensor::quantization::QuantizedDType dtype,
        tensor::quantization::QuantizationParameters parameters) const;
    void adoptDeviceWeights(LayerId layerId,
                            ExpertId id,
                            std::shared_ptr<backend::DeviceBuffer> buffer);
    void adoptHostWeights(LayerId layerId,
                          ExpertId id,
                          std::shared_ptr<const std::vector<std::byte>> buffer);
    [[nodiscard]] ExpertResidencyLease acquireResidentExpert(
        LayerId layerId, ExpertId id);
    [[nodiscard]] std::size_t expertCount() const;
    [[nodiscard]] ExpertManagerStats stats() const;

private:
    struct ManagedExpert {
        Expert metadata;
        ExpertId policyId{};
        std::optional<MemoryAllocation> allocation;
        std::shared_ptr<const std::vector<std::byte>> weights;
        std::shared_ptr<backend::DeviceBuffer> deviceWeights;
        std::shared_ptr<std::atomic_size_t> residencyLeases;
    };

    using ExpertKey = std::uint64_t;
    [[nodiscard]] static constexpr ExpertKey key(LayerId layerId,
                                                  ExpertId id) noexcept {
        return (static_cast<ExpertKey>(layerId) << 32U) | id;
    }
    void moveExpertLocked(LayerId layerId,
                          ExpertId id,
                          MemoryTier destination,
                          const std::unordered_set<ExpertId>& pinned);
    void makeRoomLocked(MemoryTier tier,
                        std::size_t requiredBytes,
                        const std::unordered_set<ExpertId>& pinned);
    [[nodiscard]] std::vector<ExpertId> candidatesLocked(MemoryTier tier) const;
    [[nodiscard]] ManagedExpert& requireExpertLocked(LayerId layerId, ExpertId id);
    [[nodiscard]] LayerId resolveLegacyLayer(ExpertId id) const;
    friend class ExpertResidencyLease;

    MemoryManager& memory_;
    std::unique_ptr<CachePolicy> policy_;
    std::shared_ptr<TransferManager> transfers_;
    mutable std::mutex mutex_;
    std::unordered_map<ExpertKey, ManagedExpert> experts_;
    std::unordered_map<ExpertId, ExpertKey> policyExperts_;
    std::unordered_map<ExpertId, std::vector<ExpertKey>> legacyExperts_;
    std::uint64_t nextPolicyId_{};
    ExpertManagerStats stats_;
};

} // namespace hypermoe
