#include "backend/CpuBackend.hpp"
#include "hypermoe/experts/expert.hpp"
#include "models/ModelManifest.hpp"
#include "storage/ExpertIndex.hpp"
#include "storage/MappedFile.hpp"
#include "tensor/TensorError.hpp"
#include "tensor/backend/CpuTensorBackend.hpp"
#include "tensor/quantization/QuantizedTensor.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures{};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function>
void expectThrows(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const Exception&) {
        expect(true, message);
    } catch (...) {
        expect(false, message);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t sequence{};
        path_ = std::filesystem::temp_directory_path() /
            ("hypermoe-platform-" +
             std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

hypermoe::models::ModelManifest makeManifest() {
    using namespace hypermoe;
    models::ModelManifest manifest;
    manifest.modelName = "Portable \xCE\xBB fixture";
    manifest.architecture = models::ModelArchitecture::QWEN_MOE;
    manifest.sourceArchitecture = "portable_test";
    manifest.config.modelName = manifest.modelName;
    manifest.config.layerCount = 1;
    manifest.config.expertCount = 1;
    manifest.config.hiddenSize = 2;
    manifest.config.intermediateSize = 2;
    manifest.config.capabilities = {true, true, true, true, false, false};
    manifest.router.config = {1, 1, router::RoutingNormalization::Softmax, true};
    manifest.router.layout = models::TensorLayout::InputOutput;
    models::runtime::ModelArchitecture architecture;
    architecture.layerCount = 1;
    architecture.hiddenDimension = 2;
    architecture.attentionHeads = 1;
    architecture.keyValueHeads = 1;
    architecture.headDimension = 2;
    architecture.projectionHeadDimension = 2;
    architecture.expertCount = 1;
    architecture.topK = 1;
    architecture.vocabularySize = 3;
    manifest.runtimeArchitecture = architecture;

    std::uint64_t offset = 0x1'0000'0000ULL;
    const auto add = [&](std::string name, tensor::Shape shape) {
        const auto size = static_cast<std::uint64_t>(
            shape.elementCount() * sizeof(float));
        const auto current = offset;
        offset += size;
        manifest.tensors.push_back(
            {std::move(name), "portable/expert-data.bin", current, size,
             tensor::DType::FP32, std::move(shape)});
        return current;
    };
    const auto gateOffset = add("expert.gate", {2,2});
    const auto upOffset = add("expert.up", {2,2});
    const auto downOffset = add("expert.down", {2,2});
    const auto routerOffset = add("layer.router", {2,1});
    (void)routerOffset;
    add("layer.q", {2,2});
    add("layer.k", {2,2});
    add("layer.v", {2,2});
    add("layer.o", {2,2});
    add("layer.input_norm", {2});
    add("layer.post_norm", {2});
    add("model.embedding", {3,2});
    add("model.final_norm", {2});
    add("model.lm_head", {2,3});
    manifest.experts.push_back(
        {0, 0,
         {"expert.gate", gateOffset, 16, {2,2}, models::TensorLayout::InputOutput},
         {"expert.up", upOffset, 16, {2,2}, models::TensorLayout::InputOutput},
         {"expert.down", downOffset, 16, {2,2}, models::TensorLayout::InputOutput}});
    manifest.router.tensors.push_back({0, "layer.router"});
    manifest.layers.push_back(
        {0,
         {"layer.q", models::TensorLayout::InputOutput},
         {"layer.k", models::TensorLayout::InputOutput},
         {"layer.v", models::TensorLayout::InputOutput},
         {"layer.o", models::TensorLayout::InputOutput},
         "layer.input_norm", "layer.post_norm", "layer.router"});
    manifest.modelIO = models::ManifestModelIO{
        "model.embedding", "model.final_norm",
        {"model.lm_head", models::TensorLayout::InputOutput}, false};
    return manifest;
}

void testFixedWidthLittleEndianIndex() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    const auto path = temporary.path() / "experts.index";
    const storage::ExpertRecord record{
        0x01020304U, 0x11223344U, 0x0102030405060708ULL,
        0x1112131415161718ULL,
        static_cast<std::uint32_t>(QuantizationType::Q4), 0xa1b2c3d4U};
    storage::ExpertIndex({record}).save(path);
    std::ifstream input(path, std::ios::binary);
    const std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    expect(bytes.size() == 64 && bytes[8] == 1 && bytes[12] == 4 &&
               bytes[13] == 3 && bytes[14] == 2 && bytes[15] == 1 &&
               bytes[32] == 4 && bytes[33] == 3 && bytes[34] == 2 &&
               bytes[35] == 1 && bytes[40] == 8 && bytes[47] == 1,
           "expert index uses fixed-width little-endian fields, not native struct layout");
    const auto loaded = storage::ExpertIndex::load(path).find(
        record.layer_id, record.expert_id);
    expect(loaded && *loaded == record,
           "expert index round-trips 64-bit offsets without native-width narrowing");

    storage::MappedFile mapped(path);
    storage::MappedFile moved(std::move(mapped));
    expect(!mapped.isOpen() && mapped.bytes().empty() && moved.size() == bytes.size(),
           "moved mapped files expose a safe empty span and retain one owner");

    const auto corruptPath = temporary.path() / "bad-endian.index";
    storage::ExpertIndex({record}).save(corruptPath);
    std::fstream corrupt(corruptPath, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekp(12);
    corrupt.put('\0');
    corrupt.close();
    expectThrows<storage::StorageError>(
        [&] { (void)storage::ExpertIndex::load(corruptPath); },
        "expert index rejects a corrupted byte-order marker");
    auto invalid = record;
    invalid.quantization_type = 0xffffffffU;
    expectThrows<storage::StorageError>(
        [&] { (void)storage::ExpertIndex({invalid}); },
        "expert index rejects unknown serialized quantization codes");
}

void testAlignedOwnershipAndAliasing() {
    using namespace hypermoe;
    auto compute = std::make_shared<backend::CpuBackend>();
    void* allocation = compute->allocate(257);
    expect(reinterpret_cast<std::uintptr_t>(allocation) % 64U == 0,
           "CPU backend provides the documented 64-byte alignment");
    compute->copyToDevice(nullptr, nullptr, 0);
    compute->free(allocation);
    expect(compute->stats().allocatedBytes == 0,
           "CPU allocation accounting returns to zero after release");

    auto owner = std::shared_ptr<void>(::operator new(16), [](void* pointer) {
        ::operator delete(pointer);
    });
    auto* misaligned = static_cast<std::byte*>(owner.get()) + 1;
    expectThrows<tensor::TensorError>(
        [&] {
            (void)tensor::Tensor::fromStorage(
                {1}, tensor::DType::FP32, tensor::Device::cpu(),
                misaligned, sizeof(float), owner);
        }, "tensor construction rejects storage misaligned for FP32 access");
    expectThrows<tensor::TensorError>(
        [&] {
            (void)tensor::Tensor::fromStorage(
                {1}, tensor::DType::FP32,
                {static_cast<tensor::DeviceType>(0xffU), 0}, owner.get(),
                sizeof(float), owner);
        }, "tensor construction rejects unknown device enum values");
    expectThrows<tensor::TensorError>(
        [&] {
            (void)tensor::quantization::QuantizedTensor::fromStorage(
                {1}, tensor::quantization::QuantizedDType::INT8, {},
                {static_cast<tensor::DeviceType>(0xffU), 0}, owner.get(), 1,
                owner);
        }, "quantized tensors reject unknown device enum values");

    auto backend = std::make_shared<tensor::CpuTensorBackend>();
    auto left = backend->allocateTensor({2,2}, tensor::DType::FP32);
    auto right = backend->allocateTensor({2,2}, tensor::DType::FP32);
    expectThrows<std::invalid_argument>(
        [&] { backend->matmul(left.view(), right.view(), left.view()); },
        "CPU GEMM rejects output aliasing consistently with CUDA GEMM");
    tensor::TensorView expired;
    {
        auto temporaryTensor = backend->allocateTensor({1}, tensor::DType::FP32);
        expired = temporaryTensor.view();
    }
    expect(!expired && expired.data() == nullptr,
           "tensor views fail closed after their owning storage expires");
    expectThrows<std::overflow_error>(
        [] {
            (void)tensor::Shape{std::numeric_limits<std::size_t>::max(), 2};
        }, "tensor shape rejects platform-width multiplication overflow");
}

void testPortableManifestRoundTrip() {
    using namespace hypermoe;
    TemporaryDirectory temporary;
    auto manifest = makeManifest();
    manifest.validate();
    const auto path = temporary.path() / "portable-manifest.json";
    manifest.save(path);
    const auto loaded = models::ModelManifest::load(path);
    expect(loaded.modelName == manifest.modelName &&
               loaded.tensors.front().offset > std::numeric_limits<std::uint32_t>::max() &&
               loaded.tensors.front().offset == manifest.tensors.front().offset &&
               loaded.modelIO && loaded.layers.size() == 1,
           "manifest JSON preserves UTF-8 identity, 64-bit offsets, and logical mappings");
    loaded.validate();
    auto invalidLayout = loaded;
    invalidLayout.router.layout = static_cast<models::TensorLayout>(0xffffffffU);
    expectThrows<std::invalid_argument>(
        [&] { invalidLayout.validate(); },
        "manifest rejects unknown programmatic tensor-layout enum values");
    auto invalidDtype = loaded;
    invalidDtype.tensors.front().dtype = static_cast<tensor::DType>(0xffffffffU);
    expectThrows<std::invalid_argument>(
        [&] { invalidDtype.validate(); },
        "manifest rejects unknown dtype values before byte-size arithmetic");
    auto unsafePath = loaded;
    unsafePath.tensors.front().sourceFile = "../outside.bin";
    expectThrows<std::invalid_argument>(
        [&] { unsafePath.validate(); },
        "manifest rejects platform-independent parent traversal");
}

} // namespace

int main() {
    testFixedWidthLittleEndianIndex();
    testAlignedOwnershipAndAliasing();
    testPortableManifestRoundTrip();
    if (failures != 0) {
        std::cerr << failures << " platform assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All cross-platform regression tests passed\n";
    return EXIT_SUCCESS;
}
