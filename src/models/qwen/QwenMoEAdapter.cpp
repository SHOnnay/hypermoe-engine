#include "models/qwen/QwenMoEAdapter.hpp"

#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypermoe::models::qwen {
namespace {

using metadata::JsonValue;
using metadata::MetadataError;

std::filesystem::path resolveManifest(const std::filesystem::path& path) {
    std::error_code error;
    if (std::filesystem::is_directory(path, error) && !error) {
        return path / "metadata.json";
    }
    return path;
}

std::size_t asSize(const JsonValue& value, std::string_view field) {
    const auto number = value.asUInt64();
    if (number > std::numeric_limits<std::size_t>::max()) {
        throw MetadataError(std::string(field) + " exceeds size_t");
    }
    return static_cast<std::size_t>(number);
}

std::uint32_t asId(const JsonValue& value, std::string_view field) {
    const auto number = value.asUInt64();
    if (number > std::numeric_limits<std::uint32_t>::max()) {
        throw MetadataError(std::string(field) + " exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(number);
}

TensorMetadata parseTensor(const JsonValue& value) {
    TensorMetadata tensor;
    tensor.name = value.require("name").asString();
    if (tensor.name.empty()) throw MetadataError("tensor name must not be empty");
    std::vector<std::size_t> dimensions;
    for (const auto& dimension : value.require("shape").asArray()) {
        dimensions.push_back(asSize(dimension, "tensor shape"));
    }
    if (dimensions.empty()) throw MetadataError("tensor shape must have at least one axis");
    tensor.shape = tensor::Shape(std::move(dimensions));

    const auto& dtype = value.require("dtype").asString();
    if (dtype == "FP32") tensor.dtype = tensor::DType::FP32;
    else if (dtype == "FP16") tensor.dtype = tensor::DType::FP16;
    else if (dtype == "INT8") tensor.dtype = tensor::DType::INT8;
    else if (dtype == "Q4") {
        tensor.dtype = tensor::DType::INT8;
        tensor.quantizedDType = tensor::quantization::QuantizedDType::Q4;
    } else {
        throw MetadataError("unsupported tensor dtype: " + dtype);
    }
    if (const auto* quantization = value.find("quantization")) {
        const auto& encoding = quantization->asString();
        if (encoding == "INT8") {
            tensor.quantizedDType = tensor::quantization::QuantizedDType::INT8;
        } else if (encoding == "Q4") {
            tensor.quantizedDType = tensor::quantization::QuantizedDType::Q4;
        } else {
            throw MetadataError("unsupported tensor quantization: " + encoding);
        }
    }
    tensor.offset = value.require("offset").asUInt64();
    tensor.size = value.require("size").asUInt64();
    tensor.layerId = asId(value.require("layer_id"), "layer_id");
    if (const auto* expert = value.find("expert_id")) {
        if (!expert->isNull()) tensor.expertId = asId(*expert, "expert_id");
    }

    std::uint64_t requiredBytes{};
    if (tensor.quantizedDType) {
        requiredBytes = tensor::quantization::storageSizeBytes(
            tensor.shape, *tensor.quantizedDType);
    } else {
        const auto elementBytes = tensor::sizeOf(tensor.dtype);
        if (tensor.shape.storageElementCount() >
            std::numeric_limits<std::uint64_t>::max() / elementBytes) {
            throw MetadataError("tensor byte size overflow");
        }
        requiredBytes = tensor.shape.storageElementCount() * elementBytes;
    }
    if (tensor.size != requiredBytes) {
        throw MetadataError("tensor size does not match shape and dtype: " + tensor.name);
    }
    if (tensor.offset > std::numeric_limits<std::uint64_t>::max() - tensor.size) {
        throw MetadataError("tensor offset range overflow: " + tensor.name);
    }
    return tensor;
}

struct ParsedName {
    std::uint32_t layerId{};
    std::optional<std::uint32_t> expertId;
    std::optional<ExpertWeightType> weightType;
    bool router{};
};

std::optional<ParsedName> parseQwenName(const std::string& name) {
    static const std::regex expertPattern(
        R"(^model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.(gate_proj|up_proj|down_proj)\.weight$)",
        std::regex::ECMAScript | std::regex::optimize);
    static const std::regex routerPattern(
        R"(^model\.layers\.([0-9]+)\.mlp\.gate\.weight$)",
        std::regex::ECMAScript | std::regex::optimize);
    std::smatch match;
    if (std::regex_match(name, match, expertPattern)) {
        const auto layer = std::stoull(match[1].str());
        const auto expert = std::stoull(match[2].str());
        if (layer > std::numeric_limits<std::uint32_t>::max() ||
            expert > std::numeric_limits<std::uint32_t>::max()) {
            throw MetadataError("Qwen tensor name contains an out-of-range ID");
        }
        ExpertWeightType type = ExpertWeightType::GATE;
        if (match[3].str() == "up_proj") type = ExpertWeightType::UP;
        else if (match[3].str() == "down_proj") type = ExpertWeightType::DOWN;
        return ParsedName{static_cast<std::uint32_t>(layer),
                          static_cast<std::uint32_t>(expert), type, false};
    }
    if (std::regex_match(name, match, routerPattern)) {
        const auto layer = std::stoull(match[1].str());
        if (layer > std::numeric_limits<std::uint32_t>::max()) {
            throw MetadataError("Qwen router name contains an out-of-range layer ID");
        }
        return ParsedName{static_cast<std::uint32_t>(layer), std::nullopt,
                          std::nullopt, true};
    }
    return std::nullopt;
}

JsonValue loadRoot(const std::filesystem::path& path) {
    return metadata::parseJsonFile(resolveManifest(path));
}

std::vector<TensorMetadata> parseTensorIndex(const JsonValue& root) {
    std::vector<TensorMetadata> tensors;
    for (const auto& tensor : root.require("tensors").asArray()) {
        tensors.push_back(parseTensor(tensor));
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto& left, const auto& right) {
        return left.offset < right.offset;
    });
    for (std::size_t index = 1; index < tensors.size(); ++index) {
        if (tensors[index - 1].offset + tensors[index - 1].size >
            tensors[index].offset) {
            throw MetadataError("tensor storage ranges overlap");
        }
    }
    return tensors;
}

} // namespace

ModelMetadata QwenMoEAdapter::loadModelMetadata(
    const std::filesystem::path& manifestPath) const {
    const auto root = loadRoot(manifestPath);
    ModelMetadata result;
    result.schema = root.require("schema").asString();
    if (result.schema != "hypermoe.model-manifest.v1") {
        throw MetadataError("unsupported HyperMoE model manifest schema");
    }
    result.architectureName = root.require("architecture").asString();
    if (result.architectureName != "QWEN_MOE") {
        throw MetadataError("Qwen adapter received a different architecture");
    }
    result.config.modelName = root.require("model_name").asString();
    result.config.layerCount = asSize(root.require("layer_count"), "layer_count");
    result.config.expertCount = asSize(root.require("expert_count"), "expert_count");
    result.config.hiddenSize = asSize(root.require("hidden_size"), "hidden_size");
    result.config.intermediateSize =
        asSize(root.require("intermediate_size"), "intermediate_size");
    if (result.config.modelName.empty() || result.config.layerCount == 0 ||
        result.config.expertCount == 0 || result.config.hiddenSize == 0 ||
        result.config.intermediateSize == 0) {
        throw MetadataError("model dimensions and counts must be nonzero");
    }
    if (result.config.layerCount > std::numeric_limits<std::uint32_t>::max() ||
        result.config.expertCount > std::numeric_limits<std::uint32_t>::max()) {
        throw MetadataError("model layer or expert count exceeds runtime IDs");
    }
    result.config.capabilities = capabilities();

    const auto& router = root.require("router");
    result.router.expertCount = asSize(router.require("expert_count"),
                                       "router.expert_count");
    result.router.topK = asSize(router.require("top_k"), "router.top_k");
    const auto& normalization = router.require("normalization").asString();
    if (normalization == "SOFTMAX") {
        result.router.normalization = router::RoutingNormalization::Softmax;
    } else if (normalization == "NONE") {
        result.router.normalization = router::RoutingNormalization::None;
    } else {
        throw MetadataError("unsupported router normalization");
    }
    if (const auto* renormalize = router.find("renormalize_selected")) {
        result.router.renormalizeSelected = renormalize->asBool();
    }
    result.router.validate();
    if (result.router.expertCount != result.config.expertCount) {
        throw MetadataError("router and model expert counts differ");
    }

    result.tensors = parseTensorIndex(root);
    result.config.capabilities.quantizedExpertWeights =
        std::any_of(result.tensors.begin(), result.tensors.end(),
                    [](const auto& tensor) { return tensor.isQuantized(); });
    result.layers.resize(result.config.layerCount);
    for (std::size_t layer = 0; layer < result.layers.size(); ++layer) {
        result.layers[layer].layerId = static_cast<std::uint32_t>(layer);
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> expertPositions;
    for (const auto& tensor : result.tensors) {
        if (tensor.layerId >= result.config.layerCount) {
            throw MetadataError("tensor layer ID exceeds model layer count");
        }
        const auto parsed = parseQwenName(tensor.name);
        if (!parsed) {
            if (tensor.expertId) {
                throw MetadataError("expert tensor has an unrecognized Qwen name: " +
                                    tensor.name);
            }
            continue;
        }
        if (parsed->layerId != tensor.layerId || parsed->expertId != tensor.expertId) {
            throw MetadataError("tensor identity disagrees with Qwen tensor name");
        }
        auto& layer = result.layers[tensor.layerId];
        if (parsed->router) {
            if (layer.routerTensor) throw MetadataError("duplicate router tensor for layer");
            layer.routerTensor = tensor;
            continue;
        }
        if (!tensor.expertId || *tensor.expertId >= result.config.expertCount) {
            throw MetadataError("tensor expert ID exceeds model expert count");
        }
        const auto identity = std::pair{tensor.layerId, *tensor.expertId};
        const auto [position, inserted] =
            expertPositions.emplace(identity, layer.experts.size());
        if (inserted) layer.experts.push_back({tensor.layerId, *tensor.expertId, {}});
        layer.experts[position->second].tensors.push_back(tensor);
    }
    const auto mappings = getExpertMapping(result);
    for (auto& layer : result.layers) {
        std::sort(layer.experts.begin(), layer.experts.end(),
                  [](const auto& left, const auto& right) {
                      return left.expertId < right.expertId;
                  });
        if (!layer.experts.empty() && !layer.routerTensor) {
            throw MetadataError("MoE layer is missing its router tensor");
        }
        if (!layer.experts.empty() &&
            layer.experts.size() != result.config.expertCount) {
            throw MetadataError("MoE layer does not contain the declared expert count");
        }
        if (layer.routerTensor) {
            const auto& shape = layer.routerTensor->shape.dimensions();
            if (shape.size() != 2 || shape[0] != result.config.hiddenSize ||
                shape[1] != result.config.expertCount ||
                layer.routerTensor->dtype != tensor::DType::FP32 ||
                layer.routerTensor->isQuantized()) {
                throw MetadataError("router tensor shape or dtype is incompatible");
            }
        }
        for (const auto& expert : layer.experts) {
            const auto& mapping = mappings.require(layer.layerId, expert.expertId);
            const auto validUp = [&](const TensorMetadata& tensor) {
                const auto& shape = tensor.shape.dimensions();
                return shape.size() == 2 && shape[0] == result.config.hiddenSize &&
                       shape[1] == result.config.intermediateSize;
            };
            const auto& downShape = mapping.downProjection->shape.dimensions();
            if (!validUp(*mapping.gateProjection) ||
                !validUp(*mapping.upProjection) || downShape.size() != 2 ||
                downShape[0] != result.config.intermediateSize ||
                downShape[1] != result.config.hiddenSize) {
                throw MetadataError("expert projection shapes are incompatible");
            }
        }
    }
    return result;
}

std::vector<TensorMetadata> QwenMoEAdapter::loadTensorIndex(
    const std::filesystem::path& manifestPath) const {
    return parseTensorIndex(loadRoot(manifestPath));
}

ModelArchitecture QwenMoEAdapter::getArchitecture() const noexcept {
    return ModelArchitecture::QWEN_MOE;
}

ModelCapabilities QwenMoEAdapter::capabilities() const noexcept {
    return {.routedExperts = true,
            .configurableTopK = true,
            .normalizedRouting = true,
            .gatedExpertMlp = true,
            .sharedExperts = false,
            .quantizedExpertWeights = true};
}

std::size_t QwenMoEAdapter::getLayerCount(const ModelMetadata& metadata) const noexcept {
    return metadata.config.layerCount;
}

std::size_t QwenMoEAdapter::getExpertCount(const ModelMetadata& metadata) const noexcept {
    return metadata.config.expertCount;
}

router::RouterConfig QwenMoEAdapter::getRouterConfiguration(
    const ModelMetadata& metadata) const {
    metadata.router.validate();
    return metadata.router;
}

ExpertWeightMap QwenMoEAdapter::getExpertMapping(
    const ModelMetadata& metadata) const {
    ExpertWeightMap mappings;
    for (const auto& tensor : metadata.tensors) {
        const auto parsed = parseQwenName(tensor.name);
        if (!parsed || !parsed->expertId || !parsed->weightType) continue;
        mappings.add(parsed->layerId, *parsed->expertId, *parsed->weightType, tensor);
    }
    return mappings;
}

} // namespace hypermoe::models::qwen
