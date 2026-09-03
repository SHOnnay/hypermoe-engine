#include "models/ModelManifest.hpp"

#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace hypermoe::models {
namespace {

using metadata::JsonValue;
using metadata::MetadataError;

std::string escapeJson(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    for (const char rawCharacter : value) {
        const auto character = static_cast<unsigned char>(rawCharacter);
        switch (character) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (character < 0x20U) {
                constexpr char digits[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped.push_back(digits[character >> 4U]);
                escaped.push_back(digits[character & 0x0FU]);
            } else {
                escaped.push_back(static_cast<char>(character));
            }
        }
    }
    return escaped;
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

tensor::Shape parseShape(const JsonValue& value) {
    std::vector<std::size_t> dimensions;
    for (const auto& dimension : value.asArray()) {
        dimensions.push_back(asSize(dimension, "shape dimension"));
    }
    if (dimensions.empty()) throw MetadataError("tensor shape cannot be empty");
    return tensor::Shape(std::move(dimensions));
}

tensor::DType parseDType(std::string_view value) {
    if (value == "FP32") return tensor::DType::FP32;
    if (value == "FP16") return tensor::DType::FP16;
    if (value == "BF16") return tensor::DType::BF16;
    if (value == "INT8") return tensor::DType::INT8;
    throw MetadataError("unsupported manifest dtype: " + std::string(value));
}

TensorLayout parseLayout(std::string_view value) {
    if (value == "INPUT_OUTPUT") return TensorLayout::InputOutput;
    if (value == "OUTPUT_INPUT") return TensorLayout::OutputInput;
    throw MetadataError("unsupported tensor layout: " + std::string(value));
}

ModelArchitecture parseArchitecture(std::string_view value) {
    if (value == "QWEN_MOE") return ModelArchitecture::QWEN_MOE;
    if (value == "DEEPSEEK_MOE") return ModelArchitecture::DEEPSEEK_MOE;
    if (value == "GLM_MOE") return ModelArchitecture::GLM_MOE;
    if (value == "KIMI_MOE") return ModelArchitecture::KIMI_MOE;
    if (value == "MIXTRAL_MOE") return ModelArchitecture::MIXTRAL_MOE;
    if (value == "UNKNOWN") return ModelArchitecture::UNKNOWN;
    throw MetadataError("unsupported manifest architecture enum");
}

ProjectionLocation parseProjection(const JsonValue& value) {
    ProjectionLocation result;
    result.tensorName = value.require("tensor").asString();
    result.offset = value.require("offset").asUInt64();
    result.size = value.require("size").asUInt64();
    result.shape = parseShape(value.require("shape"));
    result.layout = parseLayout(value.require("layout").asString());
    return result;
}

void writeShape(std::ostream& output, const tensor::Shape& shape) {
    output << '[';
    for (std::size_t index = 0; index < shape.dimensions().size(); ++index) {
        if (index != 0) output << ',';
        output << shape.dimensions()[index];
    }
    output << ']';
}

void writeProjection(std::ostream& output, const ProjectionLocation& projection) {
    output << "{\"tensor\":\"" << escapeJson(projection.tensorName)
           << "\",\"offset\":" << projection.offset
           << ",\"size\":" << projection.size << ",\"shape\":";
    writeShape(output, projection.shape);
    output << ",\"layout\":\"" << toString(projection.layout) << "\"}";
}

void validatePath(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) {
        throw std::invalid_argument("manifest tensor source path must be relative");
    }
    for (const auto& component : path) {
        if (component == "..") {
            throw std::invalid_argument("manifest tensor source path escapes artifact root");
        }
    }
}

std::uint64_t requiredBytes(const tensor::Shape& shape, tensor::DType dtype) {
    const auto elements = shape.storageElementCount();
    const auto width = tensor::sizeOf(dtype);
    if (elements > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::overflow_error("manifest tensor size overflow");
    }
    return static_cast<std::uint64_t>(elements) * width;
}

} // namespace

void ModelManifest::validate() const {
    if (schema != schemaVersion) throw std::invalid_argument("unsupported manifest schema");
    if (modelName.empty() || sourceArchitecture.empty()) {
        throw std::invalid_argument("manifest model identity is incomplete");
    }
    if (config.layerCount == 0 || config.expertCount == 0 ||
        config.hiddenSize == 0 || config.intermediateSize == 0) {
        throw std::invalid_argument("manifest model dimensions must be nonzero");
    }
    router.config.validate();
    if (router.config.expertCount != config.expertCount) {
        throw std::invalid_argument("manifest router expert count disagrees with model");
    }
    if (tensors.empty() || experts.empty() || router.tensors.empty()) {
        throw std::invalid_argument("manifest must contain router and expert tensors");
    }

    std::unordered_map<std::string, const ManifestTensor*> byName;
    byName.reserve(tensors.size());
    for (const auto& value : tensors) {
        if (value.name.empty() || value.size == 0) {
            throw std::invalid_argument("manifest tensor identity or size is invalid");
        }
        validatePath(value.sourceFile);
        if (value.offset > std::numeric_limits<std::uint64_t>::max() - value.size) {
            throw std::overflow_error("manifest tensor byte range overflows");
        }
        if (requiredBytes(value.shape, value.dtype) != value.size) {
            throw std::invalid_argument("manifest tensor shape does not match byte size");
        }
        if (!byName.emplace(value.name, &value).second) {
            throw std::invalid_argument("duplicate manifest tensor name");
        }
    }
    std::set<std::uint32_t> routerLayers;
    for (const auto& routerTensor : router.tensors) {
        if (routerTensor.layerId >= config.layerCount ||
            !routerLayers.insert(routerTensor.layerId).second) {
            throw std::invalid_argument("manifest router layer is invalid or duplicated");
        }
        const auto found = byName.find(routerTensor.tensorName);
        if (found == byName.end()) {
            throw std::invalid_argument("manifest router references an unknown tensor");
        }
        const auto expected = router.layout == TensorLayout::OutputInput
            ? tensor::Shape{config.expertCount, config.hiddenSize}
            : tensor::Shape{config.hiddenSize, config.expertCount};
        if (found->second->shape != expected) {
            throw std::invalid_argument("manifest router tensor shape is incompatible");
        }
    }

    std::set<std::pair<std::uint32_t, std::uint32_t>> identities;
    std::unordered_map<std::uint32_t, std::size_t> expertsPerLayer;
    const auto validateProjection = [&](const ProjectionLocation& projection,
                                        bool down) {
        const auto found = byName.find(projection.tensorName);
        if (found == byName.end()) {
            throw std::invalid_argument("expert projection references an unknown tensor");
        }
        const auto& source = *found->second;
        if (projection.size == 0 ||
            projection.offset < source.offset ||
            projection.offset > std::numeric_limits<std::uint64_t>::max() -
                                    projection.size ||
            projection.offset + projection.size > source.offset + source.size) {
            throw std::invalid_argument("expert projection range escapes source tensor");
        }
        if (requiredBytes(projection.shape, source.dtype) != projection.size) {
            throw std::invalid_argument("expert projection shape does not match its bytes");
        }
        tensor::Shape expected;
        if (projection.layout == TensorLayout::OutputInput) {
            expected = down ? tensor::Shape{config.hiddenSize, config.intermediateSize}
                            : tensor::Shape{config.intermediateSize, config.hiddenSize};
        } else {
            expected = down ? tensor::Shape{config.intermediateSize, config.hiddenSize}
                            : tensor::Shape{config.hiddenSize, config.intermediateSize};
        }
        if (projection.shape != expected) {
            throw std::invalid_argument("expert projection shape and layout are incompatible");
        }
    };
    for (const auto& expert : experts) {
        if (expert.layerId >= config.layerCount || expert.expertId >= config.expertCount) {
            throw std::invalid_argument("manifest expert identity is outside model bounds");
        }
        if (!identities.emplace(expert.layerId, expert.expertId).second) {
            throw std::invalid_argument("duplicate manifest expert mapping");
        }
        ++expertsPerLayer[expert.layerId];
        validateProjection(expert.gate, false);
        validateProjection(expert.up, false);
        validateProjection(expert.down, true);
    }
    for (const auto& [layer, count] : expertsPerLayer) {
        if (count != config.expertCount) {
            throw std::invalid_argument(
                "manifest MoE layer does not contain the declared expert count");
        }
        if (!routerLayers.contains(layer)) {
            throw std::invalid_argument("manifest MoE layer has no router tensor");
        }
    }
    for (const auto layer : routerLayers) {
        if (!expertsPerLayer.contains(layer)) {
            throw std::invalid_argument("manifest router layer has no expert mappings");
        }
    }
}

const ManifestTensor* ModelManifest::findTensor(std::string_view name) const noexcept {
    const auto found = std::find_if(tensors.begin(), tensors.end(),
                                    [&](const auto& tensor) { return tensor.name == name; });
    return found == tensors.end() ? nullptr : &*found;
}

const ManifestExpertMapping* ModelManifest::findExpert(
    std::uint32_t layerId, std::uint32_t expertId) const noexcept {
    const auto found = std::find_if(experts.begin(), experts.end(), [&](const auto& value) {
        return value.layerId == layerId && value.expertId == expertId;
    });
    return found == experts.end() ? nullptr : &*found;
}

std::string ModelManifest::toJson() const {
    validate();
    std::ostringstream output;
    output << "{\n  \"schema\": \"" << schema << "\",\n"
           << "  \"model_name\": \"" << escapeJson(modelName) << "\",\n"
           << "  \"architecture\": \"" << toString(architecture) << "\",\n"
           << "  \"source_architecture\": \"" << escapeJson(sourceArchitecture)
           << "\",\n  \"layer_count\": " << config.layerCount
           << ",\n  \"expert_count\": " << config.expertCount
           << ",\n  \"hidden_size\": " << config.hiddenSize
           << ",\n  \"intermediate_size\": " << config.intermediateSize << ",\n"
           << "  \"capabilities\": {\"routed_experts\":"
           << (config.capabilities.routedExperts ? "true" : "false")
           << ",\"configurable_top_k\":"
           << (config.capabilities.configurableTopK ? "true" : "false")
           << ",\"normalized_routing\":"
           << (config.capabilities.normalizedRouting ? "true" : "false")
           << ",\"gated_expert_mlp\":"
           << (config.capabilities.gatedExpertMlp ? "true" : "false")
           << ",\"shared_experts\":"
           << (config.capabilities.sharedExperts ? "true" : "false")
           << ",\"quantized_expert_weights\":"
           << (config.capabilities.quantizedExpertWeights ? "true" : "false")
           << "},\n"
           << "  \"router\": {\"expert_count\":" << router.config.expertCount
           << ",\"top_k\":" << router.config.topK
           << ",\"normalization\":\"" << toString(router.config.normalization)
           << "\",\"renormalize_selected\":"
           << (router.config.renormalizeSelected ? "true" : "false")
           << ",\"layout\":\"" << toString(router.layout) << "\",\"tensors\":[";
    for (std::size_t index = 0; index < router.tensors.size(); ++index) {
        if (index != 0) output << ',';
        output << "{\"layer_id\":" << router.tensors[index].layerId
               << ",\"tensor\":\""
               << escapeJson(router.tensors[index].tensorName) << "\"}";
    }
    output << "]},\n  \"tensors\": [\n";
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& value = tensors[index];
        output << "    {\"name\":\"" << escapeJson(value.name)
               << "\",\"source_file\":\"" << escapeJson(value.sourceFile.generic_string())
               << "\",\"offset\":" << value.offset << ",\"size\":" << value.size
               << ",\"dtype\":\"" << tensor::toString(value.dtype) << "\",\"shape\":";
        writeShape(output, value.shape);
        output << '}' << (index + 1 == tensors.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"experts\": [\n";
    for (std::size_t index = 0; index < experts.size(); ++index) {
        const auto& expert = experts[index];
        output << "    {\"layer_id\":" << expert.layerId
               << ",\"expert_id\":" << expert.expertId << ",\"gate\":";
        writeProjection(output, expert.gate);
        output << ",\"up\":";
        writeProjection(output, expert.up);
        output << ",\"down\":";
        writeProjection(output, expert.down);
        output << '}' << (index + 1 == experts.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return output.str();
}

void ModelManifest::save(const std::filesystem::path& path) const {
    if (path.empty()) throw std::invalid_argument("manifest output path is empty");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create HyperMoE manifest");
    output << toJson();
    if (!output) throw std::runtime_error("failed writing HyperMoE manifest");
}

ModelManifest ModelManifest::load(const std::filesystem::path& path) {
    const auto root = metadata::parseJsonFile(path);
    ModelManifest result;
    result.schema = root.require("schema").asString();
    result.modelName = root.require("model_name").asString();
    result.architecture = parseArchitecture(root.require("architecture").asString());
    result.sourceArchitecture = root.require("source_architecture").asString();
    result.config.modelName = result.modelName;
    result.config.layerCount = asSize(root.require("layer_count"), "layer_count");
    result.config.expertCount = asSize(root.require("expert_count"), "expert_count");
    result.config.hiddenSize = asSize(root.require("hidden_size"), "hidden_size");
    result.config.intermediateSize =
        asSize(root.require("intermediate_size"), "intermediate_size");
    const auto& capabilities = root.require("capabilities");
    result.config.capabilities = {
        .routedExperts = capabilities.require("routed_experts").asBool(),
        .configurableTopK = capabilities.require("configurable_top_k").asBool(),
        .normalizedRouting = capabilities.require("normalized_routing").asBool(),
        .gatedExpertMlp = capabilities.require("gated_expert_mlp").asBool(),
        .sharedExperts = capabilities.require("shared_experts").asBool(),
        .quantizedExpertWeights =
            capabilities.require("quantized_expert_weights").asBool()};
    const auto& routerValue = root.require("router");
    result.router.config.expertCount =
        asSize(routerValue.require("expert_count"), "router expert_count");
    result.router.config.topK = asSize(routerValue.require("top_k"), "router top_k");
    const auto normalization = routerValue.require("normalization").asString();
    if (normalization == "SOFTMAX") {
        result.router.config.normalization = router::RoutingNormalization::Softmax;
    } else if (normalization == "NONE") {
        result.router.config.normalization = router::RoutingNormalization::None;
    } else {
        throw MetadataError("unsupported router normalization");
    }
    result.router.config.renormalizeSelected =
        routerValue.require("renormalize_selected").asBool();
    result.router.layout = parseLayout(routerValue.require("layout").asString());
    for (const auto& tensorValue : routerValue.require("tensors").asArray()) {
        result.router.tensors.push_back({
            asId(tensorValue.require("layer_id"), "router layer_id"),
            tensorValue.require("tensor").asString()});
    }
    for (const auto& tensorValue : root.require("tensors").asArray()) {
        ManifestTensor value;
        value.name = tensorValue.require("name").asString();
        value.sourceFile = tensorValue.require("source_file").asString();
        value.offset = tensorValue.require("offset").asUInt64();
        value.size = tensorValue.require("size").asUInt64();
        value.dtype = parseDType(tensorValue.require("dtype").asString());
        value.shape = parseShape(tensorValue.require("shape"));
        result.tensors.push_back(std::move(value));
    }
    for (const auto& expertValue : root.require("experts").asArray()) {
        ManifestExpertMapping value;
        value.layerId = asId(expertValue.require("layer_id"), "layer_id");
        value.expertId = asId(expertValue.require("expert_id"), "expert_id");
        value.gate = parseProjection(expertValue.require("gate"));
        value.up = parseProjection(expertValue.require("up"));
        value.down = parseProjection(expertValue.require("down"));
        result.experts.push_back(std::move(value));
    }
    result.validate();
    return result;
}

} // namespace hypermoe::models
