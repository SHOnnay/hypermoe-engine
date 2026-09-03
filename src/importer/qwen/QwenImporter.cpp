#include "importer/qwen/QwenImporter.hpp"

#include "importer/SafeTensors.hpp"
#include "models/metadata/JsonValue.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace hypermoe::importer::qwen {
namespace {

using models::ManifestExpertMapping;
using models::ManifestTensor;
using models::ProjectionLocation;
using models::TensorLayout;
using models::metadata::JsonValue;
using models::metadata::MetadataError;

std::filesystem::path artifactRoot(const std::filesystem::path& artifact) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(artifact, error);
    if (error) throw MetadataError("Qwen artifact does not exist");
    return std::filesystem::is_directory(canonical) ? canonical
                                                     : canonical.parent_path();
}

std::size_t requiredSize(const JsonValue& root, std::string_view field) {
    const auto value = root.require(field).asUInt64();
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw MetadataError("Qwen config field is zero or too large: " +
                            std::string(field));
    }
    return static_cast<std::size_t>(value);
}

std::string detectArchitecture(const JsonValue& config) {
    if (const auto* architectures = config.find("architectures")) {
        for (const auto& value : architectures->asArray()) {
            const auto& name = value.asString();
            if (name.find("Qwen3Moe") != std::string::npos ||
                name.find("Qwen2Moe") != std::string::npos) {
                return name;
            }
        }
    }
    if (const auto* modelType = config.find("model_type")) {
        const auto& value = modelType->asString();
        if (value == "qwen3_moe" || value == "qwen2_moe") return value;
    }
    throw MetadataError("config.json is not a supported Qwen MoE architecture");
}

std::string modelName(const JsonValue& config, const std::filesystem::path& root) {
    if (const auto* name = config.find("_name_or_path")) {
        if (name->isString() && !name->asString().empty()) return name->asString();
    }
    const auto fallback = root.filename().string();
    return fallback.empty() ? "Qwen MoE" : fallback;
}

struct ParsedTensorName {
    enum class Kind { Individual, FusedGateUp, FusedDown, Router };
    Kind kind{};
    std::uint32_t layer{};
    std::optional<std::uint32_t> expert;
    std::string projection;
};

std::optional<ParsedTensorName> parseName(const std::string& name) {
    static const std::regex individual(
        R"(^model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.(gate_proj|up_proj|down_proj)\.weight$)",
        std::regex::ECMAScript | std::regex::optimize);
    static const std::regex fused(
        R"(^model\.layers\.([0-9]+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$)",
        std::regex::ECMAScript | std::regex::optimize);
    static const std::regex router(
        R"(^model\.layers\.([0-9]+)\.mlp\.gate\.weight$)",
        std::regex::ECMAScript | std::regex::optimize);
    std::smatch match;
    const auto checkedId = [](const std::ssub_match& value) {
        const auto parsed = std::stoull(value.str());
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw MetadataError("Qwen tensor name contains an out-of-range ID");
        }
        return static_cast<std::uint32_t>(parsed);
    };
    if (std::regex_match(name, match, individual)) {
        return ParsedTensorName{ParsedTensorName::Kind::Individual,
                                checkedId(match[1]), checkedId(match[2]),
                                match[3].str()};
    }
    if (std::regex_match(name, match, fused)) {
        const auto gateUp = match[2].str() == "gate_up_proj";
        return ParsedTensorName{gateUp ? ParsedTensorName::Kind::FusedGateUp
                                      : ParsedTensorName::Kind::FusedDown,
                                checkedId(match[1]), std::nullopt, match[2].str()};
    }
    if (std::regex_match(name, match, router)) {
        return ParsedTensorName{ParsedTensorName::Kind::Router,
                                checkedId(match[1]), std::nullopt, "router"};
    }
    return std::nullopt;
}

std::uint64_t bytesFor(std::initializer_list<std::size_t> dimensions,
                       tensor::DType dtype) {
    std::uint64_t elements = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0 ||
            elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw MetadataError("Qwen projection size overflow");
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / tensor::sizeOf(dtype)) {
        throw MetadataError("Qwen projection byte size overflow");
    }
    return elements * tensor::sizeOf(dtype);
}

ProjectionLocation wholeProjection(const ManifestTensor& tensor,
                                   const tensor::Shape& expected) {
    if (tensor.shape != expected) {
        throw MetadataError("Qwen projection has an unexpected shape: " + tensor.name);
    }
    return {tensor.name, tensor.offset, tensor.size, tensor.shape,
            TensorLayout::OutputInput};
}

ProjectionLocation sliceProjection(const ManifestTensor& tensor,
                                   std::uint64_t relativeOffset,
                                   std::uint64_t bytes,
                                   tensor::Shape shape) {
    if (relativeOffset > tensor.size || bytes > tensor.size - relativeOffset ||
        tensor.offset > std::numeric_limits<std::uint64_t>::max() - relativeOffset) {
        throw MetadataError("Qwen fused projection slice exceeds tensor");
    }
    return {tensor.name, tensor.offset + relativeOffset, bytes, std::move(shape),
            TensorLayout::OutputInput};
}

} // namespace

std::string_view QwenImporter::name() const noexcept {
    return "Qwen SafeTensors importer";
}

bool QwenImporter::canImport(const std::filesystem::path& artifact) const noexcept {
    std::error_code error;
    const auto root = std::filesystem::is_directory(artifact, error)
                          ? artifact
                          : artifact.parent_path();
    return !error && std::filesystem::is_regular_file(root / "config.json", error) &&
           !error;
}

models::ModelManifest QwenImporter::inspect(
    const std::filesystem::path& artifact) const {
    const auto rootPath = artifactRoot(artifact);
    const auto configJson = models::metadata::parseJsonFile(rootPath / "config.json");
    models::ModelManifest manifest;
    manifest.architecture = models::ModelArchitecture::QWEN_MOE;
    manifest.sourceArchitecture = detectArchitecture(configJson);
    manifest.modelName = modelName(configJson, rootPath);
    manifest.config.modelName = manifest.modelName;
    manifest.config.layerCount = requiredSize(configJson, "num_hidden_layers");
    manifest.config.expertCount = requiredSize(configJson, "num_experts");
    manifest.config.hiddenSize = requiredSize(configJson, "hidden_size");
    manifest.config.intermediateSize =
        requiredSize(configJson, "moe_intermediate_size");
    if (manifest.config.layerCount > std::numeric_limits<std::uint32_t>::max() ||
        manifest.config.expertCount > std::numeric_limits<std::uint32_t>::max() ||
        manifest.config.intermediateSize >
            std::numeric_limits<std::size_t>::max() / 2) {
        throw MetadataError("Qwen model dimensions exceed manifest limits");
    }
    manifest.config.capabilities = {.routedExperts = true,
                                    .configurableTopK = true,
                                    .normalizedRouting = true,
                                    .gatedExpertMlp = true,
                                    .sharedExperts = false,
                                    .quantizedExpertWeights = false};
    if (const auto* sharedSize = configJson.find("shared_expert_intermediate_size")) {
        manifest.config.capabilities.sharedExperts = sharedSize->asUInt64() != 0;
    }
    manifest.router.config.expertCount = manifest.config.expertCount;
    manifest.router.config.topK = requiredSize(configJson, "num_experts_per_tok");
    manifest.router.config.normalization = router::RoutingNormalization::Softmax;
    manifest.router.config.renormalizeSelected =
        configJson.require("norm_topk_prob").asBool();
    manifest.router.layout = TensorLayout::OutputInput;
    manifest.router.config.validate();

    const auto allTensors = SafeTensors::inspectArtifact(artifact);
    manifest.config.capabilities.quantizedExpertWeights =
        std::any_of(allTensors.begin(), allTensors.end(), [](const auto& value) {
            return value.dtype == tensor::DType::INT8;
        });
    std::map<std::pair<std::uint32_t, std::uint32_t>, ManifestExpertMapping> individual;
    std::map<std::uint32_t, const ManifestTensor*> fusedGateUp;
    std::map<std::uint32_t, const ManifestTensor*> fusedDown;
    std::set<std::string> relevantNames;
    for (const auto& tensor : allTensors) {
        const auto parsed = parseName(tensor.name);
        if (!parsed) continue;
        if (parsed->layer >= manifest.config.layerCount) {
            throw MetadataError("Qwen tensor layer exceeds num_hidden_layers");
        }
        relevantNames.insert(tensor.name);
        if (parsed->kind == ParsedTensorName::Kind::Router) {
            const tensor::Shape expected{manifest.config.expertCount,
                                         manifest.config.hiddenSize};
            if (tensor.shape != expected) {
                throw MetadataError("Qwen router tensor shape is incompatible");
            }
            manifest.router.tensors.push_back({parsed->layer, tensor.name});
        } else if (parsed->kind == ParsedTensorName::Kind::FusedGateUp) {
            if (!fusedGateUp.emplace(parsed->layer, &tensor).second) {
                throw MetadataError("duplicate Qwen fused gate/up tensor");
            }
        } else if (parsed->kind == ParsedTensorName::Kind::FusedDown) {
            if (!fusedDown.emplace(parsed->layer, &tensor).second) {
                throw MetadataError("duplicate Qwen fused down tensor");
            }
        } else {
            const auto identity = std::pair{parsed->layer, *parsed->expert};
            if (*parsed->expert >= manifest.config.expertCount) {
                throw MetadataError("Qwen tensor expert exceeds num_experts");
            }
            auto& mapping = individual[identity];
            mapping.layerId = parsed->layer;
            mapping.expertId = *parsed->expert;
            const auto upShape = tensor::Shape{manifest.config.intermediateSize,
                                               manifest.config.hiddenSize};
            const auto downShape = tensor::Shape{manifest.config.hiddenSize,
                                                 manifest.config.intermediateSize};
            auto projection = wholeProjection(
                tensor, parsed->projection == "down_proj" ? downShape : upShape);
            if (parsed->projection == "gate_proj") {
                if (!mapping.gate.tensorName.empty()) throw MetadataError("duplicate gate projection");
                mapping.gate = std::move(projection);
            } else if (parsed->projection == "up_proj") {
                if (!mapping.up.tensorName.empty()) throw MetadataError("duplicate up projection");
                mapping.up = std::move(projection);
            } else {
                if (!mapping.down.tensorName.empty()) throw MetadataError("duplicate down projection");
                mapping.down = std::move(projection);
            }
        }
    }

    std::set<std::uint32_t> moeLayers;
    for (const auto& [identity, mapping] : individual) {
        if (mapping.gate.tensorName.empty() || mapping.up.tensorName.empty() ||
            mapping.down.tensorName.empty()) {
            throw MetadataError("Qwen expert has incomplete individual projections");
        }
        manifest.experts.push_back(mapping);
        moeLayers.insert(identity.first);
    }
    for (const auto& [layer, gateUp] : fusedGateUp) {
        const auto downFound = fusedDown.find(layer);
        if (downFound == fusedDown.end()) {
            throw MetadataError("Qwen fused expert layer is missing down projection");
        }
        if (std::any_of(individual.begin(), individual.end(), [&](const auto& value) {
                return value.first.first == layer;
            })) {
            throw MetadataError("Qwen layer mixes fused and individual expert layouts");
        }
        const auto expectedGateUp = tensor::Shape{
            manifest.config.expertCount, 2 * manifest.config.intermediateSize,
            manifest.config.hiddenSize};
        const auto expectedDown = tensor::Shape{
            manifest.config.expertCount, manifest.config.hiddenSize,
            manifest.config.intermediateSize};
        if (gateUp->shape != expectedGateUp || downFound->second->shape != expectedDown ||
            gateUp->dtype != downFound->second->dtype) {
            throw MetadataError("Qwen fused expert tensors have incompatible metadata");
        }
        const auto projectionBytes = bytesFor(
            {manifest.config.intermediateSize, manifest.config.hiddenSize}, gateUp->dtype);
        const auto downBytes = bytesFor(
            {manifest.config.hiddenSize, manifest.config.intermediateSize}, gateUp->dtype);
        for (std::size_t expert = 0; expert < manifest.config.expertCount; ++expert) {
            ManifestExpertMapping mapping;
            mapping.layerId = layer;
            mapping.expertId = static_cast<std::uint32_t>(expert);
            const auto gateOffset = static_cast<std::uint64_t>(expert) *
                                    2ULL * projectionBytes;
            mapping.gate = sliceProjection(*gateUp, gateOffset, projectionBytes,
                tensor::Shape{manifest.config.intermediateSize, manifest.config.hiddenSize});
            mapping.up = sliceProjection(*gateUp, gateOffset + projectionBytes,
                projectionBytes,
                tensor::Shape{manifest.config.intermediateSize, manifest.config.hiddenSize});
            mapping.down = sliceProjection(*downFound->second,
                static_cast<std::uint64_t>(expert) * downBytes, downBytes,
                tensor::Shape{manifest.config.hiddenSize, manifest.config.intermediateSize});
            manifest.experts.push_back(std::move(mapping));
        }
        moeLayers.insert(layer);
    }
    for (const auto& [layer, tensor] : fusedDown) {
        (void)tensor;
        if (!fusedGateUp.contains(layer)) {
            throw MetadataError("Qwen fused down tensor has no gate/up tensor");
        }
    }
    if (manifest.experts.empty()) throw MetadataError("no Qwen MoE experts were discovered");
    for (const auto layer : moeLayers) {
        const auto count = static_cast<std::size_t>(std::count_if(
            manifest.experts.begin(), manifest.experts.end(),
            [&](const auto& expert) { return expert.layerId == layer; }));
        if (count != manifest.config.expertCount) {
            throw MetadataError("Qwen MoE layer does not contain every declared expert");
        }
        const auto routerName = "model.layers." + std::to_string(layer) + ".mlp.gate.weight";
        if (std::none_of(manifest.router.tensors.begin(),
                         manifest.router.tensors.end(), [&](const auto& value) {
                             return value.layerId == layer &&
                                    value.tensorName == routerName;
                         })) {
            throw MetadataError("Qwen MoE layer is missing its router tensor");
        }
    }
    for (const auto& tensor : allTensors) {
        if (relevantNames.contains(tensor.name)) manifest.tensors.push_back(tensor);
    }
    std::sort(manifest.router.tensors.begin(), manifest.router.tensors.end(),
              [](const auto& left, const auto& right) {
                  return left.layerId < right.layerId;
              });
    std::sort(manifest.experts.begin(), manifest.experts.end(),
              [](const auto& left, const auto& right) {
                  return std::tie(left.layerId, left.expertId) <
                         std::tie(right.layerId, right.expertId);
              });
    manifest.validate();
    return manifest;
}

} // namespace hypermoe::importer::qwen
