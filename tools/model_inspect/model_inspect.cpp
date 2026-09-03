#include "models/metadata/JsonValue.hpp"
#include "models/qwen/QwenMoEAdapter.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {

std::filesystem::path manifestPath(const std::filesystem::path& input) {
    std::error_code error;
    if (std::filesystem::is_directory(input, error) && !error) {
        return input / "metadata.json";
    }
    return input;
}

std::unique_ptr<hypermoe::models::ModelAdapter>
selectAdapter(const std::filesystem::path& input) {
    const auto root = hypermoe::models::metadata::parseJsonFile(manifestPath(input));
    const auto& architecture = root.require("architecture").asString();
    if (architecture == "QWEN_MOE") {
        return std::make_unique<hypermoe::models::qwen::QwenMoEAdapter>();
    }
    throw std::runtime_error("no installed adapter for architecture: " + architecture);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: hypermoe_model_inspect <model-directory|manifest>");
        }
        const std::filesystem::path input(argv[1]);
        const auto adapter = selectAdapter(input);
        const auto metadata = adapter->loadModelMetadata(input);
        const auto mappings = adapter->getExpertMapping(metadata);
        const auto router = adapter->getRouterConfiguration(metadata);
        std::cout << "Model: " << metadata.config.modelName << '\n'
                  << "Architecture: "
                  << hypermoe::models::toString(adapter->getArchitecture()) << '\n'
                  << "Layers: " << adapter->getLayerCount(metadata) << '\n'
                  << "Experts per MoE layer: " << adapter->getExpertCount(metadata) << '\n'
                  << "Router: top-" << router.topK << " "
                  << hypermoe::router::toString(router.normalization) << '\n'
                  << "Expert tensors:\n";
        for (const auto& binding : mappings.entries()) {
            std::cout << "  Layer " << binding.layerId << ", Expert "
                      << binding.expertId << '\n';
            if (binding.gateProjection) {
                std::cout << "    GATE: " << binding.gateProjection->name << '\n';
            }
            if (binding.upProjection) {
                std::cout << "    UP:   " << binding.upProjection->name << '\n';
            }
            if (binding.downProjection) {
                std::cout << "    DOWN: " << binding.downProjection->name << '\n';
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "model inspection failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
