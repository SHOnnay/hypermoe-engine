#include "importer/qwen/QwenImporter.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: hypermoe_model_import <artifact> <manifest-output>");
        }
        const std::filesystem::path artifact(argv[1]);
        const std::filesystem::path output(argv[2]);
        hypermoe::importer::qwen::QwenImporter importer;
        if (!importer.canImport(artifact)) {
            throw std::runtime_error("no installed importer recognizes the artifact");
        }
        const auto manifest = importer.importModel(artifact, output);
        std::cout << "Imported " << manifest.modelName << " ("
                  << manifest.experts.size() << " expert mappings)\n"
                  << "Manifest: " << output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "model import failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
