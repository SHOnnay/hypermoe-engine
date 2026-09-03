#include "importer/qwen/QwenImporter.hpp"
#include "tools/model_convert/ExpertPacker.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: hypermoe_model_convert <qwen-artifact> <output-directory>");
        }
        const std::filesystem::path source(argv[1]);
        const std::filesystem::path output(argv[2]);
        hypermoe::importer::qwen::QwenImporter importer;
        const auto manifest = importer.inspect(source);
        const auto start = std::chrono::steady_clock::now();
        const auto report = hypermoe::conversion::ExpertPacker{}.pack(
            manifest, std::filesystem::is_directory(source) ? source : source.parent_path(),
            output);
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "Packed " << report.experts << " experts and "
                  << report.projections << " projections in " << elapsed << " s\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "model conversion failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
