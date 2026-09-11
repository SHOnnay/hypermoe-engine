#include "validation/RealModelValidation.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::vector<std::uint32_t> parseTokens(std::string_view text) {
    std::vector<std::uint32_t> result;
    while (!text.empty()) {
        const auto separator = text.find(',');
        const auto field = text.substr(0, separator);
        std::uint64_t value{};
        const auto [end, error] = std::from_chars(
            field.data(), field.data() + field.size(), value);
        if (field.empty() || error != std::errc{} ||
            end != field.data() + field.size() ||
            value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("token IDs must be comma-separated uint32 values");
        }
        result.push_back(static_cast<std::uint32_t>(value));
        if (separator == std::string_view::npos) break;
        text.remove_prefix(separator + 1);
    }
    if (result.empty()) throw std::invalid_argument("at least one token ID is required");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: hypermoe_real_qwen_validate <runtime-artifact> "
                         "<token-ids-csv> [report.json]\n";
            return 2;
        }
        const auto report = hypermoe::validation::RealModelValidator::validateCpuCuda(
            std::filesystem::path{argv[1]}, parseTokens(argv[2]));
        const auto json = report.toJson();
        if (argc == 4) {
            std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
            output.write(json.data(), static_cast<std::streamsize>(json.size()));
            if (!output) throw std::runtime_error("cannot write validation report");
        }
        std::cout << json;
        return report.cudaAvailable && !report.matches() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "real Qwen validation failed: " << error.what() << '\n';
        return 1;
    }
}
