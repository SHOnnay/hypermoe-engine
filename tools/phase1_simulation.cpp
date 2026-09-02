#include "hypermoe/experts/expert_manager.hpp"

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t MiB = 1024 * 1024;

struct Options {
    std::size_t requests{10'000};
    std::uint32_t seed{42};
    std::size_t vramMiB{256};
    std::size_t ramMiB{768};
};

template <typename T>
T parseNumber(std::string_view value, std::string_view name) {
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto equals = argument.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("expected --name=value argument");
        }
        const auto name = argument.substr(0, equals);
        const auto value = argument.substr(equals + 1);
        if (name == "--requests") options.requests = parseNumber<std::size_t>(value, name);
        else if (name == "--seed") options.seed = parseNumber<std::uint32_t>(value, name);
        else if (name == "--vram-mib") options.vramMiB = parseNumber<std::size_t>(value, name);
        else if (name == "--ram-mib") options.ramMiB = parseNumber<std::size_t>(value, name);
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    return options;
}

double percent(std::uint64_t value, std::uint64_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(value) / static_cast<double>(total);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        hypermoe::MemoryManager memory(options.vramMiB * MiB, options.ramMiB * MiB);
        hypermoe::ExpertManager experts(
            memory, std::make_unique<hypermoe::LruCachePolicy>());

        constexpr hypermoe::ExpertId expertCount = 100;
        constexpr std::size_t expertSize = 16 * MiB;
        for (hypermoe::ExpertId id = 0; id < expertCount; ++id) {
            experts.registerExpert({id, id / 10, expertSize,
                                    hypermoe::QuantizationType::Q4,
                                    hypermoe::MemoryTier::Nvme});
        }

        std::mt19937 generator(options.seed);
        // Zipf-like hot set: 80% of routing decisions target 20% of experts.
        std::bernoulli_distribution chooseHot(0.80);
        std::uniform_int_distribution<hypermoe::ExpertId> hotExpert(0, 19);
        std::uniform_int_distribution<hypermoe::ExpertId> coldExpert(20, 99);
        for (std::size_t request = 0; request < options.requests; ++request) {
            const auto id = chooseHot(generator) ? hotExpert(generator) : coldExpert(generator);
            (void)experts.requestExpert(id);
        }

        const auto stats = experts.stats();
        const auto usage = memory.snapshot();
        std::cout << "HyperMoE Phase 1 simulation\n"
                  << "  experts:                 " << expertCount << '\n'
                  << "  requests:                " << stats.requests << '\n'
                  << std::fixed << std::setprecision(2)
                  << "  VRAM hit rate:           " << percent(stats.vramHits, stats.requests) << "%\n"
                  << "  RAM hit rate:            " << percent(stats.ramHits, stats.requests) << "%\n"
                  << "  NVMe load rate:          " << percent(stats.nvmeLoads, stats.requests) << "%\n"
                  << "  simulated load latency:  " << stats.simulatedLoadingLatencyMs << " ms\n"
                  << "  average request latency: "
                  << (stats.requests == 0 ? 0.0 : stats.simulatedLoadingLatencyMs /
                                                 static_cast<double>(stats.requests))
                  << " ms\n"
                  << "  NVMe bytes read:         " << stats.nvmeBytesRead / MiB << " MiB\n"
                  << "  VRAM usage:              " << usage.vram.usedBytes / MiB << " / "
                  << usage.vram.limitBytes / MiB << " MiB\n"
                  << "  RAM usage:               " << usage.ram.usedBytes / MiB << " / "
                  << usage.ram.limitBytes / MiB << " MiB\n"
                  << "  VRAM evictions:          " << stats.vramEvictions << '\n'
                  << "  RAM evictions:           " << stats.ramEvictions << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "simulation failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
