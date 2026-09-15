#include "tests/support/ExpertPipelineFixture.hpp"
#include "cache/HybridPolicy.hpp"
#include "backend/CudaBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "cpu";
        if (argc > 3 || (mode != "cpu" && mode != "cuda")) {
            throw std::invalid_argument("usage: hypermoe_residency_benchmark [cpu|cuda] [report.json]");
        }
        std::shared_ptr<hypermoe::tensor::TensorBackend> tensors;
        std::shared_ptr<hypermoe::backend::ComputeBackend> transfer;
        if (mode == "cuda") {
            auto cuda = std::make_shared<hypermoe::tensor::CudaTensorBackend>();
            if (!cuda->available() || !cuda->nativeKernelsAvailable()) {
                throw std::runtime_error("native CUDA unavailable; no GPU measurements produced");
            }
            tensors = cuda;
            transfer = std::make_shared<hypermoe::backend::CudaBackend>();
        }
        std::ofstream output(argc > 2 ? argv[2] : "residency_report.json");
        output << std::fixed << std::setprecision(6)
               << "{\n  \"schema\": \"hypermoe.residency.v1\",\n"
               << "  \"scope\": \"tiny_INT8_fixture_not_real_Qwen_or_generation\",\n"
               << "  \"device\": \"" << mode << "\",\n  \"runs\": [\n";
        bool first = true;
        for (const auto capacity : {1U, 2U, 4U}) {
            for (const bool adaptive : {false, true}) {
                std::unique_ptr<hypermoe::CachePolicy> policy;
                if (adaptive) policy = std::make_unique<hypermoe::HybridPolicy>(true);
                hypermoe::test::ExpertPipelineFixture fixture(
                    true, 12U * capacity, mode == "cpu" ? 12U * capacity : 48U,
                    true, tensors, transfer, std::move(policy));
                std::uint64_t hits{}, misses{};
                const auto initial = fixture.compute->stats();
                const auto start = std::chrono::steady_clock::now();
                for (unsigned i = 0; i < 100; ++i) {
                    // Hot expert 0 plus a changing second expert; deterministic.
                    fixture.select(0, 1U + i % 3U);
                    const auto result = fixture.execute();
                    hits += result.execution.expertCacheHits;
                    misses += result.execution.expertCacheMisses;
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                const auto memory = fixture.memory->snapshot();
                const auto stats = fixture.experts->stats();
                std::size_t residents{};
                for (const auto& expert : fixture.experts->residencySnapshot()) {
                    if (expert.location == (mode == "cpu" ? hypermoe::MemoryTier::Ram
                                                           : hypermoe::MemoryTier::Vram)) ++residents;
                }
                if (!first) output << ",\n";
                first = false;
                output << "    {\"policy\": \"" << (adaptive ? "age_aware_hybrid" : "lru") << "\""
                       << ", \"expert_capacity\": " << capacity
                       << ", \"execution_tier_budget_bytes\": " << (mode == "cpu" ? memory.ram.limitBytes : memory.vram.limitBytes)
                       << ", \"total_ms\": " << std::chrono::duration<double, std::milli>(elapsed).count()
                       << ", \"cache_hit_rate\": " << static_cast<double>(hits) / static_cast<double>(hits + misses)
                       << ", \"resident_experts\": " << residents
                       << ", \"resident_device_bytes\": " << memory.vram.usedBytes
                       << ", \"resident_ram_bytes\": " << memory.ram.usedBytes
                       << ", \"nvme_transfer_bytes\": " << fixture.profiler->snapshot().nvmeBytes
                       << ", \"ram_to_vram_transfer_bytes\": " << fixture.compute->stats().hostToDeviceBytes - initial.hostToDeviceBytes
                       << ", \"vram_promotions\": " << stats.vramPromotions
                       << ", \"vram_evictions\": " << stats.vramEvictions
                       << ", \"ram_evictions\": " << stats.ramEvictions << "}";
            }
        }
        output << "\n  ]\n}\n";
        if (!output) throw std::runtime_error("cannot write residency benchmark report");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
