#include "tests/support/ExpertPipelineFixture.hpp"
#include "backend/CudaBackend.hpp"
#include "tensor/backend/CudaTensorBackend.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "cpu";
        if (argc > 3 || (mode != "cpu" && mode != "cuda")) {
            throw std::invalid_argument("usage: hypermoe_expert_pipeline_benchmark [cpu|cuda] [report.json]");
        }
        std::shared_ptr<hypermoe::tensor::TensorBackend> tensors;
        std::shared_ptr<hypermoe::backend::ComputeBackend> transfers;
        if (mode == "cuda") {
            auto cuda = std::make_shared<hypermoe::tensor::CudaTensorBackend>();
            if (!cuda->available() || !cuda->nativeKernelsAvailable()) {
                throw std::runtime_error("native CUDA kernels unavailable; no GPU numbers produced");
            }
            tensors = cuda;
            transfers = std::make_shared<hypermoe::backend::CudaBackend>();
        }
        std::ofstream output(argc > 2 ? argv[2] : "expert_pipeline_report.json");
        output << std::fixed << std::setprecision(6)
               << "{\n  \"schema\": \"hypermoe.expert-pipeline.v1\",\n"
               << "  \"scope\": \"tiny_INT8_artifact_not_real_Qwen_or_generation\",\n"
               << "  \"device\": \"" << mode << "\",\n  \"runs\": [\n";
        bool first = true;
        for (const auto capacity : {1U, 2U, 4U}) {
            for (const bool overlap : {false, true}) {
                hypermoe::test::ExpertPipelineFixture fixture(
                    overlap, 12U * capacity, mode == "cpu" ? 12U * capacity : 48U,
                    true, tensors, transfers);
                std::uint64_t hits{}, misses{};
                const auto start = std::chrono::steady_clock::now();
                const auto initialTransfers = fixture.compute->stats();
                constexpr unsigned iterations = 100;
                for (unsigned index = 0; index < iterations; ++index) {
                    fixture.select(index % 4, (index + 1) % 4);
                    const auto result = fixture.execute();
                    hits += result.execution.expertCacheHits;
                    misses += result.execution.expertCacheMisses;
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                const auto memory = fixture.memory->snapshot();
                const auto profiler = fixture.profiler->snapshot();
                const auto manager = fixture.experts->stats();
                if (!first) output << ",\n";
                first = false;
                output << "    {\"overlap\": " << (overlap ? "true" : "false")
                       << ", \"expert_capacity\": " << capacity
                       << ", \"total_ms\": " << std::chrono::duration<double, std::milli>(elapsed).count()
                       << ", \"expert_layer_calls_per_second\": " << iterations / std::chrono::duration<double>(elapsed).count()
                       << ", \"cache_hit_rate\": " << static_cast<double>(hits) / static_cast<double>(hits + misses)
                       << ", \"nvme_read_bytes\": " << profiler.nvmeBytes
                       << ", \"host_to_device_bytes\": " << fixture.compute->stats().hostToDeviceBytes - initialTransfers.hostToDeviceBytes
                       << ", \"vram_evictions\": " << manager.vramEvictions
                       << ", \"ram_evictions\": " << manager.ramEvictions
                       << ", \"resident_device_bytes\": " << memory.vram.usedBytes
                       << ", \"resident_ram_bytes\": " << memory.ram.usedBytes << "}";
            }
        }
        output << "\n  ]\n}\n";
        if (!output) throw std::runtime_error("cannot write pipeline benchmark report");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
