cmake_minimum_required(VERSION 3.20)

project(HyperMoE VERSION 0.4.0 LANGUAGES CXX)

option(HYPERMOE_BUILD_TESTS "Build HyperMoE tests" ON)
option(HYPERMOE_BUILD_BENCHMARKS "Build HyperMoE simulation benchmarks" ON)
option(HYPERMOE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(HYPERMOE_ENABLE_CUDA "Enable CUDA backend when the toolkit is available" ON)

find_package(Threads REQUIRED)
if(HYPERMOE_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

add_library(hypermoe
    src/memory/vram_manager/memory_manager.cpp
    src/experts/expert_manager/expert_manager.cpp
    src/experts/expert_cache/lru_cache_policy.cpp
    src/storage/ExpertIndex.cpp
    src/storage/MappedFile.cpp
    src/storage/ExpertStore.cpp
    src/storage/DiskLoader.cpp
    src/memory/TransferManager.cpp
    src/memory/MemoryPressureController.cpp
    src/cache/LRUPolicy.cpp
    src/cache/LFUPolicy.cpp
    src/cache/HybridPolicy.cpp
    src/profiling/Profiler.cpp
    src/backend/Backend.cpp
    src/backend/CpuBackend.cpp
    src/backend/cuda/CudaMemoryManager.cpp
    src/backend/cuda/CudaRuntime.cpp
    src/backend/cuda/CudaMemoryPool.cpp
    src/backend/cuda/CudaStreamManager.cpp
    src/memory/PinnedBuffer.cpp
    src/hardware/HardwareInfo.cpp
    src/scheduler/ExpertState.cpp
    src/scheduler/RuntimeEvent.cpp
    src/scheduler/Prefetcher.cpp
    src/scheduler/Scheduler.cpp
)
add_library(HyperMoE::hypermoe ALIAS hypermoe)

target_compile_features(hypermoe PUBLIC cxx_std_20)
target_include_directories(hypermoe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(hypermoe PUBLIC Threads::Threads)

if(CUDAToolkit_FOUND)
    target_compile_definitions(hypermoe PUBLIC HYPERMOE_HAS_CUDA=1)
    target_link_libraries(hypermoe PUBLIC CUDA::cudart)
    message(STATUS "HyperMoE CUDA backend enabled")
else()
    message(STATUS "HyperMoE CUDA backend disabled; CPU fallback enabled")
endif()

if(MSVC)
    target_compile_options(hypermoe PRIVATE /W4 /permissive-)
else()
    target_compile_options(hypermoe PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(HYPERMOE_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(hypermoe PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(hypermoe PUBLIC -fsanitize=address,undefined)
endif()

if(HYPERMOE_BUILD_TESTS)
    include(CTest)
    add_executable(hypermoe_tests tests/phase1_tests.cpp)
    target_link_libraries(hypermoe_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase1 COMMAND hypermoe_tests)

    add_executable(hypermoe_phase2_tests tests/phase2_tests.cpp)
    target_link_libraries(hypermoe_phase2_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase2 COMMAND hypermoe_phase2_tests)

    add_executable(hypermoe_phase2_integration tests/phase2_integration_tests.cpp)
    target_link_libraries(hypermoe_phase2_integration PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase2_integration COMMAND hypermoe_phase2_integration)

    add_executable(hypermoe_phase3_tests tests/phase3_tests.cpp)
    target_link_libraries(hypermoe_phase3_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase3 COMMAND hypermoe_phase3_tests)

    add_executable(hypermoe_scheduler_tests tests/scheduler_tests.cpp)
    target_link_libraries(hypermoe_scheduler_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_scheduler COMMAND hypermoe_scheduler_tests)

    add_executable(hypermoe_phase4_tests tests/phase4_tests.cpp)
    target_link_libraries(hypermoe_phase4_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase4 COMMAND hypermoe_phase4_tests)
endif()

if(HYPERMOE_BUILD_BENCHMARKS)
    add_executable(hypermoe_simulation tools/phase1_simulation.cpp)
    target_link_libraries(hypermoe_simulation PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_phase2_simulation tools/phase2_simulation.cpp)
    target_link_libraries(hypermoe_phase2_simulation PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_hardware_benchmark
        benchmarks/hardware/hardware_benchmark.cpp)
    target_link_libraries(hypermoe_hardware_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_pipeline_benchmark
        benchmarks/pipeline/pipeline_benchmark.cpp)
    target_link_libraries(hypermoe_pipeline_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_cuda_benchmark
        benchmarks/cuda/cuda_benchmark.cpp)
    target_link_libraries(hypermoe_cuda_benchmark PRIVATE HyperMoE::hypermoe)
endif()
