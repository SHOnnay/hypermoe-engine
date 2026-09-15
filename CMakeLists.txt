cmake_minimum_required(VERSION 3.20)

project(HyperMoE VERSION 0.22.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HYPERMOE_BUILD_TESTS "Build HyperMoE tests" ON)
option(HYPERMOE_BUILD_BENCHMARKS "Build HyperMoE simulation benchmarks" ON)
option(HYPERMOE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(HYPERMOE_ENABLE_CUDA "Enable CUDA backend when the toolkit is available" ON)
option(HYPERMOE_WARNINGS_AS_ERRORS "Treat HyperMoE compiler warnings as errors" OFF)

if(CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR "HyperMoE requires a 64-bit target to address model artifacts")
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[
    #include <bit>
    #include <climits>
    #include <cstdint>
    #include <span>
    static_assert(CHAR_BIT == 8);
    static_assert(sizeof(std::uint32_t) == 4);
    static_assert(sizeof(std::uint64_t) == 8);
    static_assert(std::endian::native == std::endian::little);
    int main() { int value = 0; std::span<int> values(&value, 1); return values[0]; }
]] HYPERMOE_PORTABLE_CXX20_RUNTIME)
if(NOT HYPERMOE_PORTABLE_CXX20_RUNTIME)
    message(FATAL_ERROR
        "HyperMoE requires a C++20 standard library, 8-bit bytes, and little-endian byte order")
endif()

message(STATUS "HyperMoE platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} (64-bit)")
message(STATUS "HyperMoE compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

find_package(Threads REQUIRED)
set(HYPERMOE_CUDA_ENABLED OFF)
set(HYPERMOE_CUDA_KERNELS_ENABLED OFF)
if(HYPERMOE_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(HYPERMOE_CUDA_ENABLED ON)
        message(STATUS
            "HyperMoE CUDA: toolkit ${CUDAToolkit_VERSION}; runtime API and cuBLAS enabled")
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(HYPERMOE_CUDA_KERNELS_ENABLED ON)
            message(STATUS
                "HyperMoE CUDA kernels: compiler ${CMAKE_CUDA_COMPILER_VERSION} enabled")
        else()
            message(STATUS
                "HyperMoE CUDA kernels: compiler not found; cuBLAS/staged fallback enabled")
        endif()
    else()
        message(STATUS
            "HyperMoE CUDA: requested, but CUDAToolkit was not found; CPU fallback enabled")
    endif()
else()
    message(STATUS "HyperMoE CUDA: disabled by HYPERMOE_ENABLE_CUDA=OFF")
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
    # Generate PDBs in every configuration so crash dumps stay debuggable.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
    add_link_options($<$<CONFIG:Debug>:/DEBUG>)
    # Stack-frame and uninitialized-variable runtime checks in Debug builds.
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/RTC1>)
    if(HYPERMOE_WARNINGS_AS_ERRORS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>)
    if(HYPERMOE_WARNINGS_AS_ERRORS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
endif()

if(HYPERMOE_ENABLE_SANITIZERS)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>
                            $<$<COMPILE_LANGUAGE:CXX>:/Zi>)
        add_link_options(/INCREMENTAL:NO)
        message(STATUS
            "HyperMoE sanitizers: MSVC AddressSanitizer enabled; UBSan is unavailable on MSVC")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                            $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "HyperMoE sanitizers: AddressSanitizer and UBSan enabled")
    else()
        message(FATAL_ERROR
            "HYPERMOE_ENABLE_SANITIZERS is unsupported by this compiler")
    endif()
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
    src/profiling/RealModelProfile.cpp
    src/backend/Backend.cpp
    src/backend/CpuBackend.cpp
    src/backend/cuda/CudaMemoryManager.cpp
    src/backend/cuda/CudaDeviceInfo.cpp
    src/backend/cuda/CudaRuntime.cpp
    src/backend/cuda/CudaRuntimeValidator.cpp
    src/backend/cuda/CudaMemoryPool.cpp
    src/backend/cuda/CudaStreamManager.cpp
    src/memory/PinnedBuffer.cpp
    src/hardware/HardwareInfo.cpp
    src/scheduler/ExpertState.cpp
    src/scheduler/RuntimeEvent.cpp
    src/scheduler/Prefetcher.cpp
    src/scheduler/Scheduler.cpp
    src/tensor/Shape.cpp
    src/tensor/Tensor.cpp
    src/tensor/TensorView.cpp
    src/tensor/activation/Activation.cpp
    src/tensor/quantization/Quantization.cpp
    src/tensor/quantization/QuantizedTensor.cpp
    src/tensor/backend/CpuTensorBackend.cpp
    src/tensor/backend/CudaTensorBackend.cpp
    src/experts/ExpertExecutor.cpp
    src/experts/ExpertBatch.cpp
    src/runtime/InferenceContext.cpp
    src/runtime/metrics/RuntimeMetrics.cpp
    src/runtime/cache/KVCache.cpp
    src/runtime/cache/CudaKVCache.cpp
    src/runtime/cache/KVCacheManager.cpp
    src/runtime/generation/ForwardState.cpp
    src/runtime/generation/GenerationModel.cpp
    src/runtime/generation/GenerationState.cpp
    src/runtime/generation/InferenceSession.cpp
    src/runtime/generation/Decoder.cpp
    src/tokenizer/qwen/QwenTokenizerAdapter.cpp
    src/generation/LogitsProcessor.cpp
    src/generation/Sampler.cpp
    src/generation/Generator.cpp
    src/models/ExpertWeightMap.cpp
    src/models/ModelManifest.cpp
    src/models/runtime/ModelArchitecture.cpp
    src/models/runtime/TransformerModelRuntime.cpp
    src/models/runtime/ModelRuntime.cpp
    src/models/runtime/PackedModelRuntime.cpp
    src/models/metadata/JsonValue.cpp
    src/models/qwen/QwenMoEAdapter.cpp
    src/importer/SafeTensors.cpp
    src/importer/SafeTensorShardManager.cpp
    src/importer/validation/CheckpointValidator.cpp
    src/importer/qwen/QwenImporter.cpp
    src/importer/qwen/QwenCheckpointLoader.cpp
    src/router/RouterConfig.cpp
    src/router/Router.cpp
    src/router/CpuRouterBackend.cpp
    src/router/CudaRouterBackend.cpp
    src/prediction/ExpertHistory.cpp
    src/prediction/TransitionDatabase.cpp
    src/prediction/ExpertPredictor.cpp
    src/core/runtime/MoERuntime.cpp
    src/tensor/precision/DTypeConverter.cpp
    src/tensor/quantization/QuantizationPolicy.cpp
    src/transformer/MoELayer.cpp
    src/transformer/attention/CpuAttention.cpp
    src/transformer/attention/CudaAttention.cpp
    src/transformer/norm/RMSNorm.cpp
    src/transformer/position/RoPE.cpp
    src/transformer/runtime/TransformerBlock.cpp
    src/transformer/embedding/Embedding.cpp
    src/transformer/output/FinalNorm.cpp
    src/transformer/output/LMHead.cpp
    tools/model_convert/WeightConverter.cpp
    tools/model_convert/ExpertPacker.cpp
    tools/model_convert/real_checkpoint/RealCheckpointConverter.cpp
    src/validation/CorrectnessOracle.cpp
    src/validation/RealModelValidation.cpp
)
add_library(HyperMoE::hypermoe ALIAS hypermoe)

target_compile_features(hypermoe PUBLIC cxx_std_20)
target_include_directories(hypermoe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(hypermoe PUBLIC Threads::Threads)

if(HYPERMOE_CUDA_ENABLED)
    target_compile_definitions(hypermoe PUBLIC
        HYPERMOE_HAS_CUDA=1
        HYPERMOE_HAS_CUBLAS=1)
    target_link_libraries(hypermoe PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(HYPERMOE_CUDA_KERNELS_ENABLED)
    target_sources(hypermoe PRIVATE src/backend/cuda/CudaKernels.cu)
    target_compile_definitions(hypermoe PRIVATE HYPERMOE_HAS_CUDA_KERNELS=1)
    set_target_properties(hypermoe PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
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

    add_executable(hypermoe_phase5_tests tests/phase5_tests.cpp)
    target_link_libraries(hypermoe_phase5_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase5 COMMAND hypermoe_phase5_tests)

    add_executable(hypermoe_phase6_tests tests/phase6_tests.cpp)
    target_link_libraries(hypermoe_phase6_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase6 COMMAND hypermoe_phase6_tests)

    add_executable(hypermoe_phase7_tests tests/phase7_tests.cpp)
    target_link_libraries(hypermoe_phase7_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase7 COMMAND hypermoe_phase7_tests)

    add_executable(hypermoe_phase8_tests tests/phase8_tests.cpp)
    target_link_libraries(hypermoe_phase8_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase8 COMMAND hypermoe_phase8_tests)

    add_executable(hypermoe_phase9_tests tests/phase9_tests.cpp)
    target_link_libraries(hypermoe_phase9_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase9 COMMAND hypermoe_phase9_tests)

    add_executable(hypermoe_phase10_tests tests/real_moe_layer_tests.cpp)
    target_link_libraries(hypermoe_phase10_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase10 COMMAND hypermoe_phase10_tests)

    add_executable(hypermoe_phase11_tests tests/cuda_real_expert_tests.cpp)
    target_link_libraries(hypermoe_phase11_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase11 COMMAND hypermoe_phase11_tests)

    add_executable(hypermoe_phase12_tests tests/phase12_tests.cpp)
    target_link_libraries(hypermoe_phase12_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase12 COMMAND hypermoe_phase12_tests)

    add_executable(hypermoe_phase13_tests tests/phase13_tests.cpp)
    target_link_libraries(hypermoe_phase13_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase13 COMMAND hypermoe_phase13_tests)

    add_executable(hypermoe_phase14_tests tests/phase14_tests.cpp)
    target_link_libraries(hypermoe_phase14_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase14 COMMAND hypermoe_phase14_tests)

    add_executable(hypermoe_platform_tests tests/platform_tests.cpp)
    target_link_libraries(hypermoe_platform_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_platform COMMAND hypermoe_platform_tests)

    add_executable(hypermoe_phase15_16_tests tests/phase15_16_tests.cpp)
    target_link_libraries(hypermoe_phase15_16_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase15_16 COMMAND hypermoe_phase15_16_tests)

    add_executable(hypermoe_phase17_tests tests/phase17_tests.cpp)
    target_link_libraries(hypermoe_phase17_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase17 COMMAND hypermoe_phase17_tests)

    add_executable(hypermoe_phase18_19_tests tests/phase18_19_tests.cpp)
    target_link_libraries(hypermoe_phase18_19_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase18_19 COMMAND hypermoe_phase18_19_tests)

    add_executable(hypermoe_phase20_tests tests/phase20_tests.cpp)
    target_link_libraries(hypermoe_phase20_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase20 COMMAND hypermoe_phase20_tests)

    add_executable(hypermoe_phase21_tests tests/phase21_tests.cpp)
    target_link_libraries(hypermoe_phase21_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase21 COMMAND hypermoe_phase21_tests)

    add_executable(hypermoe_phase22_tests tests/phase22_tests.cpp)
    target_link_libraries(hypermoe_phase22_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase22 COMMAND hypermoe_phase22_tests)
    add_executable(hypermoe_phase22b_tests tests/phase22b_tests.cpp)
    target_link_libraries(hypermoe_phase22b_tests PRIVATE HyperMoE::hypermoe)
    add_test(NAME hypermoe_phase22b COMMAND hypermoe_phase22b_tests)
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

    add_executable(hypermoe_tensor_benchmark
        benchmarks/tensor/tensor_benchmark.cpp)
    target_link_libraries(hypermoe_tensor_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_expert_benchmark
        benchmarks/expert/expert_benchmark.cpp)
    target_link_libraries(hypermoe_expert_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_router_benchmark
        benchmarks/router/router_benchmark.cpp)
    target_link_libraries(hypermoe_router_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_end_to_end_benchmark
        benchmarks/end_to_end/end_to_end_benchmark.cpp)
    target_link_libraries(hypermoe_end_to_end_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_real_expert_benchmark
        benchmarks/real_expert/real_expert_benchmark.cpp)
    target_link_libraries(hypermoe_real_expert_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_model_validation_benchmark
        benchmarks/model_validation/model_validation_benchmark.cpp)
    target_link_libraries(hypermoe_model_validation_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_cuda_runtime_benchmark
        benchmarks/cuda_runtime/cuda_runtime_benchmark.cpp)
    target_link_libraries(hypermoe_cuda_runtime_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_transformer_benchmark
        benchmarks/transformer/transformer_benchmark.cpp)
    target_link_libraries(hypermoe_transformer_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_model_runtime_benchmark
        benchmarks/model_runtime/model_runtime_benchmark.cpp)
    target_link_libraries(hypermoe_model_runtime_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_forward_benchmark
        benchmarks/forward/forward_benchmark.cpp)
    target_link_libraries(hypermoe_forward_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_generation_benchmark
        benchmarks/generation/generation_benchmark.cpp)
    target_link_libraries(hypermoe_generation_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_gpu_inference_benchmark
        benchmarks/gpu_inference/gpu_inference_benchmark.cpp)
    target_link_libraries(hypermoe_gpu_inference_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_real_model_benchmark
        benchmarks/real_model/real_model_benchmark.cpp)
    target_link_libraries(hypermoe_real_model_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_profile_real_model
        benchmarks/profile_real_model/profile_real_model_benchmark.cpp)
    target_link_libraries(hypermoe_profile_real_model PRIVATE HyperMoE::hypermoe)
    add_executable(hypermoe_expert_pipeline_benchmark
        benchmarks/pipeline/expert_pipeline_benchmark.cpp)
    target_link_libraries(hypermoe_expert_pipeline_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_prediction_benchmark
        benchmarks/prediction/prediction_benchmark.cpp)
    target_link_libraries(hypermoe_prediction_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_runtime_optimization_benchmark
        benchmarks/optimization/runtime_optimization_benchmark.cpp)
    target_link_libraries(hypermoe_runtime_optimization_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_quantized_expert_benchmark
        benchmarks/quantization/quantized_expert_benchmark.cpp)
    target_link_libraries(hypermoe_quantized_expert_benchmark PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_real_checkpoint_convert
        tools/model_convert/real_checkpoint/main.cpp)
    target_link_libraries(hypermoe_real_checkpoint_convert PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_real_qwen_validate
        tools/model_validate/real_qwen_validate.cpp)
    target_link_libraries(hypermoe_real_qwen_validate PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_model_inspect
        tools/model_inspect/model_inspect.cpp)
    target_link_libraries(hypermoe_model_inspect PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_model_import
        tools/model_import/model_import.cpp)
    target_link_libraries(hypermoe_model_import PRIVATE HyperMoE::hypermoe)

    add_executable(hypermoe_model_convert
        tools/model_convert/model_convert.cpp)
    target_link_libraries(hypermoe_model_convert PRIVATE HyperMoE::hypermoe)
endif()
