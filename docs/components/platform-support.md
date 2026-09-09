# Platform support

HyperMoE targets 64-bit little-endian systems with an ISO C++20 standard
library. CMake validates those requirements before configuring the runtime.

## Toolchains

| Platform | Toolchain | CPU runtime | Sanitizers |
|---|---|---:|---|
| macOS | Apple Clang | Supported and locally validated | ASan + UBSan |
| Windows x64 | Visual Studio 2022 MSVC | Supported; PC validation is ongoing | ASan |
| Linux x86-64 | GCC or Clang | Build configuration supported | ASan + UBSan |

The project disables compiler language extensions and applies `/W4
/permissive-` on MSVC or `-Wall -Wextra -Wpedantic -Wconversion` on
Clang/GCC. `HYPERMOE_WARNINGS_AS_ERRORS` optionally promotes that baseline in
CI.

## CUDA

CUDA is optional. `HYPERMOE_ENABLE_CUDA=ON` asks CMake to find the CUDA Toolkit.
If `CUDA::cudart` and `CUDA::cublas` are available, HyperMoE enables its runtime
API, streams, events, memory pool, transfers, and cuBLAS tensor backend. No CUDA
compiler or custom `.cu` kernel is required at this stage. If the toolkit or a
runtime device is unavailable, CPU targets and tests remain usable.

For RTX 4070 validation, use a Windows x64 or Linux x86-64 build with an NVIDIA
driver compatible with the selected toolkit. Runtime validation—not the GPU
model name alone—must confirm device availability, VRAM queries, streams,
events, transfers, and cuBLAS correctness before performance results are used.

## Artifact portability

- `experts.index` uses fixed record widths and explicit little-endian integer
  encoding. Native structure padding is never serialized.
- Manifests are bounded UTF-8 JSON with explicit 64-bit numeric range checks and
  safe relative paths.
- Packed tensor bytes retain SafeTensors' little-endian element encoding.
- Model offsets are `uint64_t`; the supported runtime is intentionally 64-bit.
- mmap/Windows file mappings own OS handles with move-only RAII semantics.
- Tensor owners and views validate dtype alignment, storage size, device enum,
  device ordinal, and lifetime before typed access.

Big-endian hosts and 32-bit processes are rejected during CMake configuration.
Supporting either later requires an explicit tensor byte-swap/addressing design,
not a relaxation of these checks.

## Current validation boundary

This document describes supported contracts, not a claim that every hardware
combination has been exercised. Apple Clang CPU tests are run locally. Windows,
MSVC, and RTX 4070 execution are validated separately on the PC environment;
Linux and CUDA performance numbers must be produced on their actual hosts.
