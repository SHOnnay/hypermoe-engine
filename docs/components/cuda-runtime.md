# CUDA runtime validation

`CudaRuntime` remains the optional owner of one NVIDIA device, its streams, and
its events. `CudaDeviceInfo` is the stable diagnostic record for device ordinal,
name, compute capability, live total/free VRAM, managed stream count, CUDA runtime
version, and driver version. The existing `DeviceInfo` name remains an alias so
Phase 4 callers do not break.

`CudaRuntimeValidator` performs a bounded hardware readiness check. It verifies
device properties, compute capability, VRAM consistency, runtime/driver versions,
the compute/transfer/prefetch stream set, and timed event recording. The result is
serialized as `hypermoe.cuda-validation.v1` and can be embedded in benchmark
reports or written as a standalone hardware report.

Validation has three outcomes: `PASSED`, `FAILED`, and `SKIPPED`. A build without
CUDA, or a CUDA build without a usable NVIDIA device, is `SKIPPED` with an explicit
reason rather than a test failure. CUDA operation failures on an otherwise
available device are `FAILED`. This distinction keeps storage and CPU tooling
portable while preventing missing GPU measurements from being presented as zero-
performance hardware results.
