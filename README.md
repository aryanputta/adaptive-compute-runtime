# Adaptive Heterogeneous Compute Runtime

A C++17 / CUDA runtime that profiles incoming workloads at submission time and
dynamically selects the optimal execution path — CPU serial, CPU parallel
(OpenMP), GPU direct, or GPU batched — based on arithmetic intensity, estimated
memory transfer cost, and latency requirements.

> **Core insight:** the fastest processor is not always the fastest system.
> A small workload routed to the GPU wastes more time on PCIe transfers than
> the kernel saves. A large compute-bound workload left on the CPU leaves
> peak throughput on the table. This runtime makes that decision automatically
> and benchmarks every choice.

---

## Architecture

```
                  ┌─────────────────────────────────┐
  submit(wd) ───► │         Runtime                 │
                  │                                 │
                  │  FeatureExtractor               │
                  │    ├─ arithmetic intensity      │
                  │    ├─ estimated transfer cost   │
                  │    └─ workload type tag         │
                  │                                 │
                  │  Scheduler (rules engine)       │
                  │    ├─ COMPUTE_BOUND  → GPU      │
                  │    ├─ MEMORY_BOUND   → CPU/GPU  │
                  │    ├─ TRANSFER_BOUND → defer    │
                  │    └─ LATENCY_SENS  → CPU fast  │
                  │                                 │
                  │  Executors                      │
                  │    ├─ CPUExecutor (serial/OMP)  │
                  │    └─ GPUExecutor (stream/batch)│
                  │                                 │
                  │  MemoryManager (buffer pool)    │
                  │  Profiler (CSV + summary)       │
                  └─────────────────────────────────┘
```

### Modules

| Module | File | Responsibility |
|---|---|---|
| `FeatureExtractor` | `src/runtime/feature_extractor.cpp` | Roofline-model cost estimation, workload classification |
| `Scheduler` | `src/scheduler/scheduler.cpp` | Rules-based execution path selection |
| `CPUExecutor` | `src/cpu/cpu_executor.cpp` | Serial and OpenMP parallel CPU paths |
| `GPUExecutor` | `src/gpu/gpu_executor.cpp` | CUDA stream launch, batched coalesced execution |
| `gpu_kernels.cu` | `src/gpu/gpu_kernels.cu` | CUDA kernels (scale-bias-relu, batched variant, reduction) |
| `MemoryManager` | `src/memory/memory_manager.cpp` | Pinned host + device buffer free-list pool |
| `Profiler` | `src/profiling/profiler.cpp` | Per-result timing log, CSV export, summary table |
| `Runtime` | `src/runtime/runtime.cpp` | Top-level orchestration, batch coalescing |

---

## Workload Classification

The `FeatureExtractor` tags each workload before scheduling:

| Type | Condition | Default path |
|---|---|---|
| `COMPUTE_BOUND` | Arithmetic intensity ≥ 16 FLOP/byte | GPU direct |
| `MEMORY_BOUND` | Intensity < 2 FLOP/byte | CPU (small), GPU (large) |
| `TRANSFER_BOUND` | PCIe transfer cost > 50% of GPU compute | CPU or deferred batch |
| `LATENCY_SENSITIVE` | Caller-flagged | CPU parallel or GPU direct (never deferred) |

All thresholds are configurable via `configs/default.json`.

---

## Build

**Requirements:** CMake ≥ 3.18, C++17 compiler. CUDA toolkit optional.

```bash
# Release build — auto-detects CUDA
./scripts/build.sh

# Force CPU-only (no CUDA required)
./scripts/build.sh nocuda

# Debug build
./scripts/build.sh debug
```

Or manually:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DACR_ENABLE_CUDA=ON \       # OFF to skip CUDA
  -DACR_ENABLE_OPENMP=ON \
  -DACR_BUILD_TESTS=ON \
  -DACR_BUILD_BENCH=ON

cmake --build build --parallel $(nproc)
```

---

## Run Tests

```bash
cd build && ctest --output-on-failure
```

26 tests across four suites:

| Suite | Coverage |
|---|---|
| `unit/test_scheduler` | Path selection logic for all workload types and edge cases |
| `unit/test_feature_extractor` | Cost estimation, classification correctness |
| `unit/test_memory_manager` | Pool round-trips, cache hits, flush |
| `integration/test_runtime_e2e` | End-to-end correctness, batch submission, profiler |
| `performance/test_perf_regression` | Adaptive ≤ 2× serial CPU; parallel faster than serial for large input |

---

## Run Benchmark

```bash
./scripts/run_bench.sh

# Analyse results offline
python3 scripts/plot_results.py
```

The benchmark runs five workload profiles (tiny latency-sensitive → giant
streaming) through three runtimes and reports wall time and improvement vs.
the best static baseline:

```
Workload             CPU(ms)    GPU(ms)   Adaptive(ms)  vs Best-Static
------------------------------------------------------------------------
tiny_latency          0.000      0.000        0.000        ...
small_elemwise        0.014      0.014        0.016        ...
large_matmul          2.579      2.524        2.507       +0.7%
giant_streaming      10.115     10.538        9.831       +2.9%
medium_sparse         0.129      0.128        0.159        ...
```

With a CUDA-capable GPU the large compute-bound and giant streaming workloads
show meaningful separation between paths.

---

## Profiling with Nsight

```bash
# Kernel-level profile
ncu --set full ./build/acr_bench

# System-level timeline
nsys profile --trace=cuda,osrt ./build/acr_bench
```

Key metrics to examine: kernel occupancy, achieved memory throughput (vs.
roofline peak), H2D/D2H transfer time vs. kernel time ratio.

---

## Configuration

`configs/default.json` exposes all scheduler thresholds:

```json
{
  "scheduler": {
    "gpu_launch_threshold": 8192,
    "transfer_compute_ratio_threshold": 0.5,
    "memory_bound_intensity_threshold": 2.0,
    "deferred_batch_target": 64
  }
}
```

---

## Extending the Kernel

The current kernel is a fused scale-bias-relu (representative of inference
preprocessing). To substitute your own:

1. Add your CUDA kernel to `src/gpu/gpu_kernels.cu` with a `extern "C"` launcher.
2. Add the matching CPU implementation in `src/cpu/cpu_executor.cpp`.
3. Update `GPUExecutor::run` and `CPUExecutor::run_serial/run_parallel` to call it.
4. The scheduler, profiler, and memory manager require no changes.

---

## Project Structure

```
adaptive-compute-runtime/
├── include/             # All public headers
├── src/
│   ├── runtime/         # Runtime + FeatureExtractor
│   ├── scheduler/       # Execution path selector
│   ├── cpu/             # Serial + OpenMP executors
│   ├── gpu/             # CUDA executor + kernels
│   ├── memory/          # Buffer pool manager
│   └── profiling/       # Timing logger
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── benchmarks/          # Comparison benchmark driver
├── scripts/             # build.sh, run_bench.sh, plot_results.py
├── configs/             # Tunable scheduler thresholds
└── .github/workflows/   # CI (CPU-only build + test)
```

---

## License

MIT — see [LICENSE](LICENSE).
