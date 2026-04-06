#include "cpu_executor.h"
#include "profiler.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace acr {

// Workload kernel: fused scale-bias-relu, representative of inference preprocessing.
// op: y[i] = max(0, alpha * x[i] + beta)
// Replace with any kernel appropriate to the benchmark (e.g. matrix-vector, reduction).
static constexpr float kAlpha = 1.0f / 255.0f;
static constexpr float kBeta  = -0.5f;

static void kernel_cpu(const float* __restrict__ in,
                       float*       __restrict__ out,
                       size_t                   n) {
    for (size_t i = 0; i < n; ++i) {
        float v = kAlpha * in[i] + kBeta;
        out[i]  = v > 0.0f ? v : 0.0f;
    }
}

static void kernel_cpu_omp(const float* __restrict__ in,
                            float*       __restrict__ out,
                            size_t                   n) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float v = kAlpha * in[i] + kBeta;
        out[i]  = v > 0.0f ? v : 0.0f;
    }
#else
    kernel_cpu(in, out, n);
#endif
}

ExecutionResult CPUExecutor::run_serial(WorkloadDescriptor& wd) {
    WallTimer t;
    kernel_cpu(wd.input_data, wd.output_data, wd.element_count);
    double wall = t.elapsed_ms();

    ExecutionResult r{};
    r.workload_id  = wd.id;
    r.path_used    = ExecutionPath::CPU_SERIAL;
    r.wall_ms      = wall;
    r.kernel_ms    = wall;
    r.h2d_ms       = 0.0;
    r.d2h_ms       = 0.0;
    r.memory_throughput_GBs =
        (static_cast<double>(wd.element_count) * sizeof(float) * 2.0) / (wall * 1e-3 * 1e9);
    r.correct = true;
    return r;
}

ExecutionResult CPUExecutor::run_parallel(WorkloadDescriptor& wd) {
    WallTimer t;
    kernel_cpu_omp(wd.input_data, wd.output_data, wd.element_count);
    double wall = t.elapsed_ms();

    ExecutionResult r{};
    r.workload_id  = wd.id;
    r.path_used    = ExecutionPath::CPU_PARALLEL;
    r.wall_ms      = wall;
    r.kernel_ms    = wall;
    r.h2d_ms       = 0.0;
    r.d2h_ms       = 0.0;
    r.memory_throughput_GBs =
        (static_cast<double>(wd.element_count) * sizeof(float) * 2.0) / (wall * 1e-3 * 1e9);
    r.correct = true;
    return r;
}

} // namespace acr
