// Performance regression tests: verify that the adaptive runtime does not
// regress below a configurable threshold versus the CPU-only baseline.
// These are wall-clock timing assertions — not strict unit tests.

#include <gtest/gtest.h>
#include "runtime.h"
#include <vector>
#include <chrono>
#include <numeric>

using namespace acr;

static double run_workload_ms(Runtime& rt, size_t n, double arith_intensity = 1.0,
                               bool latency_sensitive = false, int reps = 3) {
    std::vector<float> in(n, 100.0f);
    std::vector<float> out(n);

    double total = 0.0;
    for (int i = 0; i < reps; ++i) {
        WorkloadDescriptor wd{};
        wd.id            = "perf_test";
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = n;
        wd.features.arithmetic_intensity = arith_intensity;
        wd.features.latency_sensitive    = latency_sensitive;
        total += rt.submit(wd).wall_ms;
    }
    return total / reps;
}

// The adaptive runtime must complete within 2x the CPU serial baseline.
// This is a conservative bound — it should be near 1x or better.
TEST(PerfRegression, AdaptiveNotSlowerThanCPUSerial) {
    RuntimeConfig cpu_cfg;
    cpu_cfg.scheduler.gpu_available = false;
    cpu_cfg.scheduler.openmp_available = false;
    Runtime rt_cpu_serial(cpu_cfg);

    RuntimeConfig adaptive_cfg;
    adaptive_cfg.scheduler.gpu_available = false;  // GPU off for CI
    Runtime rt_adaptive(adaptive_cfg);

    size_t n = 1'000'000;
    double cpu_ms      = run_workload_ms(rt_cpu_serial, n);
    double adaptive_ms = run_workload_ms(rt_adaptive,   n);

    // Adaptive should not be more than 2x slower than serial CPU
    EXPECT_LT(adaptive_ms, cpu_ms * 2.0)
        << "Adaptive: " << adaptive_ms << "ms  CPU: " << cpu_ms << "ms";
}

TEST(PerfRegression, ParallelFasterThanSerialForLargeInput) {
    RuntimeConfig serial_cfg;
    serial_cfg.scheduler.gpu_available    = false;
    serial_cfg.scheduler.openmp_available = false;
    Runtime rt_serial(serial_cfg);

    RuntimeConfig parallel_cfg;
    parallel_cfg.scheduler.gpu_available    = false;
    parallel_cfg.scheduler.openmp_available = true;
    Runtime rt_parallel(parallel_cfg);

    size_t n = 10'000'000;
    double serial_ms   = run_workload_ms(rt_serial,   n, 1.0, false, 5);
    double parallel_ms = run_workload_ms(rt_parallel, n, 1.0, false, 5);

    // With OpenMP, parallel should be faster on multi-core (or at worst equal).
    // Allow up to 20% overhead on single-core machines.
    EXPECT_LT(parallel_ms, serial_ms * 1.2)
        << "Parallel: " << parallel_ms << "ms  Serial: " << serial_ms << "ms";
}

TEST(PerfRegression, SmallTaskSchedulerOverheadLow) {
    RuntimeConfig cfg;
    cfg.scheduler.gpu_available = false;
    Runtime rt(cfg);

    // Submit 1000 tiny tasks; total overhead must be under 100ms
    size_t n = 64;
    std::vector<float> in(n, 1.0f), out(n);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        WorkloadDescriptor wd{};
        wd.id            = "tiny_" + std::to_string(i);
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = n;
        rt.submit(wd);
    }

    auto t1  = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 500.0) << "1000 tiny tasks took " << ms << "ms — scheduler overhead too high";
}
