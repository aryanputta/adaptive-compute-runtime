// Performance regression tests: verify that the adaptive runtime does not
// regress below a configurable threshold versus the CPU-only baseline.
//
// Methodology:
//   - WARMUP_REPS warm-up iterations are discarded before measurement.
//   - MEASURE_REPS iterations are collected; mean and stddev are computed.
//   - Assertions check mean + 2*stddev (i.e., ~95th pct) against the bound,
//     so a single slow outlier does not cause a spurious failure but systematic
//     regressions are caught.
//   - Bounds are tightened from the original 2.0x to 1.5x for the adaptive
//     vs. serial comparison; the parallel test uses a 1.1x threshold.

#include <gtest/gtest.h>
#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace acr;

static constexpr int WARMUP_REPS  = 3;
static constexpr int MEASURE_REPS = 15;

struct RunStats {
    double mean   = 0.0;
    double stddev = 0.0;
};

static RunStats run_workload(Runtime& rt, size_t n, double arith_intensity = 1.0,
                              bool latency_sensitive = false) {
    std::vector<float> in(n, 100.0f);
    std::vector<float> out(n);

    auto submit = [&]() {
        WorkloadDescriptor wd{};
        wd.id            = "perf_test";
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = n;
        wd.features.arithmetic_intensity = arith_intensity;
        wd.features.latency_sensitive    = latency_sensitive;
        return rt.submit(wd).wall_ms;
    };

    // Warm-up — discard
    for (int i = 0; i < WARMUP_REPS; ++i) submit();

    // Measure
    std::vector<double> samples;
    samples.reserve(MEASURE_REPS);
    for (int i = 0; i < MEASURE_REPS; ++i) samples.push_back(submit());

    RunStats s;
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double sq = 0.0;
    for (double x : samples) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / samples.size());
    return s;
}

// The adaptive runtime's mean + 2σ must be within 1.5x of the CPU serial baseline mean.
// (Original bound was 2.0x — tightened now that we have proper warm-up.)
TEST(PerfRegression, AdaptiveNotSlowerThanCPUSerial) {
    RuntimeConfig cpu_cfg;
    cpu_cfg.scheduler.gpu_available    = false;
    cpu_cfg.scheduler.openmp_available = false;
    Runtime rt_cpu_serial(cpu_cfg);

    RuntimeConfig adaptive_cfg;
    adaptive_cfg.scheduler.gpu_available = false;  // GPU off for CI
    Runtime rt_adaptive(adaptive_cfg);

    size_t n = 1'000'000;
    RunStats cpu      = run_workload(rt_cpu_serial, n);
    RunStats adaptive = run_workload(rt_adaptive,   n);

    double adaptive_p95 = adaptive.mean + 2.0 * adaptive.stddev;

    EXPECT_LT(adaptive_p95, cpu.mean * 1.5)
        << "Adaptive p95=" << adaptive_p95 << "ms  CPU mean=" << cpu.mean << "ms  "
        << "(adaptive mean=" << adaptive.mean << " σ=" << adaptive.stddev << ")";
}

// Parallel should be faster than serial for large inputs.
// Allow up to 10% overhead on single-core machines (was 20%).
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
    RunStats serial   = run_workload(rt_serial,   n);
    RunStats parallel = run_workload(rt_parallel, n);

    EXPECT_LT(parallel.mean, serial.mean * 1.1)
        << "Parallel mean=" << parallel.mean << "ms  Serial mean=" << serial.mean << "ms";
}

// 1000 tiny tasks must complete in under 200ms total (was 500ms).
TEST(PerfRegression, SmallTaskSchedulerOverheadLow) {
    RuntimeConfig cfg;
    cfg.scheduler.gpu_available = false;
    Runtime rt(cfg);

    size_t n = 64;
    std::vector<float> in(n, 1.0f), out(n);

    // Warm-up
    for (int i = 0; i < WARMUP_REPS; ++i) {
        WorkloadDescriptor wd{};
        wd.id            = "warm";
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = n;
        rt.submit(wd);
    }

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

    EXPECT_LT(ms, 200.0) << "1000 tiny tasks took " << ms << "ms — scheduler overhead too high";
}

// Variance sanity: stddev/mean (CV) for a stable workload must stay below 20%.
// High CV indicates measurement noise that masks real regressions.
TEST(PerfRegression, MeasurementVarianceSanity) {
    RuntimeConfig cfg;
    cfg.scheduler.gpu_available = false;
    Runtime rt(cfg);

    RunStats s = run_workload(rt, 500'000);
    double cv = (s.mean > 1e-9) ? s.stddev / s.mean : 0.0;

    EXPECT_LT(cv, 0.20)
        << "High measurement variance: mean=" << s.mean << "ms stddev=" << s.stddev
        << "ms CV=" << cv * 100.0 << "% — results unreliable";
}
