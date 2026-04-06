// Benchmark driver: runs workload profiles through three execution strategies
// (CPU-only, GPU-only, adaptive) and prints a statistical comparison table.
//
// Workload profiles:
//   W1: sub_alloc       — 256 elements,   allocation overhead dominates
//   W2: tiny_latency    — 1K elements,    latency-sensitive
//   W3: small_elemwise  — 64K elements,   element-wise op
//   W4: medium_sparse   — 512K elements,  moderate intensity
//   W5: large_matmul    — 4M elements,    high arithmetic intensity (GEMM proxy)
//   W6: giant_streaming — 16M elements,   memory-bound streaming
//   W7: bw_sweep_512k   — 512K floats,    read-only bandwidth stress
//   W8: bw_sweep_4m     — 4M floats,      read-only bandwidth stress (larger)
//   W9: reduction       — 2M elements,    reduction (irregular memory access)
//
// Each case runs WARMUP_REPS warm-up iterations (discarded) followed by
// MEASURE_REPS measurement iterations.  Results report mean, stddev, p95,
// and coefficient of variation (CV = stddev/mean).  High-CV runs (>15%) are
// flagged as unreliable.

#include "runtime.h"
#include "feature_extractor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace acr;

static constexpr int WARMUP_REPS  = 3;
static constexpr int MEASURE_REPS = 15;

// ---- Statistical helpers ------------------------------------------------

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0;
    double min    = 0.0;
    double max    = 0.0;
    double p50    = 0.0;
    double p95    = 0.0;

    double cv() const { return (mean > 1e-9) ? stddev / mean : 0.0; }
};

static Stats compute_stats(std::vector<double> v) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());

    Stats s;
    s.min  = v.front();
    s.max  = v.back();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();

    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq_sum / v.size());

    auto percentile = [&](double pct) {
        double idx = pct * (v.size() - 1);
        size_t lo  = static_cast<size_t>(idx);
        size_t hi  = std::min(lo + 1, v.size() - 1);
        double frac = idx - lo;
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };
    s.p50 = percentile(0.50);
    s.p95 = percentile(0.95);
    return s;
}

// ---- Input generation ---------------------------------------------------

static std::vector<float> make_input(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ---- Workload descriptor ------------------------------------------------

struct BenchCase {
    std::string name;
    size_t      n;
    double      arith_intensity;
    bool        latency_sensitive;
};

// ---- Core measurement ---------------------------------------------------

struct CaseResult {
    Stats cpu;
    Stats gpu;
    Stats adaptive;
};

static CaseResult run_case(Runtime& rt_adaptive,
                           Runtime& rt_cpu,
                           Runtime& rt_gpu,
                           const BenchCase& bc) {
    std::vector<float> in  = make_input(bc.n);
    std::vector<float> out(bc.n);

    auto make_wd = [&](const std::string& suffix) {
        WorkloadDescriptor wd{};
        wd.id            = bc.name + "_" + suffix;
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = bc.n;
        wd.features.arithmetic_intensity = bc.arith_intensity;
        wd.features.latency_sensitive    = bc.latency_sensitive;
        return wd;
    };

    // Warm-up (results discarded)
    for (int i = 0; i < WARMUP_REPS; ++i) {
        rt_adaptive.submit(make_wd("warm"));
        rt_cpu.submit(make_wd("warm"));
        rt_gpu.submit(make_wd("warm"));
    }

    // Measured runs
    std::vector<double> adaptive_ms, cpu_ms, gpu_ms;
    adaptive_ms.reserve(MEASURE_REPS);
    cpu_ms.reserve(MEASURE_REPS);
    gpu_ms.reserve(MEASURE_REPS);

    for (int i = 0; i < MEASURE_REPS; ++i) {
        adaptive_ms.push_back(rt_adaptive.submit(make_wd("adaptive")).wall_ms);
        cpu_ms.push_back(rt_cpu.submit(make_wd("cpu")).wall_ms);
        gpu_ms.push_back(rt_gpu.submit(make_wd("gpu")).wall_ms);
    }

    return {compute_stats(cpu_ms), compute_stats(gpu_ms), compute_stats(adaptive_ms)};
}

// ---- Reporting ----------------------------------------------------------

static void print_header() {
    std::printf("\n%-20s  %10s %7s %6s  %10s %7s %6s  %10s %7s %6s  %8s  %s\n",
                "Workload",
                "CPU(ms)", "±σ", "CV%",
                "GPU(ms)", "±σ", "CV%",
                "Adap(ms)", "±σ", "CV%",
                "Speedup", "Reliable?");
    std::printf("%s\n", std::string(115, '-').c_str());
}

static void print_row(const BenchCase& bc, const CaseResult& r) {
    double best_static = std::min(r.cpu.mean, r.gpu.mean);
    double speedup     = best_static / (r.adaptive.mean + 1e-9);

    // Flag as unreliable if any strategy has CV > 15%
    bool reliable = (r.cpu.cv() < 0.15 && r.gpu.cv() < 0.15 && r.adaptive.cv() < 0.15);

    std::printf("%-20s  %10.3f %7.3f %5.1f%%  %10.3f %7.3f %5.1f%%  %10.3f %7.3f %5.1f%%  %+7.1f%%  %s\n",
                bc.name.c_str(),
                r.cpu.mean,      r.cpu.stddev,      r.cpu.cv()      * 100.0,
                r.gpu.mean,      r.gpu.stddev,      r.gpu.cv()      * 100.0,
                r.adaptive.mean, r.adaptive.stddev, r.adaptive.cv() * 100.0,
                (speedup - 1.0) * 100.0,
                reliable ? "YES" : "HIGH-VAR");
}

static void print_percentile_table(const BenchCase& bc, const CaseResult& r) {
    std::printf("  %-20s  p50: CPU=%.3f GPU=%.3f Adap=%.3f  "
                "p95: CPU=%.3f GPU=%.3f Adap=%.3f\n",
                bc.name.c_str(),
                r.cpu.p50, r.gpu.p50, r.adaptive.p50,
                r.cpu.p95, r.gpu.p95, r.adaptive.p95);
}

// ---- Entry point --------------------------------------------------------

int main() {
    // Adaptive runtime
    RuntimeConfig adaptive_cfg{};
    Runtime rt_adaptive(adaptive_cfg);

    // CPU-only runtime
    RuntimeConfig cpu_cfg{};
    cpu_cfg.scheduler.gpu_available = false;
    Runtime rt_cpu(cpu_cfg);

    // GPU-forced runtime (lowest threshold so it always routes to GPU)
    RuntimeConfig gpu_cfg{};
    gpu_cfg.scheduler.gpu_launch_threshold              = 0;
    gpu_cfg.scheduler.transfer_compute_ratio_threshold  = 999.0;
    Runtime rt_gpu(gpu_cfg);

    std::printf("Warmup reps: %d   Measurement reps: %d\n", WARMUP_REPS, MEASURE_REPS);

    std::vector<BenchCase> cases = {
        // Edge case: sub-1KB, allocation overhead should dominate
        {"sub_alloc",       256,          1.0,   true},
        // Latency-sensitive tiny workload
        {"tiny_latency",    1'024,        1.0,   true},
        // Small element-wise: CPU often wins due to transfer cost
        {"small_elemwise",  65'536,       1.0,   false},
        // Medium intensity: scheduler boundary region
        {"medium_sparse",   524'288,      2.5,   false},
        // High-intensity compute proxy (GEMM-like): GPU should win at scale
        {"large_matmul",    4'194'304,   64.0,   false},
        // Memory-bound streaming: bandwidth limited
        {"giant_streaming", 16'777'216,   0.5,   false},
        // Bandwidth stress — read-dominated, moderate size
        {"bw_sweep_512k",   524'288,      0.25,  false},
        // Bandwidth stress — larger
        {"bw_sweep_4m",     4'194'304,    0.25,  false},
        // Reduction proxy — low intensity, irregular per-element cost
        {"reduction_2m",    2'097'152,    0.75,  false},
    };

    print_header();

    std::vector<CaseResult> results(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        results[i] = run_case(rt_adaptive, rt_cpu, rt_gpu, cases[i]);
        print_row(cases[i], results[i]);
    }

    std::printf("\n--- Percentile Breakdown (p50 / p95) ---\n");
    for (size_t i = 0; i < cases.size(); ++i)
        print_percentile_table(cases[i], results[i]);

    std::printf("\n");

    // Adaptive profiler summary (decisions + per-path breakdown)
    rt_adaptive.profiler().print_summary();
    rt_adaptive.profiler().dump_csv("benchmark_results.csv");
    std::printf("\nResults written to benchmark_results.csv\n");

    return 0;
}
