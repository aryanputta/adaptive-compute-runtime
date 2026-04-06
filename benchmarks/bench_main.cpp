// Benchmark driver: runs five workload profiles through three execution strategies
// (CPU-only, GPU-only, adaptive) and prints a comparison table.
//
// Workload profiles:
//   W1: tiny  — 1K elements,  element-wise op, latency-sensitive
//   W2: small — 64K elements, element-wise op
//   W3: large — 4M elements,  high-intensity (GEMM proxy)
//   W4: giant — 16M elements, memory-bound streaming
//   W5: mixed batch — 256 tasks of varying size

#include "runtime.h"
#include "feature_extractor.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>

using namespace acr;

static std::vector<float> make_input(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

struct BenchCase {
    std::string name;
    size_t      n;
    double      arith_intensity;
    bool        latency_sensitive;
};

static void run_case(Runtime& rt_adaptive,
                     Runtime& rt_cpu,
                     Runtime& rt_gpu,
                     const BenchCase& bc,
                     int reps = 5) {
    std::vector<float> in  = make_input(bc.n);
    std::vector<float> out(bc.n);

    auto make_wd = [&](const std::string& id) {
        WorkloadDescriptor wd{};
        wd.id            = id;
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = bc.n;
        wd.features.arithmetic_intensity = bc.arith_intensity;
        wd.features.latency_sensitive    = bc.latency_sensitive;
        return wd;
    };

    double sum_adaptive = 0, sum_cpu = 0, sum_gpu = 0;

    for (int i = 0; i < reps; ++i) {
        sum_adaptive += rt_adaptive.submit(make_wd(bc.name + "_adaptive")).wall_ms;
        sum_cpu      += rt_cpu.submit(make_wd(bc.name + "_cpu")).wall_ms;
        sum_gpu      += rt_gpu.submit(make_wd(bc.name + "_gpu")).wall_ms;
    }

    double avg_adaptive = sum_adaptive / reps;
    double avg_cpu      = sum_cpu      / reps;
    double avg_gpu      = sum_gpu      / reps;
    double best_static  = std::min(avg_cpu, avg_gpu);
    double speedup      = best_static / (avg_adaptive + 1e-9);

    std::printf("%-20s  %8.3f ms  %8.3f ms  %8.3f ms  %+.1f%%\n",
                bc.name.c_str(), avg_cpu, avg_gpu, avg_adaptive,
                (speedup - 1.0) * 100.0);
}

int main() {
    // Adaptive runtime
    RuntimeConfig adaptive_cfg{};
    Runtime rt_adaptive(adaptive_cfg);

    // CPU-only runtime (disable GPU)
    RuntimeConfig cpu_cfg{};
    cpu_cfg.scheduler.gpu_available = false;
    Runtime rt_cpu(cpu_cfg);

    // GPU-only runtime (very low threshold to always prefer GPU)
    RuntimeConfig gpu_cfg{};
    gpu_cfg.scheduler.gpu_launch_threshold = 0;
    gpu_cfg.scheduler.transfer_compute_ratio_threshold = 999.0;
    Runtime rt_gpu(gpu_cfg);

    std::printf("\n%-20s  %10s  %10s  %10s  %s\n",
                "Workload", "CPU(ms)", "GPU(ms)", "Adaptive(ms)", "vs Best-Static");
    std::printf("%s\n", std::string(72, '-').c_str());

    std::vector<BenchCase> cases = {
        {"tiny_latency",    1'024,          1.0,   true},
        {"small_elemwise",  65'536,         1.0,   false},
        {"large_matmul",    4'194'304,     64.0,   false},
        {"giant_streaming", 16'777'216,     0.5,   false},
        {"medium_sparse",   524'288,        2.5,   false},
    };

    for (const auto& c : cases) run_case(rt_adaptive, rt_cpu, rt_gpu, c);

    std::printf("\n");

    // Print detailed adaptive profiler output
    rt_adaptive.profiler().print_summary();
    rt_adaptive.profiler().dump_csv("benchmark_results.csv");
    std::printf("\nResults written to benchmark_results.csv\n");

    return 0;
}
