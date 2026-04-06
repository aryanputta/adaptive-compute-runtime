#include "feature_extractor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace acr {

// Rough FLOP model: assume each element touches 2 operands and does ~8 FLOPs
// (representative of a fused multiply-add heavy kernel like GEMM or convolution).
static constexpr double kFLOPsPerElement = 8.0;

// Static member definitions — conservative fallbacks used before detect_hardware().
double FeatureExtractor::pcie_bandwidth_GBs    = 20.0;   // PCIe Gen4 x16 conservative
double FeatureExtractor::gpu_tflops            = 10.0;   // modest fallback
double FeatureExtractor::cpu_gflops            = 100.0;  // modest fallback
double FeatureExtractor::cpu_mem_bandwidth_GBs = 40.0;   // DDR4-3200 dual-channel ~50 GB/s
bool   FeatureExtractor::hardware_detected_    = false;

// ---------------------------------------------------------------------------
// Hardware detection helpers
// ---------------------------------------------------------------------------

// Measure CPU memory bandwidth by streaming a large buffer (read + write).
// Returns GB/s sustained.
static double measure_cpu_bandwidth() {
    const size_t kBytes = 256 * 1024 * 1024; // 256 MiB — fits in DRAM, not L3
    const size_t kN     = kBytes / sizeof(float);
    const int    kReps  = 4;

    std::vector<float> src(kN, 1.0f);
    std::vector<float> dst(kN, 0.0f);

    double best_GBs = 0.0;
    for (int r = 0; r < kReps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Stream copy — compiler should vectorise this loop.
        for (size_t i = 0; i < kN; ++i) dst[i] = src[i];
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        double GBs = (2.0 * kBytes / 1e9) / sec; // read + write
        if (GBs > best_GBs) best_GBs = GBs;
    }
    return best_GBs;
}

// Estimate CPU peak GFLOPS from hardware_concurrency and a rough per-core rate.
// On systems with AVX2: ~16 FLOPs/cycle/core * freq (GHz) per core.
static double estimate_cpu_gflops() {
    unsigned cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 4;
    // Conservatively assume 2 GHz with 8 FLOPs/cycle (SSE/scalar; halved for safety)
    return static_cast<double>(cores) * 2.0 * 8.0; // GFLOPS
}

void FeatureExtractor::detect_hardware() {
    if (hardware_detected_) return;
    hardware_detected_ = true;

    // --- CPU ---
    cpu_mem_bandwidth_GBs = measure_cpu_bandwidth();
    cpu_gflops            = estimate_cpu_gflops();

#ifdef HAVE_CUDA
    // --- GPU via CUDA runtime ---
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            // Peak fp32 TFLOPS = SM_count * 128 CUDA cores/SM * 2 FLOPs * clock_GHz
            double clock_ghz  = prop.clockRate * 1e-6; // kHz → GHz
            double cores_total = prop.multiProcessorCount * 128.0; // rough — varies by arch
            gpu_tflops = (cores_total * 2.0 * clock_ghz) / 1e3; // TFLOPS

            // PCIe bandwidth: busWidth bits at pciBusID (use theoretical peak)
            // cudaDeviceProp does not expose PCIe speed directly; use memory bus as proxy
            // for integrated/discrete and default to gen4 estimate for discrete.
            if (prop.integrated) {
                pcie_bandwidth_GBs = cpu_mem_bandwidth_GBs; // shared memory
            } else {
                pcie_bandwidth_GBs = 20.0; // PCIe Gen4 x16 conservative
            }
        }
    }
#endif

    std::printf("[FeatureExtractor] Hardware detected:\n"
                "  CPU  %.1f GFLOPS  Mem BW %.1f GB/s\n"
                "  GPU  %.1f TFLOPS  PCIe BW %.1f GB/s\n",
                cpu_gflops, cpu_mem_bandwidth_GBs,
                gpu_tflops, pcie_bandwidth_GBs);
}

// ---------------------------------------------------------------------------

double FeatureExtractor::estimate_transfer_ms(size_t bytes) {
    double gb = static_cast<double>(bytes) / 1e9;
    return (gb / pcie_bandwidth_GBs) * 1e3;
}

double FeatureExtractor::estimate_gpu_compute_ms(size_t elements, double arith_intensity) {
    double total_flops = static_cast<double>(elements) * kFLOPsPerElement;
    double peak_flops  = gpu_tflops * 1e12;
    // Roofline: memory ceiling uses measured CPU BW as a stand-in when GPU BW unknown.
    double mem_bw_Bps  = cpu_mem_bandwidth_GBs * 1e9 * 20.0; // GPU HBM ~20x CPU DDR
    double mem_ceiling = arith_intensity * mem_bw_Bps;
    double effective   = std::min(peak_flops, mem_ceiling);
    return (total_flops / effective) * 1e3;
}

double FeatureExtractor::estimate_cpu_compute_ms(size_t elements, double arith_intensity) {
    double total_flops = static_cast<double>(elements) * kFLOPsPerElement;
    double peak_flops  = cpu_gflops * 1e9;
    (void)arith_intensity;
    return (total_flops / peak_flops) * 1e3;
}

WorkloadType FeatureExtractor::classify(const WorkloadFeatures& f) {
    if (f.latency_sensitive) {
        return WorkloadType::LATENCY_SENSITIVE;
    }
    // High arithmetic intensity → compute-bound by nature (e.g. GEMM, convolution).
    // The scheduler decides separately whether PCIe transfer cost makes GPU worthwhile.
    if (f.arithmetic_intensity >= 16.0) {
        return WorkloadType::COMPUTE_BOUND;
    }
    // Low arithmetic intensity → memory-bound streaming kernel.
    if (f.arithmetic_intensity < 2.0) {
        return WorkloadType::MEMORY_BOUND;
    }
    // Medium intensity: if transfer dominates estimated GPU compute, flag as transfer-bound
    // so the scheduler can consider batching or CPU execution.
    if (f.estimated_transfer_ms > 0.5 * (f.estimated_compute_ms + 1e-6)) {
        return WorkloadType::TRANSFER_BOUND;
    }
    return WorkloadType::COMPUTE_BOUND;
}

void FeatureExtractor::extract(WorkloadDescriptor& wd) {
    WorkloadFeatures& f = wd.features;

    // Sizes
    f.input_bytes  = wd.element_count * sizeof(float);
    f.output_bytes = wd.element_count * sizeof(float);

    // Arithmetic intensity: caller may pre-fill; otherwise estimate from element count.
    // For a pure element-wise op: 1 read + 1 write, ~2 FLOPs → intensity ~1.
    // For a GEMM of N×N: ~2N^3 FLOPs / N^2 * 4 bytes → intensity grows with N.
    if (f.arithmetic_intensity <= 0.0) {
        // Heuristic: sqrt of element count as a proxy for matrix dimension ratio
        double n = std::sqrt(static_cast<double>(wd.element_count));
        f.arithmetic_intensity = std::clamp(n / 64.0, 0.5, 512.0);
    }

    size_t total_transfer = f.input_bytes + f.output_bytes;
    f.estimated_transfer_ms  = estimate_transfer_ms(total_transfer);
    f.estimated_compute_ms   = estimate_gpu_compute_ms(wd.element_count, f.arithmetic_intensity);

    wd.type = classify(f);
}

} // namespace acr
