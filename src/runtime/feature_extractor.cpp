#include "feature_extractor.h"
#include <cmath>
#include <algorithm>

namespace acr {

// Rough FLOP model: assume each element touches 2 operands and does ~8 FLOPs
// (representative of a fused multiply-add heavy kernel like GEMM or convolution).
// Callers that know their exact FLOP count can override arithmetic_intensity directly.
static constexpr double kFLOPsPerElement = 8.0;

double FeatureExtractor::estimate_transfer_ms(size_t bytes) {
    double gb = static_cast<double>(bytes) / 1e9;
    return (gb / kPCIeBandwidthGBs) * 1e3;
}

double FeatureExtractor::estimate_gpu_compute_ms(size_t elements, double arith_intensity) {
    double total_flops = static_cast<double>(elements) * kFLOPsPerElement;
    double tflops      = kGPUTFLOPS * 1e12;
    // Roofline: bound by min(peak compute, memory bandwidth * intensity)
    double mem_ceiling = arith_intensity * (900e9);  // ~900 GB/s HBM estimate
    double effective   = std::min(tflops, mem_ceiling);
    return (total_flops / effective) * 1e3;
}

double FeatureExtractor::estimate_cpu_compute_ms(size_t elements, double arith_intensity) {
    double total_flops = static_cast<double>(elements) * kFLOPsPerElement;
    double gflops      = kCPUGFLOPS * 1e9;
    (void)arith_intensity;
    return (total_flops / gflops) * 1e3;
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
