#pragma once
#include "workload.h"

namespace acr {

// Estimates runtime features from a raw workload descriptor.
// Called before the scheduler makes a placement decision.
class FeatureExtractor {
public:
    // Fills wd.features and wd.type in-place.
    static void extract(WorkloadDescriptor& wd);

private:
    // Bandwidth model: PCIe Gen4 x16 peak ~32 GB/s; use 20 GB/s as conservative estimate.
    static constexpr double kPCIeBandwidthGBs = 20.0;

    // GPU peak TFLOPS (fp32) rough estimate for common student hardware (RTX 3080 ~29 TFLOPS).
    static constexpr double kGPUTFLOPS = 29.0;

    // CPU peak GFLOPS (fp32, 8-core AVX2 estimate).
    static constexpr double kCPUGFLOPS = 400.0;

    static double estimate_transfer_ms(size_t bytes);
    static double estimate_gpu_compute_ms(size_t elements, double arith_intensity);
    static double estimate_cpu_compute_ms(size_t elements, double arith_intensity);
    static WorkloadType classify(const WorkloadFeatures& f);
};

} // namespace acr
