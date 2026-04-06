#pragma once
#include "workload.h"

namespace acr {

// Estimates runtime features from a raw workload descriptor.
// Called before the scheduler makes a placement decision.
//
// Hardware constants are initialised once via detect_hardware() which probes
// the actual system rather than relying on compile-time guesses.  Call
// detect_hardware() at program start (or from Runtime's constructor) before
// submitting any workloads.
class FeatureExtractor {
public:
    // Fills wd.features and wd.type in-place.
    static void extract(WorkloadDescriptor& wd);

    // Probe the host system and populate the hardware constants below.
    // Safe to call multiple times; subsequent calls are no-ops.
    static void detect_hardware();

    // Measured (or fallback) hardware parameters — readable for diagnostics.
    static double pcie_bandwidth_GBs;   // PCIe host↔device bandwidth
    static double gpu_tflops;           // GPU fp32 peak TFLOPS
    static double cpu_gflops;           // CPU fp32 peak GFLOPS
    static double cpu_mem_bandwidth_GBs; // CPU DRAM sustained bandwidth

private:
    static bool  hardware_detected_;

    static double estimate_transfer_ms(size_t bytes);
    static double estimate_gpu_compute_ms(size_t elements, double arith_intensity);
    static double estimate_cpu_compute_ms(size_t elements, double arith_intensity);
    static WorkloadType classify(const WorkloadFeatures& f);
};

} // namespace acr
