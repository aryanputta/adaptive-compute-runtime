#pragma once
#include "workload.h"
#include <vector>

namespace acr {

struct SchedulerConfig {
    // Minimum element count to justify GPU launch overhead.
    size_t gpu_launch_threshold = 8192;

    // If estimated transfer cost exceeds this fraction of estimated GPU compute,
    // prefer CPU or defer for batching.
    double transfer_compute_ratio_threshold = 0.5;

    // If arithmetic intensity is below this, workload is memory-bound; keep on CPU unless large.
    double memory_bound_intensity_threshold = 2.0;

    // Batch accumulation target before GPU offload (deferred path).
    size_t deferred_batch_target = 64;

    // Latency-sensitive tasks never go to deferred path.
    bool allow_deferred_for_latency_sensitive = false;

    bool gpu_available = true;
    bool openmp_available = true;
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig cfg = {});

    // Decide execution path for a single workload.
    ExecutionPath select(const WorkloadDescriptor& wd) const;

    // Decide execution paths for a batch (may coalesce small tasks).
    std::vector<ExecutionPath> select_batch(const std::vector<WorkloadDescriptor>& batch) const;

    const SchedulerConfig& config() const { return cfg_; }

private:
    SchedulerConfig cfg_;

    ExecutionPath handle_compute_bound(const WorkloadDescriptor& wd) const;
    ExecutionPath handle_memory_bound(const WorkloadDescriptor& wd) const;
    ExecutionPath handle_transfer_bound(const WorkloadDescriptor& wd) const;
    ExecutionPath handle_latency_sensitive(const WorkloadDescriptor& wd) const;
};

} // namespace acr
