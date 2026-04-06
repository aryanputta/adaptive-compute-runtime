#include "scheduler.h"
#include <numeric>

namespace acr {

Scheduler::Scheduler(SchedulerConfig cfg) : cfg_(std::move(cfg)) {}

ExecutionPath Scheduler::select(const WorkloadDescriptor& wd) const {
    if (!cfg_.gpu_available) {
        return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
    }
    switch (wd.type) {
        case WorkloadType::COMPUTE_BOUND:      return handle_compute_bound(wd);
        case WorkloadType::MEMORY_BOUND:       return handle_memory_bound(wd);
        case WorkloadType::TRANSFER_BOUND:     return handle_transfer_bound(wd);
        case WorkloadType::LATENCY_SENSITIVE:  return handle_latency_sensitive(wd);
        default:
            // Unknown: fall back to CPU_PARALLEL for safety
            return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
    }
}

std::vector<ExecutionPath> Scheduler::select_batch(const std::vector<WorkloadDescriptor>& batch) const {
    std::vector<ExecutionPath> paths;
    paths.reserve(batch.size());

    // Count how many small tasks could be coalesced into GPU_BATCHED
    size_t total_elements = 0;
    for (const auto& wd : batch) total_elements += wd.element_count;

    for (const auto& wd : batch) {
        ExecutionPath p = select(wd);
        // If a GPU_DIRECT task is too small on its own, but the total batch is large,
        // upgrade to GPU_BATCHED so the caller fuses them.
        if (p == ExecutionPath::CPU_PARALLEL &&
            wd.element_count < cfg_.gpu_launch_threshold &&
            total_elements   >= cfg_.gpu_launch_threshold &&
            !wd.features.latency_sensitive) {
            p = ExecutionPath::GPU_BATCHED;
        }
        paths.push_back(p);
    }
    return paths;
}

ExecutionPath Scheduler::handle_compute_bound(const WorkloadDescriptor& wd) const {
    if (wd.element_count < cfg_.gpu_launch_threshold) {
        // Too small — GPU launch overhead likely dominates.
        return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
    }
    // Large compute-bound → GPU is almost always best.
    return ExecutionPath::GPU_DIRECT;
}

ExecutionPath Scheduler::handle_memory_bound(const WorkloadDescriptor& wd) const {
    // Memory-bound on GPU = streaming memory kernel.  Worth it only for large inputs
    // where GPU HBM bandwidth beats CPU DDR4 bandwidth.
    if (wd.element_count < cfg_.gpu_launch_threshold * 4) {
        // Small memory-bound work: CPU DDR4 is often competitive with PCIe + HBM.
        return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
    }
    // Check if transfer dominates
    const auto& f = wd.features;
    double ratio = f.estimated_transfer_ms / (f.estimated_compute_ms + 1e-9);
    if (ratio > cfg_.transfer_compute_ratio_threshold) {
        // Transfer dominates even for large work — defer and batch.
        return ExecutionPath::GPU_BATCHED;
    }
    return ExecutionPath::GPU_DIRECT;
}

ExecutionPath Scheduler::handle_transfer_bound(const WorkloadDescriptor& wd) const {
    if (wd.features.latency_sensitive) {
        // Cannot afford batching; must execute now even if suboptimal.
        return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
    }
    if (wd.element_count >= cfg_.gpu_launch_threshold) {
        // Large task: batch with deferred strategy to amortize transfer cost.
        return ExecutionPath::DEFERRED;
    }
    // Small transfer-bound → CPU wins.
    return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
}

ExecutionPath Scheduler::handle_latency_sensitive(const WorkloadDescriptor& wd) const {
    // Never defer. Use GPU only if large enough to benefit after transfer.
    if (wd.element_count >= cfg_.gpu_launch_threshold * 2) {
        return ExecutionPath::GPU_DIRECT;
    }
    return cfg_.openmp_available ? ExecutionPath::CPU_PARALLEL : ExecutionPath::CPU_SERIAL;
}

} // namespace acr
