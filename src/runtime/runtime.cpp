#include "runtime.h"
#include "feature_extractor.h"
#include <stdexcept>
#include <algorithm>

namespace acr {

Runtime::Runtime(RuntimeConfig cfg)
    : cfg_(std::move(cfg)),
      scheduler_(cfg_.scheduler),
      gpu_exec_(mm_) {
    FeatureExtractor::detect_hardware();
}

ExecutionResult Runtime::submit(WorkloadDescriptor wd) {
    FeatureExtractor::extract(wd);
    ExecutionPath path = scheduler_.select(wd);
    ExecutionResult r  = dispatch(wd, path);
    if (cfg_.enable_profiling) profiler_.record(r);
    return r;
}

std::vector<ExecutionResult> Runtime::submit_batch(std::vector<WorkloadDescriptor> batch) {
    // Extract features for all workloads first.
    for (auto& wd : batch) FeatureExtractor::extract(wd);

    auto paths = scheduler_.select_batch(batch);

    // Separate GPU_BATCHED tasks for coalesced execution.
    std::vector<size_t> gpu_batch_indices;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (paths[i] == ExecutionPath::GPU_BATCHED) {
            gpu_batch_indices.push_back(i);
        }
    }

    std::vector<ExecutionResult> results(batch.size());

    // Execute non-batched GPU tasks individually.
    for (size_t i = 0; i < batch.size(); ++i) {
        if (paths[i] != ExecutionPath::GPU_BATCHED) {
            results[i] = dispatch(batch[i], paths[i]);
        }
    }

    // Execute coalesced GPU batch.
    if (!gpu_batch_indices.empty() && gpu_exec_.is_available()) {
        std::vector<WorkloadDescriptor> sub_batch;
        sub_batch.reserve(gpu_batch_indices.size());
        for (size_t idx : gpu_batch_indices) sub_batch.push_back(batch[idx]);

        auto batch_results = gpu_exec_.run_batch(sub_batch);
        for (size_t j = 0; j < gpu_batch_indices.size(); ++j) {
            results[gpu_batch_indices[j]] = batch_results[j];
        }
    } else {
        // Fallback: run each GPU_BATCHED task on CPU.
        for (size_t idx : gpu_batch_indices) {
            results[idx] = dispatch(batch[idx], ExecutionPath::CPU_PARALLEL);
        }
    }

    if (cfg_.enable_profiling) {
        for (const auto& r : results) profiler_.record(r);
    }
    return results;
}

ExecutionResult Runtime::dispatch(WorkloadDescriptor& wd, ExecutionPath path) {
    switch (path) {
        case ExecutionPath::CPU_SERIAL:
            return cpu_exec_.run_serial(wd);

        case ExecutionPath::CPU_PARALLEL:
            return cpu_exec_.run_parallel(wd);

        case ExecutionPath::GPU_DIRECT:
            if (gpu_exec_.is_available()) return gpu_exec_.run(wd);
            // Fallback
            return cpu_exec_.run_parallel(wd);

        case ExecutionPath::GPU_BATCHED:
            // Single-item batched path — wrap and run.
            if (gpu_exec_.is_available()) {
                std::vector<WorkloadDescriptor> single{wd};
                auto r = gpu_exec_.run_batch(single);
                return r[0];
            }
            return cpu_exec_.run_parallel(wd);

        case ExecutionPath::DEFERRED:
            // For now, deferred falls through to CPU parallel (batching handled at submit_batch level).
            return cpu_exec_.run_parallel(wd);

        default:
            throw std::runtime_error("Unknown execution path");
    }
}

} // namespace acr
