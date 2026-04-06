#pragma once
#include "workload.h"
#include "memory_manager.h"

namespace acr {

class GPUExecutor {
public:
    explicit GPUExecutor(MemoryManager& mm);
    ~GPUExecutor();

    // Single workload — allocate, transfer, launch, transfer back, free.
    ExecutionResult run(WorkloadDescriptor& wd);

    // Fused batch — coalesce multiple small workloads into one H2D transfer.
    std::vector<ExecutionResult> run_batch(std::vector<WorkloadDescriptor>& batch);

    bool is_available() const { return gpu_available_; }

private:
    MemoryManager& mm_;
    bool gpu_available_ = false;
    [[maybe_unused]] int device_id_ = 0;

    void check_gpu();
};

} // namespace acr
