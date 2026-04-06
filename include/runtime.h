#pragma once
#include "workload.h"
#include "scheduler.h"
#include "memory_manager.h"
#include "profiler.h"
#include "cpu_executor.h"
#include "gpu_executor.h"
#include <vector>
#include <memory>

namespace acr {

struct RuntimeConfig {
    SchedulerConfig scheduler;
    bool            enable_profiling = true;
    bool            verify_results   = true;  // cross-check CPU vs GPU for correctness
};

// Top-level entry point. Owns all subsystems.
class Runtime {
public:
    explicit Runtime(RuntimeConfig cfg = {});

    // Submit a single workload. Returns immediately with result.
    ExecutionResult submit(WorkloadDescriptor wd);

    // Submit many workloads. The scheduler may reorder or batch them.
    std::vector<ExecutionResult> submit_batch(std::vector<WorkloadDescriptor> batch);

    Profiler&       profiler()        { return profiler_; }
    MemoryManager&  memory_manager()  { return mm_; }
    const Scheduler& scheduler() const { return scheduler_; }

private:
    RuntimeConfig cfg_;
    MemoryManager mm_;
    Scheduler     scheduler_;
    CPUExecutor   cpu_exec_;
    GPUExecutor   gpu_exec_;
    Profiler      profiler_;

    ExecutionResult dispatch(WorkloadDescriptor& wd, ExecutionPath path);
};

} // namespace acr
