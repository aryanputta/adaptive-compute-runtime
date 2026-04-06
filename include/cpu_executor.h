#pragma once
#include "workload.h"

namespace acr {

class CPUExecutor {
public:
    // Serial path — single-threaded reference implementation.
    ExecutionResult run_serial(WorkloadDescriptor& wd);

    // Parallel path — OpenMP threaded.
    ExecutionResult run_parallel(WorkloadDescriptor& wd);
};

} // namespace acr
