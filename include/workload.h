#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace acr {

enum class WorkloadType {
    COMPUTE_BOUND,
    MEMORY_BOUND,
    TRANSFER_BOUND,
    LATENCY_SENSITIVE,
    UNKNOWN
};

enum class ExecutionPath {
    CPU_SERIAL,
    CPU_PARALLEL,
    GPU_DIRECT,
    GPU_BATCHED,
    HYBRID,
    DEFERRED
};

struct WorkloadFeatures {
    size_t input_bytes;
    size_t output_bytes;
    size_t batch_size;
    double arithmetic_intensity;   // FLOP / byte
    double estimated_transfer_ms;
    double estimated_compute_ms;
    bool latency_sensitive;
    bool sparse;
};

struct WorkloadDescriptor {
    std::string id;
    WorkloadFeatures features;
    WorkloadType type;
    ExecutionPath preferred_path;

    // Raw data pointers (host side)
    const float* input_data;
    float*       output_data;
    size_t       element_count;
};

struct ExecutionResult {
    std::string workload_id;
    ExecutionPath path_used;
    double wall_ms;
    double kernel_ms;
    double h2d_ms;
    double d2h_ms;
    double memory_throughput_GBs;
    bool   correct;
};

} // namespace acr
