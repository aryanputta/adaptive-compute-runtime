#include <gtest/gtest.h>
#include "scheduler.h"
#include "workload.h"

using namespace acr;

static WorkloadDescriptor make_wd(size_t n, WorkloadType type,
                                  double arith_intensity = 1.0,
                                  bool latency_sensitive = false) {
    WorkloadDescriptor wd{};
    wd.element_count = n;
    wd.type          = type;
    wd.features.arithmetic_intensity  = arith_intensity;
    wd.features.latency_sensitive     = latency_sensitive;
    wd.features.estimated_transfer_ms = static_cast<double>(n * sizeof(float)) / (20e9 / 1e3);
    wd.features.estimated_compute_ms  = 0.1;
    return wd;
}

TEST(Scheduler, TinyWorkloadStaysOnCPU) {
    Scheduler s;
    auto wd = make_wd(512, WorkloadType::COMPUTE_BOUND);
    auto path = s.select(wd);
    EXPECT_NE(path, ExecutionPath::GPU_DIRECT);
    EXPECT_NE(path, ExecutionPath::GPU_BATCHED);
}

TEST(Scheduler, LargeComputeBoundGoesToGPU) {
    Scheduler s;
    auto wd = make_wd(1'000'000, WorkloadType::COMPUTE_BOUND, 64.0);
    auto path = s.select(wd);
    EXPECT_EQ(path, ExecutionPath::GPU_DIRECT);
}

TEST(Scheduler, LatencySensitiveNeverDeferred) {
    Scheduler s;
    auto wd = make_wd(500'000, WorkloadType::LATENCY_SENSITIVE, 1.0, true);
    auto path = s.select(wd);
    EXPECT_NE(path, ExecutionPath::DEFERRED);
}

TEST(Scheduler, GPUUnavailableFallsBackToCPU) {
    SchedulerConfig cfg;
    cfg.gpu_available = false;
    Scheduler s(cfg);
    auto wd = make_wd(10'000'000, WorkloadType::COMPUTE_BOUND, 64.0);
    auto path = s.select(wd);
    EXPECT_TRUE(path == ExecutionPath::CPU_SERIAL ||
                path == ExecutionPath::CPU_PARALLEL);
}

TEST(Scheduler, TransferBoundSmallGoesToCPU) {
    Scheduler s;
    // Small transfer-bound workload — transfer cost > compute.
    WorkloadDescriptor wd = make_wd(1024, WorkloadType::TRANSFER_BOUND, 0.5);
    wd.features.estimated_transfer_ms = 5.0;
    wd.features.estimated_compute_ms  = 0.1;
    auto path = s.select(wd);
    EXPECT_TRUE(path == ExecutionPath::CPU_SERIAL ||
                path == ExecutionPath::CPU_PARALLEL);
}

TEST(Scheduler, BatchCoalescesSmallTasks) {
    SchedulerConfig cfg;
    cfg.gpu_launch_threshold = 8192;
    Scheduler s(cfg);

    std::vector<WorkloadDescriptor> batch;
    for (int i = 0; i < 32; ++i) {
        auto wd = make_wd(1024, WorkloadType::COMPUTE_BOUND, 8.0);
        wd.id = "task_" + std::to_string(i);
        batch.push_back(wd);
    }
    auto paths = s.select_batch(batch);
    ASSERT_EQ(paths.size(), batch.size());

    // Expect some tasks to be promoted to GPU_BATCHED since total > threshold
    bool any_batched = false;
    for (auto p : paths) {
        if (p == ExecutionPath::GPU_BATCHED) { any_batched = true; break; }
    }
    EXPECT_TRUE(any_batched);
}

TEST(Scheduler, MemoryBoundSmallPrefersCPU) {
    Scheduler s;
    auto wd = make_wd(2048, WorkloadType::MEMORY_BOUND, 0.8);
    auto path = s.select(wd);
    EXPECT_TRUE(path == ExecutionPath::CPU_SERIAL ||
                path == ExecutionPath::CPU_PARALLEL);
}
