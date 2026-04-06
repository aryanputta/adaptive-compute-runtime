// End-to-end integration tests for the Runtime class.
// These tests verify that workloads complete correctly regardless of which
// execution path the scheduler selects.

#include <gtest/gtest.h>
#include "runtime.h"
#include <vector>
#include <cmath>
#include <numeric>

using namespace acr;

static std::vector<float> make_input(size_t n, float fill = 128.0f) {
    return std::vector<float>(n, fill);
}

// Verify CPU output: y = max(0, (1/255)*x - 0.5)
static bool verify_output(const float* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float expected = in[i] / 255.0f - 0.5f;
        expected = expected > 0.0f ? expected : 0.0f;
        if (std::fabs(out[i] - expected) > 1e-4f) return false;
    }
    return true;
}

class RuntimeE2E : public ::testing::Test {
protected:
    RuntimeConfig cfg{};
    // Force CPU-only so tests pass without a GPU in CI.
    void SetUp() override {
        cfg.scheduler.gpu_available = false;
        cfg.enable_profiling = true;
    }
};

TEST_F(RuntimeE2E, SingleSmallWorkload) {
    Runtime rt(cfg);
    size_t n = 1024;
    auto in  = make_input(n);
    std::vector<float> out(n);

    WorkloadDescriptor wd{};
    wd.id            = "e2e_small";
    wd.input_data    = in.data();
    wd.output_data   = out.data();
    wd.element_count = n;

    auto r = rt.submit(wd);
    EXPECT_TRUE(r.correct);
    EXPECT_GT(r.wall_ms, 0.0);
    EXPECT_TRUE(verify_output(out.data(), in.data(), n));
}

TEST_F(RuntimeE2E, LargeWorkloadCompletes) {
    Runtime rt(cfg);
    size_t n = 4'000'000;
    auto in  = make_input(n, 200.0f);
    std::vector<float> out(n);

    WorkloadDescriptor wd{};
    wd.id            = "e2e_large";
    wd.input_data    = in.data();
    wd.output_data   = out.data();
    wd.element_count = n;

    auto r = rt.submit(wd);
    EXPECT_TRUE(r.correct);
    EXPECT_GT(r.wall_ms, 0.0);
}

TEST_F(RuntimeE2E, BatchSubmissionCorrectness) {
    Runtime rt(cfg);
    size_t n = 512;
    std::vector<std::vector<float>> inputs(8, make_input(n));
    std::vector<std::vector<float>> outputs(8, std::vector<float>(n));

    std::vector<WorkloadDescriptor> batch;
    for (int i = 0; i < 8; ++i) {
        WorkloadDescriptor wd{};
        wd.id            = "batch_" + std::to_string(i);
        wd.input_data    = inputs[i].data();
        wd.output_data   = outputs[i].data();
        wd.element_count = n;
        batch.push_back(wd);
    }

    auto results = rt.submit_batch(batch);
    ASSERT_EQ(results.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(results[i].correct) << "Task " << i << " failed";
        EXPECT_TRUE(verify_output(outputs[i].data(), inputs[i].data(), n));
    }
}

TEST_F(RuntimeE2E, ProfilerRecordsAllResults) {
    Runtime rt(cfg);
    size_t n = 256;
    auto in  = make_input(n);
    std::vector<float> out(n);

    for (int i = 0; i < 5; ++i) {
        WorkloadDescriptor wd{};
        wd.id            = "profile_" + std::to_string(i);
        wd.input_data    = in.data();
        wd.output_data   = out.data();
        wd.element_count = n;
        rt.submit(wd);
    }
    EXPECT_EQ(rt.profiler().results().size(), 5u);
    EXPECT_GT(rt.profiler().avg_wall_ms(), 0.0);
}

TEST_F(RuntimeE2E, ZeroElementWorkloadHandled) {
    Runtime rt(cfg);
    WorkloadDescriptor wd{};
    wd.id            = "zero";
    wd.input_data    = nullptr;
    wd.output_data   = nullptr;
    wd.element_count = 0;
    // Should not crash
    EXPECT_NO_THROW(rt.submit(wd));
}
