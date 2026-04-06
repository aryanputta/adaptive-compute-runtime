#include <gtest/gtest.h>
#include "feature_extractor.h"
#include "workload.h"
#include <vector>

using namespace acr;

static WorkloadDescriptor make_wd(size_t n, bool latency = false, double intensity = 0.0) {
    static std::vector<float> dummy_in, dummy_out;
    dummy_in.assign(n, 1.0f);
    dummy_out.assign(n, 0.0f);

    WorkloadDescriptor wd{};
    wd.id = "test";
    wd.element_count = n;
    wd.input_data  = dummy_in.data();
    wd.output_data = dummy_out.data();
    wd.features.latency_sensitive   = latency;
    wd.features.arithmetic_intensity = intensity;
    return wd;
}

TEST(FeatureExtractor, FillsInputOutputBytes) {
    auto wd = make_wd(1024);
    FeatureExtractor::extract(wd);
    EXPECT_EQ(wd.features.input_bytes,  1024 * sizeof(float));
    EXPECT_EQ(wd.features.output_bytes, 1024 * sizeof(float));
}

TEST(FeatureExtractor, LatencySensitiveClassification) {
    auto wd = make_wd(10'000'000, /*latency=*/true);
    FeatureExtractor::extract(wd);
    EXPECT_EQ(wd.type, WorkloadType::LATENCY_SENSITIVE);
}

TEST(FeatureExtractor, SmallWorkloadIsMemoryOrTransferBound) {
    auto wd = make_wd(512);
    FeatureExtractor::extract(wd);
    // Small inputs: estimated transfer >> estimated compute → transfer-bound or memory-bound
    EXPECT_NE(wd.type, WorkloadType::COMPUTE_BOUND);
}

TEST(FeatureExtractor, LargeHighIntensityIsComputeBound) {
    auto wd = make_wd(4'000'000, false, 256.0);
    FeatureExtractor::extract(wd);
    EXPECT_EQ(wd.type, WorkloadType::COMPUTE_BOUND);
}

TEST(FeatureExtractor, TransferCostPositive) {
    auto wd = make_wd(1'000'000);
    FeatureExtractor::extract(wd);
    EXPECT_GT(wd.features.estimated_transfer_ms, 0.0);
}

TEST(FeatureExtractor, IntenistyInferredWhenZero) {
    auto wd = make_wd(64 * 64);  // intensity = 0 → extracted
    FeatureExtractor::extract(wd);
    EXPECT_GT(wd.features.arithmetic_intensity, 0.0);
}
