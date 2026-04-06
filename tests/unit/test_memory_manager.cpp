#include <gtest/gtest.h>
#include "memory_manager.h"
#include <cstring>

using namespace acr;

TEST(MemoryManager, AcquireAndReleaseRoundTrip) {
    MemoryManager mm;
    void* p = mm.acquire_pinned(1024);
    ASSERT_NE(p, nullptr);
    mm.release_pinned(p, 1024);
    // Stats should show one allocation
    EXPECT_GE(mm.stats().pinned_bytes_allocated, 1024u);
}

TEST(MemoryManager, CacheHitOnSecondAcquire) {
    MemoryManager mm;
    void* p1 = mm.acquire_pinned(4096);
    mm.release_pinned(p1, 4096);
    void* p2 = mm.acquire_pinned(4096);
    // Should reuse the same block — cache hit
    EXPECT_EQ(p1, p2);
    EXPECT_EQ(mm.stats().pinned_cache_hits, 1u);
    mm.release_pinned(p2, 4096);
}

TEST(MemoryManager, DifferentSizesGetDifferentBlocks) {
    MemoryManager mm;
    void* a = mm.acquire_pinned(1024);
    void* b = mm.acquire_pinned(2048);
    EXPECT_NE(a, b);
    mm.release_pinned(a, 1024);
    mm.release_pinned(b, 2048);
}

TEST(MemoryManager, DevicePoolRoundTrip) {
    MemoryManager mm;
    void* p = mm.acquire_device(8192);
    ASSERT_NE(p, nullptr);
    mm.release_device(p, 8192);
    EXPECT_GE(mm.stats().device_bytes_allocated, 8192u);
}

TEST(MemoryManager, FlushClearsAllPools) {
    MemoryManager mm;
    void* p = mm.acquire_pinned(512);
    mm.release_pinned(p, 512);
    mm.flush();
    // After flush, next acquire must re-allocate (no cache hit)
    void* q = mm.acquire_pinned(512);
    ASSERT_NE(q, nullptr);
    mm.release_pinned(q, 512);
    // Cache hits should still be 0 after flush cleared the pool
    EXPECT_EQ(mm.stats().pinned_cache_hits, 0u);
}
