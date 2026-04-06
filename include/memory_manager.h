#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace acr {

// Manages pinned host memory and device memory buffer pools to reduce
// repeated cudaMalloc / cudaFree overhead across workloads.
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Returns pinned host allocation of at least `bytes`. Reuses if available.
    void* acquire_pinned(size_t bytes);
    void  release_pinned(void* ptr, size_t bytes);

    // Returns device allocation of at least `bytes`. Reuses if available.
    void* acquire_device(size_t bytes);
    void  release_device(void* ptr, size_t bytes);

    // Flush all cached allocations back to the system.
    void flush();

    struct Stats {
        size_t pinned_bytes_allocated;
        size_t device_bytes_allocated;
        size_t pinned_cache_hits;
        size_t device_cache_hits;
    };
    Stats stats() const;

private:
    // Free-list keyed by exact size. Simple but effective for repeated same-shape workloads.
    struct Block { void* ptr; size_t size; };
    std::unordered_map<size_t, std::vector<Block>> pinned_pool_;
    std::unordered_map<size_t, std::vector<Block>> device_pool_;
    mutable std::mutex mtx_;
    Stats stats_ {};
};

} // namespace acr
