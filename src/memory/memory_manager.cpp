#include "memory_manager.h"
#include <stdexcept>
#include <cstdlib>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA alloc error: ") +     \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)
#endif

namespace acr {

MemoryManager::MemoryManager() = default;

MemoryManager::~MemoryManager() {
    flush();
}

void* MemoryManager::acquire_pinned(size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = pinned_pool_.find(bytes);
    if (it != pinned_pool_.end() && !it->second.empty()) {
        void* ptr = it->second.back().ptr;
        it->second.pop_back();
        stats_.pinned_cache_hits++;
        return ptr;
    }
    void* ptr = nullptr;
#ifdef HAVE_CUDA
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
#else
    ptr = std::malloc(bytes);
    if (!ptr) throw std::bad_alloc();
#endif
    stats_.pinned_bytes_allocated += bytes;
    return ptr;
}

void MemoryManager::release_pinned(void* ptr, size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    pinned_pool_[bytes].push_back({ptr, bytes});
}

void* MemoryManager::acquire_device(size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = device_pool_.find(bytes);
    if (it != device_pool_.end() && !it->second.empty()) {
        void* ptr = it->second.back().ptr;
        it->second.pop_back();
        stats_.device_cache_hits++;
        return ptr;
    }
    void* ptr = nullptr;
#ifdef HAVE_CUDA
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
#else
    ptr = std::malloc(bytes);
    if (!ptr) throw std::bad_alloc();
#endif
    stats_.device_bytes_allocated += bytes;
    return ptr;
}

void MemoryManager::release_device(void* ptr, size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    device_pool_[bytes].push_back({ptr, bytes});
}

void MemoryManager::flush() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& [size, blocks] : pinned_pool_) {
        for (auto& b : blocks) {
#ifdef HAVE_CUDA
            cudaFreeHost(b.ptr);
#else
            std::free(b.ptr);
#endif
        }
        blocks.clear();
    }
    for (auto& [size, blocks] : device_pool_) {
        for (auto& b : blocks) {
#ifdef HAVE_CUDA
            cudaFree(b.ptr);
#else
            std::free(b.ptr);
#endif
        }
        blocks.clear();
    }
}

MemoryManager::Stats MemoryManager::stats() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return stats_;
}

} // namespace acr
