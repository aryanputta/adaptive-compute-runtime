#include "gpu_executor.h"
#include "profiler.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cstring>

// Forward declarations for CUDA kernel launchers (defined in gpu_kernels.cu).
extern "C" {
void launch_scale_bias_relu(const float* d_in, float* d_out, size_t n,
                            float alpha, float beta, void* stream);
void launch_scale_bias_relu_batched(const float* d_in, float* d_out,
                                    const size_t* d_offsets, const size_t* d_lengths,
                                    int num_batches, size_t max_len,
                                    float alpha, float beta, void* stream);
}

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#endif

namespace acr {

#ifdef HAVE_CUDA
static constexpr float kAlpha = 1.0f / 255.0f;
static constexpr float kBeta  = -0.5f;
#endif

GPUExecutor::GPUExecutor(MemoryManager& mm) : mm_(mm), gpu_available_(false), device_id_(0) {
    check_gpu();
}

GPUExecutor::~GPUExecutor() = default;

void GPUExecutor::check_gpu() {
#ifdef HAVE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    gpu_available_ = (err == cudaSuccess && count > 0);
    if (gpu_available_) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
    }
#else
    gpu_available_ = false;
#endif
}

ExecutionResult GPUExecutor::run(WorkloadDescriptor& wd) {
    ExecutionResult r{};
    r.workload_id = wd.id;
    r.path_used   = ExecutionPath::GPU_DIRECT;

#ifdef HAVE_CUDA
    if (!gpu_available_) {
        throw std::runtime_error("GPU not available");
    }

    size_t bytes = wd.element_count * sizeof(float);

    // Acquire pinned host memory and device buffers from pool.
    auto* h_in  = static_cast<float*>(mm_.acquire_pinned(bytes));
    auto* h_out = static_cast<float*>(mm_.acquire_pinned(bytes));
    auto* d_in  = static_cast<float*>(mm_.acquire_device(bytes));
    auto* d_out = static_cast<float*>(mm_.acquire_device(bytes));

    std::memcpy(h_in, wd.input_data, bytes);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // CUDA events for fine-grained timing
    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_k_start, ev_k_end, ev_d2h_start, ev_d2h_end;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK(cudaEventCreate(&ev_k_start));
    CUDA_CHECK(cudaEventCreate(&ev_k_end));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_end));

    WallTimer wall;

    // H2D
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev_h2d_end, stream));

    // Kernel
    CUDA_CHECK(cudaEventRecord(ev_k_start, stream));
    launch_scale_bias_relu(d_in, d_out, wd.element_count, kAlpha, kBeta, stream);
    CUDA_CHECK(cudaEventRecord(ev_k_end, stream));

    // D2H
    CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev_d2h_end, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    r.wall_ms = wall.elapsed_ms();

    float h2d_ms, k_ms, d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&k_ms,   ev_k_start,   ev_k_end));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_end));

    r.h2d_ms    = h2d_ms;
    r.kernel_ms = k_ms;
    r.d2h_ms    = d2h_ms;
    r.memory_throughput_GBs =
        (static_cast<double>(bytes) * 2.0) / (k_ms * 1e-3 * 1e9);

    std::memcpy(wd.output_data, h_out, bytes);
    r.correct = true;

    // Cleanup events and stream
    cudaEventDestroy(ev_h2d_start); cudaEventDestroy(ev_h2d_end);
    cudaEventDestroy(ev_k_start);   cudaEventDestroy(ev_k_end);
    cudaEventDestroy(ev_d2h_start); cudaEventDestroy(ev_d2h_end);
    cudaStreamDestroy(stream);

    mm_.release_pinned(h_in,  bytes);
    mm_.release_pinned(h_out, bytes);
    mm_.release_device(d_in,  bytes);
    mm_.release_device(d_out, bytes);
#else
    // Stub: when compiled without CUDA, fall through to CPU timing
    (void)wd;
    r.wall_ms = 0.0;
    r.correct = false;
#endif

    return r;
}

std::vector<ExecutionResult> GPUExecutor::run_batch(std::vector<WorkloadDescriptor>& batch) {
    std::vector<ExecutionResult> results;
    results.reserve(batch.size());

#ifdef HAVE_CUDA
    // Coalesce all inputs into one pinned buffer.
    size_t total_elements = 0;
    for (const auto& wd : batch) total_elements += wd.element_count;

    size_t total_bytes = total_elements * sizeof(float);

    auto* h_in_flat  = static_cast<float*>(mm_.acquire_pinned(total_bytes));
    auto* h_out_flat = static_cast<float*>(mm_.acquire_pinned(total_bytes));
    auto* d_in_flat  = static_cast<float*>(mm_.acquire_device(total_bytes));
    auto* d_out_flat = static_cast<float*>(mm_.acquire_device(total_bytes));

    std::vector<size_t> offsets(batch.size()), lengths(batch.size());
    size_t offset = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        offsets[i] = offset;
        lengths[i] = batch[i].element_count;
        std::memcpy(h_in_flat + offset, batch[i].input_data,
                    batch[i].element_count * sizeof(float));
        offset += batch[i].element_count;
    }

    size_t meta_bytes = batch.size() * sizeof(size_t);
    auto* d_offsets = static_cast<size_t*>(mm_.acquire_device(meta_bytes));
    auto* d_lengths = static_cast<size_t*>(mm_.acquire_device(meta_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    WallTimer wall;
    CUDA_CHECK(cudaMemcpyAsync(d_in_flat, h_in_flat, total_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(), meta_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lengths, lengths.data(), meta_bytes, cudaMemcpyHostToDevice, stream));

    size_t max_len = *std::max_element(lengths.begin(), lengths.end());

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    launch_scale_bias_relu_batched(d_in_flat, d_out_flat, d_offsets, d_lengths,
                                   static_cast<int>(batch.size()), max_len,
                                   kAlpha, kBeta, stream);
    CUDA_CHECK(cudaEventRecord(ev_end, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_out_flat, d_out_flat, total_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double wall_ms = wall.elapsed_ms();
    float k_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_end));

    // Copy outputs back and build per-workload results.
    for (size_t i = 0; i < batch.size(); ++i) {
        std::memcpy(batch[i].output_data, h_out_flat + offsets[i],
                    lengths[i] * sizeof(float));
        ExecutionResult r{};
        r.workload_id  = batch[i].id;
        r.path_used    = ExecutionPath::GPU_BATCHED;
        r.wall_ms      = wall_ms / static_cast<double>(batch.size());  // amortized
        r.kernel_ms    = k_ms;
        r.h2d_ms       = 0.0;  // bundled
        r.d2h_ms       = 0.0;
        r.correct      = true;
        results.push_back(r);
    }

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);

    mm_.release_pinned(h_in_flat,  total_bytes);
    mm_.release_pinned(h_out_flat, total_bytes);
    mm_.release_device(d_in_flat,  total_bytes);
    mm_.release_device(d_out_flat, total_bytes);
    mm_.release_device(d_offsets,  meta_bytes);
    mm_.release_device(d_lengths,  meta_bytes);
#else
    for (auto& wd : batch) {
        ExecutionResult r{};
        r.workload_id = wd.id;
        r.path_used   = ExecutionPath::GPU_BATCHED;
        r.correct     = false;
        results.push_back(r);
    }
#endif

    return results;
}

} // namespace acr
