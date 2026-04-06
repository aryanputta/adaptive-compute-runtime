#include <cuda_runtime.h>
#include <cstddef>

// Same kernel as CPU baseline: fused scale-bias-relu.
// y[i] = max(0, alpha * x[i] + beta)
__global__ void kernel_scale_bias_relu(const float* __restrict__ in,
                                       float*       __restrict__ out,
                                       size_t                   n,
                                       float                    alpha,
                                       float                    beta) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = alpha * in[i] + beta;
        out[i]  = v > 0.0f ? v : 0.0f;
    }
}

// Batched variant: processes multiple logical workloads packed contiguously in memory.
__global__ void kernel_scale_bias_relu_batched(const float* __restrict__ in,
                                               float*       __restrict__ out,
                                               const size_t* __restrict__ offsets,
                                               const size_t* __restrict__ lengths,
                                               int                        num_batches,
                                               float                      alpha,
                                               float                      beta) {
    int bid = blockIdx.y;
    if (bid >= num_batches) return;

    size_t base = offsets[bid];
    size_t len  = lengths[bid];
    size_t i    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < len) {
        float v = alpha * in[base + i] + beta;
        out[base + i] = v > 0.0f ? v : 0.0f;
    }
}

// Reduction kernel: sum of absolute values (useful for telemetry / activation stats).
__global__ void kernel_reduce_abs_sum(const float* __restrict__ in,
                                      float*       __restrict__ partial,
                                      size_t                   n) {
    extern __shared__ float shmem[];
    size_t tid = threadIdx.x;
    size_t i   = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + tid;

    float val = 0.0f;
    if (i      < n) val += fabsf(in[i]);
    if (i + blockDim.x < n) val += fabsf(in[i + blockDim.x]);
    shmem[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shmem[0];
}

// C-linkage wrappers for the .cpp executor to call without nvcc in the include path.
extern "C" {

void launch_scale_bias_relu(const float* d_in, float* d_out, size_t n,
                            float alpha, float beta, cudaStream_t stream) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    kernel_scale_bias_relu<<<grid, block, 0, stream>>>(d_in, d_out, n, alpha, beta);
}

void launch_scale_bias_relu_batched(const float* d_in, float* d_out,
                                    const size_t* d_offsets, const size_t* d_lengths,
                                    int num_batches, size_t max_len,
                                    float alpha, float beta, cudaStream_t stream) {
    int block   = 256;
    int grid_x  = static_cast<int>((max_len + block - 1) / block);
    dim3 grid(grid_x, num_batches);
    kernel_scale_bias_relu_batched<<<grid, block, 0, stream>>>(
        d_in, d_out, d_offsets, d_lengths, num_batches, alpha, beta);
}

} // extern "C"
