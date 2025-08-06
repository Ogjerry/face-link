#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "../nerf/renderer.cuh"
#include <device_launch_parameters.h>
#include <crt/device_functions.h>
#include <cuda_bf16.h>

// In renderer.cu, replace your old distortion loss kernels with these.

// A numerically stable parallel prefix sum (scan) using shared memory.
// This is a common and highly efficient pattern in CUDA programming.
template <typename T>
__device__ void block_scan(T* temp, int n_valid_samples) {
    int tid = threadIdx.x;
    
    // Each thread loads one element into shared memory (already done before calling)
    // Parallel reduction phase (up-sweep)
    for (int d = 1; d < n_valid_samples; d *= 2) {
        __syncthreads();
        if (tid >= d) {
            temp[tid] += temp[tid - d];
        }
    }
    __syncthreads();
}

/**
 * @brief EFFICIENT O(N) forward pass for distortion loss.
 * LAUNCH CONFIG: <<<n_rays, n_samples_per_ray, shared_mem_size>>>
 * Each CUDA block processes one ray. Threads in the block cooperate.
 */
__global__ void distortion_loss_fwd_kernel_efficient(
    int n_rays,
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_ts,
    const float* __restrict__ d_all_sample_dts,
    const float* __restrict__ d_all_weights,
    float* d_total_distortion_loss // Output: single float for total loss
) {
    const int ray_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const RayCompactInfo info = d_ray_info[ray_idx];
    if (info.n_samples <= 1) {
        return;
    }

    // Shared memory for this block (one ray)
    extern __shared__ float s_data[];
    float* s_w = s_data;                                // size: n_samples
    float* s_wm = &s_data[blockDim.x];              // size: n_samples
    float* s_w_prefix = &s_data[2 * blockDim.x];    // size: n_samples
    float* s_wm_prefix = &s_data[3 * blockDim.x];   // size: n_samples

    // Step 1: Each thread loads its sample data into shared memory
    if (tid < info.n_samples) {
        const int s_idx = info.sample_start_idx + tid;
        const float w = d_all_weights[s_idx];
        const float m = d_all_sample_ts[s_idx] + d_all_sample_dts[s_idx] * 0.5f;
        s_w[tid] = w;
        s_wm[tid] = w * m;
    }
    __syncthreads();

    // Step 2: Compute prefix sums in parallel
    if (tid < info.n_samples) {
        s_w_prefix[tid] = s_w[tid];
        s_wm_prefix[tid] = s_wm[tid];
    }
    block_scan(s_w_prefix, info.n_samples);
    block_scan(s_wm_prefix, info.n_samples);

    // Step 3: Each thread computes its contribution to the loss
    float loss_term1 = 0.f;
    float loss_term2 = 0.f;

    if (tid < info.n_samples) {
        const int s_idx = info.sample_start_idx + tid;
        const float w_i = s_w[tid];
        const float m_i = d_all_sample_ts[s_idx] + d_all_sample_dts[s_idx] * 0.5f;

        if (tid > 0) {
            const float w_sum_exclusive = s_w_prefix[tid - 1];
            const float wm_sum_exclusive = s_wm_prefix[tid - 1];
            loss_term1 = w_i * (m_i * w_sum_exclusive - wm_sum_exclusive);
        }
        
        loss_term2 = (1.0f / 3.0f) * w_i * w_i * d_all_sample_dts[s_idx];
    }

    // Step 4: Reduce the loss within the block and add to global total
    // A simple reduction using atomicAdd for clarity.
    // For max performance, a shared memory reduction would be used before one atomicAdd.
    float total_loss_for_thread = 2.0f * loss_term1 + loss_term2;
    if (total_loss_for_thread > 0.f) {
        atomicAdd(d_total_distortion_loss, total_loss_for_thread);
    }
}


/**
 * @brief PARALLELIZED O(N^2) backward pass for distortion loss.
 * LAUNCH CONFIG: <<<n_rays, n_samples_per_ray>>>
 * Each CUDA thread processes one sample of one ray.
 */
__global__ void distortion_loss_bwd_kernel_parallel(
    int n_rays,
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_ts,
    const float* __restrict__ d_all_sample_dts,
    const float* __restrict__ d_all_weights,
    const __half* __restrict__ d_all_raw_densities,
    float distortion_loss_weight,
    float* dL_d_density // Additive output
) {
    const int ray_idx = blockIdx.x;
    const int sample_k_offset = threadIdx.x; // This thread handles the k'th sample

    const RayCompactInfo info = d_ray_info[ray_idx];
    if (info.n_samples <= 1 || sample_k_offset >= info.n_samples) {
        return;
    }

    const int s_idx_k = info.sample_start_idx + sample_k_offset;
    const float midpoint_k = d_all_sample_ts[s_idx_k] + d_all_sample_dts[s_idx_k] / 2.f;

    float dL_dist_d_wk = 0.f;

    // Gradient of Term 1: This is still an O(N) loop *per thread*
    for (int j = 0; j < info.n_samples; ++j) {
        const int s_idx_j = info.sample_start_idx + j;
        float midpoint_j = d_all_sample_ts[s_idx_j] + d_all_sample_dts[s_idx_j] / 2.f;
        dL_dist_d_wk += 2.f * d_all_weights[s_idx_j] * abs(midpoint_k - midpoint_j);
    }

    // Gradient of Term 2
    dL_dist_d_wk += (2.f / 3.f) * d_all_weights[s_idx_k] * d_all_sample_dts[s_idx_k];

    // Chain rule to get gradient w.r.t density
    float raw_density = __half2float(d_all_raw_densities[s_idx_k]);
    float dt = d_all_sample_dts[s_idx_k];
    float alpha = 1.f - expf(-raw_density * dt);
    float d_alpha_d_sigma = (1.f - alpha) * dt;
    float dL_d_sigma = dL_dist_d_wk * d_alpha_d_sigma;
    
    // Atomically add the final gradient contribution
    if (abs(dL_d_sigma) > 1e-7f) {
         atomicAdd(&dL_d_density[s_idx_k], distortion_loss_weight * dL_d_sigma);
    }
}