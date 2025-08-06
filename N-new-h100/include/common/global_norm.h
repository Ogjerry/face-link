#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/async/transform.h>

#include "../nerf/mlp.cuh"
#include "../nerf/hashing.cuh"



template <typename T>
struct square {
    __host__ __device__ T operator() (const T& x) const {
        return x * x;
    }
};

template <typename T>
__global__ void scale_gradients_kernel(T* grads, size_t num_elements, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grads[idx] *= scale_factor;
    }
};

void clip_gradients_by_global_norm(
    MLP* mlp,
    HashTable* hash_table,
    float clip_threshold,
    cudaStream_t stream
) {
    if (clip_threshold <= 0.0f) return;

    // --- 1. Get pointers and sizes for all gradient buffers ---
    float* d_mlp_grads = mlp->all_gradients_ptr();
    size_t n_mlp_elements = mlp->all_gradients_size_in_elements();

    float* d_hash_grads = (float*)hash_table->gradients(); // Cast to float* for Thrust
    size_t n_hash_elements = hash_table->gradients_count_floats();

    // --- 2. Calculate sum of squares for each buffer on the GPU ---
    float mlp_sum_sq = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        d_mlp_grads,
        d_mlp_grads + n_mlp_elements,
        square<float>(),
        0.0f,
        thrust::plus<float>()
    );

    float hash_sum_sq = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        d_hash_grads,
        d_hash_grads + n_hash_elements,
        square<float>(),
        0.0f,
        thrust::plus<float>()
    );
    
    // --- 3. Calculate the global norm ---
    // A sync is implicitly needed here to get the sum results from the device.
    CHECK_CUDA_THROW(cudaStreamSynchronize(stream));
    float total_sum_sq = mlp_sum_sq + hash_sum_sq;
    float global_grad_norm = sqrtf(total_sum_sq);

    // --- 4. Apply clipping if the norm exceeds the threshold ---
    if (global_grad_norm > clip_threshold) {
        float scale_factor = clip_threshold / global_grad_norm;
        
        const int threads = 256;
        
        // Scale MLP gradients
        int mlp_blocks = (n_mlp_elements + threads - 1) / threads;
        scale_gradients_kernel<<<mlp_blocks, threads, 0, stream>>>(d_mlp_grads, n_mlp_elements, scale_factor);

        // Scale Hash Table gradients
        int hash_blocks = (n_hash_elements + threads - 1) / threads;
        scale_gradients_kernel<<<hash_blocks, threads, 0, stream>>>((float*)d_hash_grads, n_hash_elements, scale_factor);
    }
}

