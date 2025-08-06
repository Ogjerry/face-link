#pragma once
#ifndef RENDERER_H
#define RENDERER_H

#include "../common/cuda_wrappers.h"
#include "hashing.cuh"
#include "mlp.cuh"
#include "occupancy_grid.cuh"
#include <cublas_v2.h>

const int N_PROPOSAL_SAMPLES = 64;
const int N_MAIN_SAMPLES = 128;

struct RayCompactInfo {
    int sample_start_idx;
    int n_samples;
};


// Helper struct for sorting
struct TValuePair {
    float t;
    int original_idx_offset; // To fetch the correct density/color
};



struct TWeightPair {
    float t;
    float weight;
};

__device__ inline void bitonic_sort_step_t_weight(TWeightPair* data, int i, int j) {
    if (data[j].t < data[i].t) {
        TWeightPair temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
};



__global__ void init_rng_kernel(curandState* states, int n, unsigned long long seed);

// NEW: A workspace to hold all temporary buffers for a single batch.
struct RendererWorkspace {
    const int max_samples;
    CudaDeviceBuffer<int> d_sample_counter;
    CudaDeviceBuffer<curandState> d_rng_states;
    CudaDeviceBuffer<float3> d_all_sample_positions;
    CudaDeviceBuffer<uint32_t> d_ray_indices;
    CudaDeviceBuffer<__half> d_all_hash_features;
    CudaDeviceBuffer<__half> d_all_sh_features;
    CudaDeviceBuffer<__half> d_all_raw_densities;
    CudaDeviceBuffer<__half> d_all_raw_colors;
    CudaDeviceBuffer<RayCompactInfo> d_ray_info;

    CudaDeviceBuffer<float> d_all_sample_ts;
    CudaDeviceBuffer<float> d_all_sample_dts;
    CudaDeviceBuffer<float> d_all_weights;
    CudaDeviceBuffer<float> d_all_alphas;

    CudaDeviceBuffer<__half> d_mlp_hidden1;
    CudaDeviceBuffer<__half> d_mlp_density_out_full;
    CudaDeviceBuffer<__half> d_mlp_color_net_input;
    CudaDeviceBuffer<__half> d_mlp_color_hidden1;
    CudaDeviceBuffer<__half> d_mlp_color_hidden2;
    CudaDeviceBuffer<__half> d_mlp_rgb_out;

    // Full Precision Gradients
    CudaDeviceBuffer<float> dL_d_density;
    CudaDeviceBuffer<float> dL_d_color;
    CudaDeviceBuffer<float> dL_d_hash_features;
    CudaDeviceBuffer<float> dL_d_sh_features;

    RendererWorkspace(int batch_size, int n_samples_per_ray) :
        max_samples(batch_size * n_samples_per_ray),
        d_sample_counter(1),
        d_rng_states(batch_size),
        d_all_sample_positions(max_samples),
        d_ray_indices(max_samples),
        d_all_hash_features(max_samples * MLP::D_in),
        d_all_sh_features(max_samples * SH_COEFS),
        d_all_raw_densities(max_samples),
        d_all_raw_colors(max_samples * 3),
        d_ray_info(batch_size),
        d_all_sample_ts(max_samples),
        d_all_sample_dts(max_samples),
        d_all_weights(max_samples),
        d_all_alphas(max_samples),
        d_mlp_hidden1(max_samples * MLP::D_hidden),
        d_mlp_density_out_full(max_samples * MLP::D_density_out),
        d_mlp_color_net_input(max_samples * MLP::D_color_in),
        d_mlp_color_hidden1(max_samples * MLP::D_color_hidden),
        d_mlp_color_hidden2(max_samples * MLP::D_color_hidden),
        d_mlp_rgb_out(max_samples * MLP::D_color_out),
        dL_d_density(max_samples),
        dL_d_color(max_samples * 3),
        dL_d_hash_features(max_samples * MLP::D_in),
        dL_d_sh_features(max_samples * SH_COEFS)
    {
        printf("d_all_hash_features size: %d\n", max_samples * MLP::D_in);

        cudaMemset(d_sample_counter.get(), 0, sizeof(int));

        init_rng_kernel<<<(batch_size + 255) / 256, 256>>>(
            d_rng_states.get(), batch_size, 12345ULL
        );
        CHECK_CUDA_THROW(cudaDeviceSynchronize());
    }
};

class Renderer {
public:
    void render_forward(
        cublasHandle_t handle,
        cudaStream_t stream,
        DeviceHashTableAccessor* table_acc,
        const MLP* mlp,
        const OccupancyGrid* grid,
        int n_rays,
        const float3* d_ray_origins,
        const float3* d_ray_directions,
        const float* d_near_bounds,
        const float* d_far_bounds,
        int n_coarse_samples,
        int n_fine_samples,
        RendererWorkspace& ws,
        float4* d_out_pixels
    );

    void render_backward(
        cublasHandle_t handle,
        cudaStream_t stream,
        DeviceHashTableAccessor* table_acc,
        MLP* mlp,
        int n_rays,
        const float3* dL_d_predicted_rgb,
        const RendererWorkspace& ws
    );
};


__global__ void generate_samples_kernel(
    int n_rays,
    int max_samples,
    const float* __restrict__ d_near_bounds,
    const float* __restrict__ d_far_bounds,
    const OccupancyGrid* grid,
    const float3* __restrict__ d_ray_origins,
    const float3* __restrict__ d_ray_directions,
    int* d_sample_counter,
    float3* d_all_sample_positions,
    float* d_all_sample_ts,
    uint32_t* d_ray_indices,
    RayCompactInfo* d_ray_info,
    curandState* __restrict__ d_rng_states,
    const float* d_weights, // Weights from the coarse pass
    int n_coarse_samples,
    int n_fine_samples
);


__global__ void evaluate_samples_kernel(
    int n_total_samples,
    DeviceHashTableAccessor* table_acc,
    const float3* __restrict__ d_all_sample_positions,
    const uint32_t* __restrict__ d_ray_indices,
    const float3* __restrict__ d_ray_directions,
    int n_rays,
    __half* d_all_hash_features,
    __half* d_all_sh_features
);

__global__ void composite_rays_kernel(
    int n_rays,
    int max_samples_per_ray,
    const float* __restrict__ d_near_bounds,
    const float* __restrict__ d_far_bounds, 
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_ts,
    const float* __restrict__ d_all_sample_dts, // Now contains 't' values
    const __half* __restrict__ d_all_densities, // Activated densities
    const __half* __restrict__ d_all_colors,    // Activated colors
    // OUTPUTS
    float4* d_out_pixels,
    float* d_all_weights, // To be used by backward pass
    float* d_all_alphas   // To be used by backward pass
);

__global__ void compute_backward_initial_grads_kernel(
    int n_rays,
    const float3* __restrict__ dL_d_predicted_rgb,
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_dts,
    const __half* __restrict__ d_all_colors,
    const float* __restrict__ d_all_weights,
    const float* __restrict__ d_all_alphas,
    float* dL_d_density,
    float* dL_d_color
);

__global__ void calculate_loss_kernel(
    const float4* predicted_pixels, 
    const float3* ground_truth_colors, 
    const RayCompactInfo* d_ray_info,
    int n_rays,
    float3* dL_d_predicted_rgb,// Output 
    float* d_per_ray_mse_loss, // Output
    float gradient_scale_factor
);

#endif // RENDERER_H