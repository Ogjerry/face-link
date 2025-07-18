#include "../../include/nerf/renderer.cuh"
#include "../../include/common/math_utils.h"

////////////////////////////////////////////////////////////////////////////////
// KERNEL IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Step 1 (Forward): Generates sample points along rays, culls them
 * with the occupancy grid, and stores valid samples in global workspace buffers.
 */
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
    float* d_all_sample_dts,
    uint32_t* d_ray_indices,
    RayCompactInfo* d_ray_info
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    curandState state;
    curand_init(1337 + ray_idx, 0, 0, &state);

    const float near_plane = d_near_bounds[ray_idx];
    const float far_plane = d_far_bounds[ray_idx];

    // Mark as invalid initially
    d_ray_info[ray_idx] = {-1, 0};
    if (near_plane >= far_plane) return;

    const float3 origin = d_ray_origins[ray_idx];
    const float3 direction = d_ray_directions[ray_idx];
    
    int sample_start_idx = -1;
    int samples_for_this_ray = 0;

    // --- STRATIFIED SAMPLING (THE FIX) ---
    // Instead of fully random, we create ordered bins and sample randomly within each bin.
    // This gives us sorted samples automatically.
    const float step_size = (far_plane - near_plane) / N_MAIN_SAMPLES;

    // Generate N_MAIN_SAMPLES samples per ray
    for (int i = 0; i < N_MAIN_SAMPLES; ++i) {
        // Uniformly sample along the ray
        const float t_start = near_plane + i * step_size;
        const float t_end = t_start + step_size;

        // Jitter the sample within the bin
        const float t = t_start + curand_uniform(&state) * (t_end - t_start);
        
        const float3 pos = origin + t * direction;

        // Occupancy Grid Culling
        if (grid != nullptr) {
            // Normalize position to [0, 1] for grid lookup
            float3 norm_pos = make_float3(
                (pos.x + 1.5f) / 3.f, 
                (pos.y + 1.5f) / 3.f, 
                (pos.z + 1.5f) / 3.f
            );
            if (!grid->is_occupied(norm_pos)) {
                continue; // Skip this sample if the voxel is empty
            }
        }

        // Get a unique global index for this valid sample
        const int global_sample_idx = atomicAdd(d_sample_counter, 1);

        if (global_sample_idx >= max_samples) {
            atomicSub(d_sample_counter, 1);
            break;
        }

        // Record the starting index for this ray's samples
        if (sample_start_idx == -1) {
            sample_start_idx = global_sample_idx;
        }

        // Store sample data in the global workspace buffers
        d_all_sample_positions[global_sample_idx] = pos;
        d_ray_indices[global_sample_idx] = ray_idx;
        // We will calculate dt later, for now store t
        d_all_sample_dts[global_sample_idx] = t;

        samples_for_this_ray++;
    }

    // Store the compact info for this ray
    if (samples_for_this_ray > 0) {
        d_ray_info[ray_idx] = {sample_start_idx, samples_for_this_ray};
    }
}

/**
 * @brief Step 2 (Forward): Takes all valid sample points and computes their
 * hash grid and spherical harmonics features in a single batch.
 */
__global__ void evaluate_samples_kernel(
    int n_total_samples,
    const DeviceHashTableAccessor* table_acc,
    const float3* __restrict__ d_all_sample_positions,
    const uint32_t* __restrict__ d_ray_indices,
    const float3* __restrict__ d_ray_directions,
    int n_rays,
    // OUTPUTS
    __half* d_all_hash_features,
    __half* d_all_sh_features
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_total_samples) return;

    const uint32_t ray_idx = d_ray_indices[sample_idx];

    // --- ADD THIS DEBUG CHECK ---
    if (ray_idx >= n_rays) {
        printf("FATAL OOB: sample_idx %d has invalid ray_idx %u (n_rays is %d)\n", sample_idx, ray_idx, n_rays);
        return; 
    }
    // --- END DEBUG CHECK ---

    const float3 pos = d_all_sample_positions[sample_idx];
    const float3 dir = d_ray_directions[ray_idx];

    // Compute and store hash features
    hash_encode_kernel(pos, table_acc, &d_all_hash_features[sample_idx * MLP::D_in]);

    // Compute and store spherical harmonics features
    sh_encode(dir, &d_all_sh_features[sample_idx * SH_COEFS]);
}


/**
 * @brief Step 3 (Forward): Performs volumetric composition ray-by-ray
 * using the globally computed densities and colors.
 */
__global__ void composite_rays_kernel(
    int n_rays,
    int max_samples,
    const float* d_near_bounds,
    const float* d_far_bounds, 
    const RayCompactInfo* d_ray_info,
    const float* d_all_sample_dts, // Now contains 't' values
    const __half* __restrict__ d_all_densities, // Activated densities
    const __half* __restrict__ d_all_colors,    // Activated colors
    // OUTPUTS
    float4* d_out_pixels,
    float* d_all_weights, // To be used by backward pass
    float* d_all_alphas   // To be used by backward pass
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    const float near_plane = d_near_bounds[ray_idx];
    const float far_plane = d_far_bounds[ray_idx];

    const float dt = (far_plane - near_plane) / (float) N_MAIN_SAMPLES;
    
    const RayCompactInfo info = d_ray_info[ray_idx];
    if (info.n_samples <= 0) {
        d_out_pixels[ray_idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    float3 acc_color = make_float3(0.f, 0.f, 0.f);
    float acc_trans = 1.f;

    // The samples for a ray are not sorted, but composition is order-dependent.
    // For a fully correct implementation, a sort would be needed here per-ray.
    // For performance, we proceed assuming the random order is acceptable for stochastic training.

    for (int i = 0; i < info.n_samples; ++i) {
        const int s_idx = info.sample_start_idx + i;
        if (s_idx >= max_samples) {
            continue;
        }
        if (acc_trans < 1e-4f) break;

        const float density = __half2float(d_all_densities[s_idx]);
        
        // A simple, approximate dt calculation between random samples
        // const float dt = (i > 0) ? fabsf(d_all_sample_dts[s_idx] - d_all_sample_dts[s_idx - 1]) : 1e-3f;
        
        const float alpha = 1.f - expf(-density * dt);
        const float weight = acc_trans * alpha;

        const float3 color = make_float3(
            __half2float(d_all_colors[s_idx * 3 + 0]),
            __half2float(d_all_colors[s_idx * 3 + 1]),
            __half2float(d_all_colors[s_idx * 3 + 2])
        );

        acc_color += weight * color;
        acc_trans *= (1.f - alpha);

        // Store weight and alpha for the backward pass
        d_all_weights[s_idx] = weight;
        d_all_alphas[s_idx] = alpha;
    }

    d_out_pixels[ray_idx] = make_float4(acc_color.x, acc_color.y, acc_color.z, 1.f - acc_trans);
}

/**
 * @brief Step 4 (Backward): Computes the initial gradient for each sample's
 * density and color based on the final pixel's color gradient.
 */
__global__ void compute_backward_initial_grads_kernel(
    int n_rays,
    const float3* __restrict__ dL_d_predicted_rgb,
    const RayCompactInfo* d_ray_info,
    const float* __restrict__ d_all_sample_dts,
    const __half* __restrict__ d_all_colors,
    const float* __restrict__ d_all_weights,
    const float* __restrict__ d_all_alphas,
    // OUTPUTS
    __half* dL_d_density,
    __half* dL_d_color
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    const RayCompactInfo info = d_ray_info[ray_idx];
    if (info.n_samples <= 0) return;

    const float3 dL_d_pixel = dL_d_predicted_rgb[ray_idx];
    float dL_d_transmittance = 0.f;

    // Loop backwards over samples for this ray
    for (int i = info.n_samples - 1; i >= 0; --i) {
        const int s_idx = info.sample_start_idx + i;

        const float weight = d_all_weights[s_idx];
        const float alpha = d_all_alphas[s_idx];
        const float3 color = make_float3(
            __half2float(d_all_colors[s_idx * 3 + 0]),
            __half2float(d_all_colors[s_idx * 3 + 1]),
            __half2float(d_all_colors[s_idx * 3 + 2])
        );
        const float dt = (i > 0) ? fabsf(d_all_sample_dts[s_idx] - d_all_sample_dts[s_idx-1]) : 1e-3f;

        // Gradient w.r.t color_i is straightforward
        float3 dL_d_c = dL_d_pixel * weight;

        // Gradient w.r.t alpha_i (from user's original code)
        float dL_d_alpha = dot(dL_d_pixel, color) * (weight / (alpha + 1e-10f)) - dL_d_transmittance * (1.f - alpha);
        
        // Gradient w.r.t density_i
        float dL_d_sigma = dL_d_alpha * (1.f - alpha) * dt;

        // Update transmittance gradient for the next (previous) sample
        dL_d_transmittance = dL_d_transmittance * (1.f - alpha) + dot(dL_d_pixel, color) * alpha;

        // Store the results
        dL_d_density[s_idx] = __float2half(dL_d_sigma);
        dL_d_color[s_idx * 3 + 0] = __float2half(dL_d_c.x);
        dL_d_color[s_idx * 3 + 1] = __float2half(dL_d_c.y);
        dL_d_color[s_idx * 3 + 2] = __float2half(dL_d_c.z);
    }
}


/**
 * @brief Step 5 (Backward): Wrapper kernel to call the hash grid gradient
 * accumulation function for all samples in the batch.
 */
__global__ void accumulate_hash_gradients_kernel(
    int n_total_samples,
    DeviceHashTableAccessor* table_acc,
    const float3* __restrict__ d_all_sample_positions,
    const __half* __restrict__ dL_d_hash_features
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_total_samples) return;

    const float3 pos = d_all_sample_positions[sample_idx];
    
    // Convert __half gradient to float for accumulation
    float dL_d_feat_float[MLP::D_in];
    const __half* dL_d_feat_half = &dL_d_hash_features[sample_idx * MLP::D_in];
    for (int i = 0; i < MLP::D_in; ++i) {
        dL_d_feat_float[i] = __half2float(dL_d_feat_half[i]);
    }

    // Call the device function from hashing.cuh
    accumulate_gradients_merged_kernel(table_acc, pos, dL_d_feat_float);
}


__global__ void calculate_loss_kernel(
    const float4* predicted_pixels, const float3* ground_truth_colors, int batch_size,
    float3* dL_d_predicted_rgb, float* d_per_ray_mse_loss
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float3 predicted_rgb = make_float3(predicted_pixels[i].x, predicted_pixels[i].y, predicted_pixels[i].z);
    float3 actual_rgb = ground_truth_colors[i];
    float3 diff = predicted_rgb - actual_rgb;

    dL_d_predicted_rgb[i] = 2.f * diff;
    d_per_ray_mse_loss[i] = dot(diff, diff);

}

////////////////////////////////////////////////////////////////////////////////
// RENDERER CLASS IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////////////

void Renderer::render_forward(
    cublasHandle_t handle,
    cudaStream_t stream,
    const DeviceHashTableAccessor* table_acc,
    const MLP* mlp,
    const OccupancyGrid* grid,
    int n_rays,
    const float3* d_ray_origins,
    const float3* d_ray_directions,
    const float* d_near_bounds,
    const float* d_far_bounds, 
    RendererWorkspace& ws,
    float4* d_out_pixels
) {
    const int threads = 256;

    // --- 1. Generate Samples ---
    // Each thread processes one ray and writes valid samples to global buffers
    CHECK_CUDA_THROW(cudaMemsetAsync(ws.d_sample_counter.get(), 0, sizeof(int), stream));
    dim3 grid_gen_samples((n_rays + threads - 1) / threads);
    generate_samples_kernel<<<grid_gen_samples, threads, 0, stream>>>(
        n_rays, ws.max_samples, d_near_bounds, d_far_bounds, grid, d_ray_origins, d_ray_directions,
        ws.d_sample_counter.get(), ws.d_all_sample_positions.get(), ws.d_all_sample_dts.get(),
        ws.d_ray_indices.get(), ws.d_ray_info.get()
    );
    // DEBUG
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("generate_samples_kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
    // DEBUG

    // --- Get total sample count ---
    int n_total_samples;
    CHECK_CUDA_THROW(cudaMemcpyAsync(&n_total_samples, ws.d_sample_counter.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_THROW(cudaStreamSynchronize(stream)); // Wait for count to be ready on host
    
    // DEBUG
    // int device_id;
    // cudaGetDevice(&device_id);
    // printf("GPU %d: Generated %d samples (max: %d)\n", device_id, n_total_samples, ws.max_samples);
    // if (n_total_samples > ws.max_samples) {
    //     printf("ERROR: Sample overflow! %d > %d\n", n_total_samples, ws.max_samples);
    // }
    // DEBUG
    
    if (n_total_samples == 0) return; // No samples, nothing to do

    // --- 2. Evaluate Samples (Feature Encoding) ---
    dim3 grid_eval_samples((n_total_samples + threads - 1) / threads);
    evaluate_samples_kernel<<<grid_eval_samples, threads, 0, stream>>>(
        n_total_samples, table_acc, ws.d_all_sample_positions.get(),
        ws.d_ray_indices.get(), d_ray_directions, n_rays, ws.d_all_hash_features.get(),
        ws.d_all_sh_features.get()
    );

    // --- 3. Batched MLP Forward Pass ---
    // This single call replaces thousands of small kernel launches
    mlp->forward(
        handle, stream, n_total_samples,
        ws.d_all_hash_features.get(),
        ws.d_all_sh_features.get(),
        ws.d_mlp_hidden1.get(), // Pass workspace buffers
        ws.d_mlp_density_out_full.get(),
        ws.d_mlp_color_net_input.get(),
        ws.d_mlp_color_hidden1.get(),
        ws.d_mlp_color_hidden2.get(),
        ws.d_mlp_rgb_out.get(),
        ws.d_all_raw_densities.get(), // This is the activated density
        ws.d_all_raw_colors.get()     // This is the activated color
    );

    // DEBUG
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("MLP forward failed: %s\n", cudaGetErrorString(err));
    // }
    // DEBUG
    
    // --- 4. Composite Rays ---
    dim3 grid_composite((n_rays + threads - 1) / threads);
    composite_rays_kernel<<<grid_composite, threads, 0, stream>>>(
        n_rays, 
        ws.max_samples,
        d_near_bounds,
        d_far_bounds, 
        ws.d_ray_info.get(), ws.d_all_sample_dts.get(),
        ws.d_all_raw_densities.get(), ws.d_all_raw_colors.get(),
        d_out_pixels, ws.d_all_weights.get(), ws.d_all_alphas.get()
    );

    // DEBUG
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("composite_rays_kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
    // DEBUG
}


void Renderer::render_backward(
    cublasHandle_t handle,
    cudaStream_t stream,
    DeviceHashTableAccessor* table_acc,
    MLP* mlp,
    int n_rays,
    const float3* dL_d_predicted_rgb,
    const RendererWorkspace& ws
) {
    const int threads = 256;

    // Get total sample count from the workspace (was determined in forward pass)
    int n_total_samples;
    CHECK_CUDA_THROW(cudaMemcpy(&n_total_samples, ws.d_sample_counter.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (n_total_samples == 0) return;

    // --- 1. Compute Initial Gradients from Pixel Loss ---
    dim3 grid_bwd_init((n_rays + threads - 1) / threads);
    compute_backward_initial_grads_kernel<<<grid_bwd_init, threads, 0, stream>>>(
        n_rays, dL_d_predicted_rgb, ws.d_ray_info.get(), ws.d_all_sample_dts.get(),
        ws.d_all_raw_colors.get(), ws.d_all_weights.get(), ws.d_all_alphas.get(),
        ws.dL_d_density.get(), ws.dL_d_color.get()
    );

    // --- 2. Batched MLP Backward Pass ---
    // This single call backpropagates through the entire MLP for all samples
    mlp->backward(
        handle, stream, n_total_samples,
        ws.dL_d_density.get(), ws.dL_d_color.get(), // Gradients
        ws.d_all_hash_features.get(), ws.d_all_sh_features.get(), // Activations
        ws.d_mlp_hidden1.get(), ws.d_mlp_density_out_full.get(),
        ws.d_mlp_color_net_input.get(), 
        ws.d_mlp_color_hidden1.get(),
        ws.d_mlp_color_hidden2.get(),
        ws.d_mlp_rgb_out.get(),
        ws.dL_d_hash_features.get(), // Final output gradient
        ws.dL_d_sh_features.get()
    );

    // --- 3. Accumulate Gradients into Hash Table ---
    dim3 grid_accum_grads((n_total_samples + threads - 1) / threads);
    accumulate_hash_gradients_kernel<<<grid_accum_grads, threads, 0, stream>>>(
        n_total_samples, (DeviceHashTableAccessor*)table_acc,
        ws.d_all_sample_positions.get(), ws.dL_d_hash_features.get()
    );
}