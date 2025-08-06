#include "../../include/nerf/renderer.cuh"
#include "../../include/common/math_utils.h"


#include <cooperative_groups.h>
#include <cmath>
#include <alloca.h>
namespace cg = cooperative_groups;

#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////
// KERNEL IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////////////


__global__ void init_rng_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}



/**
 * @brief this section has changed from ray sample based method,
 *        to image sample based method, then back to ray sample 
 *        methods again. let us see if the stupid gpt would make
 *        further spiraling error.
 */



// Device function for the parallel bitonic sort
__device__ void bitonic_sort_step(TValuePair* data, int j, int k) {
    float t_i = data[j].t;
    float t_k = data[k].t;

    bool compare = (t_i > t_k);
    
    // Perform swap if needed
    if (compare) {
        TValuePair temp = data[j];
        data[j] = data[k];
        data[k] = temp;
    }
}


/**
 * @brief Step 1 (Forward): Generates sample points along rays, culls them
 * with the occupancy grid, and stores valid samples in global workspace buffers.
 * This version implements a coarse-to-fine sampling strategy.
 */
// In src/nerf/renderer.cu

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
    const float* d_weights,
    int n_coarse_samples,
    int n_fine_samples
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    curandState state = d_rng_states[ray_idx];

    const float near_plane = d_near_bounds[ray_idx];
    const float far_plane = d_far_bounds[ray_idx];

    d_ray_info[ray_idx] = {-1, 0};


    const float3 origin = d_ray_origins[ray_idx];
    const float3 direction = d_ray_directions[ray_idx];

    int sample_start_idx = -1;
    int samples_for_this_ray = 0;

    // --- Coarse Pass ---
    const float step_size = (far_plane - near_plane) / n_coarse_samples;
    for (int i = 0; i < n_coarse_samples; ++i) {
        const float t_start = near_plane + i * step_size;
        const float t = t_start + curand_uniform(&state) * step_size;
        const float3 pos = origin + t * direction;

        // --- FIXED --- Use normalized coordinates for the occupancy grid check.
        // Debug occupancy grid checks
        if (grid != nullptr && ray_idx < 5) {
            float3 norm_pos = make_float3(
                (pos.x + 1.5f) / 3.f,
                (pos.y + 1.5f) / 3.f,
                (pos.z + 1.5f) / 3.f
            );
            // DEBUG
            bool occupied = grid->is_occupied(norm_pos);
            printf("[Ray %d, Sample %d] pos=(%.3f,%.3f,%.3f) norm=(%.3f,%.3f,%.3f) occupied=%d\n",
                ray_idx, i, pos.x, pos.y, pos.z, 
                norm_pos.x, norm_pos.y, norm_pos.z, occupied);
            // DEBUG
                
            if (!grid->is_occupied(norm_pos)) {
               continue;
            }
        }


        const int global_sample_idx = atomicAdd(d_sample_counter, 1);
        if (global_sample_idx >= max_samples) { atomicSub(d_sample_counter, 1); break; }
        if (sample_start_idx == -1) sample_start_idx = global_sample_idx;

        d_all_sample_positions[global_sample_idx] = pos;
        d_ray_indices[global_sample_idx] = ray_idx;
        d_all_sample_ts[global_sample_idx] = t;
        samples_for_this_ray++;
    }

    // --- Fine Pass ---
    if (d_weights != nullptr && samples_for_this_ray > 0) {
        float* cdf = (float*)alloca(sizeof(float) * (samples_for_this_ray + 1));
        cdf[0] = 0.f;
        float sum = 0.f;
        for (int i = 0; i < samples_for_this_ray; ++i) {
            sum += d_weights[sample_start_idx + i] + 1e-5f;
            cdf[i+1] = sum;
        }

        if (sum > 1e-5f) {
            for (int i = 1; i <= samples_for_this_ray; ++i) cdf[i] /= sum;

            // DEBUG: Add a debug print for the first ray
            //if (ray_idx == 0) {
            //    printf("--- Ray 0 Fine Sampling Debug ---\n");
            //    // Print the weights of the coarse samples
            //    for (int k = 0; k < samples_for_this_ray; k++) {
            //        printf("  Coarse Sample %d: t=%.3f, weight=%.5f\n", 
            //            k, d_all_sample_ts[sample_start_idx + k], d_weights[sample_start_idx + k]);
            //    }
            //}
            // DEBUG: Add a debug print for the first ray

            for (int i = 0; i < n_fine_samples; ++i) {
                float u = curand_uniform(&state);
                int idx = 0;
                for (int j = 0; j < samples_for_this_ray; ++j) {
                    if (u >= cdf[j] && u < cdf[j+1]) { idx = j; break; }
                }

                const float t_start = d_all_sample_ts[sample_start_idx + idx];
                const float t_end = (idx + 1 < samples_for_this_ray) ? d_all_sample_ts[sample_start_idx + idx + 1] : far_plane;
                const float t = t_start + (t_end - t_start) * curand_uniform(&state);
                const float3 pos = origin + t * direction;

                const int global_sample_idx = atomicAdd(d_sample_counter, 1);
                if (global_sample_idx >= max_samples) { atomicSub(d_sample_counter, 1); break; }

                d_all_sample_positions[global_sample_idx] = pos;
                d_ray_indices[global_sample_idx] = ray_idx;
                d_all_sample_ts[global_sample_idx] = t;
                samples_for_this_ray++;

                // DEBUG: Print where the new fine sample is being placed
                // if (ray_idx == 0) {
                //     printf("    -> Fine Sample %d placed in coarse interval %d (new t=%.3f)\n", i, idx, t);
                // }
                // DEBUG: Print where the new fine sample is being placed
            }
        }
    }

    

    if (samples_for_this_ray > 0) {
        d_ray_info[ray_idx] = {sample_start_idx, samples_for_this_ray};
    }
    
    d_rng_states[ray_idx] = state;
}


// Kernel 1: Generates only the initial coarse samples along rays.
__global__ void generate_coarse_samples_kernel(
    int n_rays,
    int n_coarse_samples,
    int max_samples,
    const float* __restrict__ d_near_bounds,
    const float* __restrict__ d_far_bounds,
    const OccupancyGrid* grid,
    const float3* __restrict__ d_ray_origins,
    const float3* __restrict__ d_ray_directions,
    curandState* __restrict__ d_rng_states,
    // --- Outputs ---
    int* d_sample_counter,
    float3* d_all_sample_positions,
    float* d_all_sample_ts,
    uint32_t* d_ray_indices,
    RayCompactInfo* d_coarse_ray_info // Note: Specific output for coarse pass info
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    curandState state = d_rng_states[ray_idx];

    const float near_plane = d_near_bounds[ray_idx];
    const float far_plane = d_far_bounds[ray_idx];

    // Initialize with invalid state
    d_coarse_ray_info[ray_idx] = {-1, 0};
    if (near_plane >= far_plane) {
        d_rng_states[ray_idx] = state;
        return;
    }

    const float3 origin = d_ray_origins[ray_idx];
    const float3 direction = d_ray_directions[ray_idx];

    int sample_start_idx = -1;
    int samples_for_this_ray = 0;
    const float step_size = (far_plane - near_plane) / n_coarse_samples;

    for (int i = 0; i < n_coarse_samples; ++i) {
        const float t_start = near_plane + i * step_size;
        const float t = t_start + curand_uniform(&state) * step_size;
        const float3 pos = origin + t * direction;

        // --- Occupancy Grid Culling ---
        // REMOVED "ray_idx < 5" to enable for all rays!
        if (grid != nullptr) {
            float3 norm_pos = make_float3(
                (pos.x + 1.5f) / 3.f,
                (pos.y + 1.5f) / 3.f,
                (pos.z + 1.5f) / 3.f
            );

            bool occupied = grid->is_occupied(norm_pos);
            // printf("[Ray %d, Sample %d] pos=(%.3f,%.3f,%.3f) norm=(%.3f,%.3f,%.3f) occupied=%d\n",
            //    ray_idx, i, pos.x, pos.y, pos.z, 
            //    norm_pos.x, norm_pos.y, norm_pos.z, occupied);

            if (!grid->is_occupied(norm_pos)) {
                continue; // Skip this empty sample
            }
        }

        const int global_sample_idx = atomicAdd(d_sample_counter, 1);
        if (global_sample_idx >= max_samples) {
            atomicSub(d_sample_counter, 1);
            break;
        }

        if (sample_start_idx == -1) {
            sample_start_idx = global_sample_idx;
        }

        d_all_sample_positions[global_sample_idx] = pos;
        d_ray_indices[global_sample_idx] = ray_idx;
        d_all_sample_ts[global_sample_idx] = t;
        samples_for_this_ray++;
    }

    if (samples_for_this_ray > 0) {
        d_coarse_ray_info[ray_idx] = {sample_start_idx, samples_for_this_ray};
    }

    d_rng_states[ray_idx] = state;
}


// Kernel 2: Generates fine samples based on the weights from the coarse pass.
__global__ void generate_fine_samples_kernel(
    int n_rays,
    int n_fine_samples,
    int max_samples,
    // --- Inputs from Coarse Pass ---
    const RayCompactInfo* __restrict__ d_coarse_ray_info,
    const float* __restrict__ d_all_sample_ts,
    const float* __restrict__ d_all_weights,
    // --- General Ray Data ---
    const float* __restrict__ d_far_bounds,
    const float3* __restrict__ d_ray_origins,
    const float3* __restrict__ d_ray_directions,
    curandState* __restrict__ d_rng_states,
    // --- Outputs ---
    int* d_sample_counter,
    float3* d_all_sample_positions,
    float* d_all_sample_ts_output,
    uint32_t* d_ray_indices,
    RayCompactInfo* d_fine_ray_info
) {
    const int thread_idx = threadIdx.x;
    const int ray_idx = blockIdx.x;
    if (ray_idx >= n_rays) return;

    // Define the CUB BlockScan type for convenience.
    // It's templated on the type we are scanning (float) and the block size.
    typedef cub::BlockScan<float, 128> BlockScan;

    curandState state = d_rng_states[ray_idx];
    const RayCompactInfo coarse_info = d_coarse_ray_info[ray_idx];

    d_fine_ray_info[ray_idx] = {-1, 0};
    if (coarse_info.n_samples <= 0) {
        d_rng_states[ray_idx] = state;
        return;
    }

    const float3 origin = d_ray_origins[ray_idx];
    const float3 direction = d_ray_directions[ray_idx];
    const float far_plane = d_far_bounds[ray_idx];

    float* cdf = (float*)alloca(sizeof(float) * (coarse_info.n_samples + 1));
    cdf[0] = 0.f;
    float sum = 0.f;
    for (int i = 0; i < coarse_info.n_samples; ++i) {
        sum += d_all_weights[coarse_info.sample_start_idx + i] + 1e-5f;
        cdf[i + 1] = sum;
    }

    if (sum < 1e-5f) {
        d_rng_states[ray_idx] = state;
        return;
    }

    for (int i = 1; i <= coarse_info.n_samples; ++i) {
        cdf[i] /= sum;
    }

    // =================== DEBUG PRINTING START (SECTION 1) ===================
    // This block prints the inputs to the sampling process for Ray 0.
    // if (ray_idx == 0) {
    //     printf("\n--- [Ray 0] Fine Sampling Debug ---\n");
    //     printf("Coarse samples for this ray: %d\n", coarse_info.n_samples);
    //     for (int k = 0; k < coarse_info.n_samples; k++) {
    //         printf("  Coarse[%d]: t=%.4f, weight=%.5f, cdf=%.5f\n", 
    //                k, 
    //                d_all_sample_ts[coarse_info.sample_start_idx + k], 
    //                d_all_weights[coarse_info.sample_start_idx + k],
    //                cdf[k+1]); // Also print the CDF value for this interval
    //     }
    //     printf("-------------------------------------\n");
    // }
    // =================== DEBUG PRINTING END (SECTION 1) =====================

    int sample_start_idx = -1;
    int samples_for_this_ray = 0;

    for (int i = 0; i < n_fine_samples; ++i) {
        const float u = curand_uniform(&state);

        // ====================================================================
        // === OPTIMIZATION: Replaced linear search with binary search    ===
        // ====================================================================
        int bin = 0;
        {
            int low = 0;
            int high = coarse_info.n_samples;

            while (low < high) {
                const int mid = low + (high - low) / 2;
                if (u >= cdf[mid]) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            bin = low - 1;
            if (bin < 0) bin = 0;
        }
        // ====================================================================
        
        const float t_start = d_all_sample_ts[coarse_info.sample_start_idx + bin];
        const float t_end = (bin + 1 < coarse_info.n_samples)
                              ? d_all_sample_ts[coarse_info.sample_start_idx + bin + 1]
                              : far_plane;
        const float t = t_start + (t_end - t_start) * curand_uniform(&state);
        const float3 pos = origin + t * direction;

        // =================== DEBUG PRINTING START (SECTION 2) ===================
        // This block prints the result of the sampling for each fine sample.
        // if (ray_idx == 0) {
        //     printf("  -> Fine Sample %d: u=%.5f -> placed in coarse interval %d (t_start=%.4f) -> new t=%.4f\n", 
        //            i, u, bin, t_start, t);
        // }
        // =================== DEBUG PRINTING END (SECTION 2) =====================

        const int global_sample_idx = atomicAdd(d_sample_counter, 1);
        if (global_sample_idx >= max_samples) {
            atomicSub(d_sample_counter, 1);
            break;
        }

        if (sample_start_idx == -1) {
            sample_start_idx = global_sample_idx;
        }

        d_all_sample_positions[global_sample_idx] = pos;
        d_ray_indices[global_sample_idx] = ray_idx;
        d_all_sample_ts_output[global_sample_idx] = t;
        samples_for_this_ray++;
    }

    if (samples_for_this_ray > 0) {
        d_fine_ray_info[ray_idx] = {sample_start_idx, samples_for_this_ray};
    }

    d_rng_states[ray_idx] = state;
}


template <int BLOCK_SIZE>
__global__ void generate_fine_samples_kernel_cub_complete(
    int n_rays,
    int n_fine_samples,
    int max_samples,
    // --- Inputs from Coarse Pass ---
    const RayCompactInfo* __restrict__ d_coarse_ray_info,
    const float* __restrict__ d_all_sample_ts,
    const float* __restrict__ d_all_weights,
    // --- General Ray Data ---
    const float* __restrict__ d_far_bounds,
    const float3* __restrict__ d_ray_origins,
    const float3* __restrict__ d_ray_directions,
    curandState* __restrict__ d_rng_states,
    // --- Outputs ---
    int* d_sample_counter,
    float3* d_all_sample_positions,
    float* d_all_sample_ts_output,
    uint32_t* d_ray_indices,
    RayCompactInfo* d_fine_ray_info
) {
    // KERNEL LAUNCH CONFIG: <<<n_rays, BLOCK_SIZE, smem_size>>>
    // where BLOCK_SIZE is n_coarse_samples (and must be a power of 2).

    const int thread_idx = threadIdx.x;
    const int ray_idx = blockIdx.x;

    if (ray_idx >= n_rays) {
        return;
    }

    // Define the CUB BlockScan type for this specific block size.
    typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
    
    // Use a union for shared memory to save space. The sorting array, the CDF array,
    // and CUB's temp storage are never needed at the same time.
    __shared__ union SharedStorage {
        TWeightPair pairs[BLOCK_SIZE];
        float cdf[BLOCK_SIZE + 1]; // +1 for binary search convenience
        typename BlockScan::TempStorage cub_temp_storage;
    } s_mem;

    // --- SETUP AND EARLY EXIT ---
    const RayCompactInfo coarse_info = d_coarse_ray_info[ray_idx];

    if (thread_idx == 0) {
        d_fine_ray_info[ray_idx] = {-1, 0};
    }

    if (coarse_info.n_samples <= 0) {
        return; // All threads in this block exit if there are no coarse samples.
    }

    // --- STEP 1: LOAD AND PAD SAMPLES INTO SHARED MEMORY ---
    if (thread_idx < coarse_info.n_samples) {
        s_mem.pairs[thread_idx] = {
            d_all_sample_ts[coarse_info.sample_start_idx + thread_idx],
            d_all_weights[coarse_info.sample_start_idx + thread_idx]
        };
    } else {
        // Pad extra threads with sentinel values that won't affect the sort.
        s_mem.pairs[thread_idx] = { INFINITY, 0.0f };
    }
    __syncthreads();

    // --- STEP 2: BITONIC SORT IN SHARED MEMORY ---
    for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = thread_idx ^ j;
            if (ixj > thread_idx) {
                if ((thread_idx & k) == 0) {
                    bitonic_sort_step_t_weight(s_mem.pairs, thread_idx, ixj);
                } else {
                    bitonic_sort_step_t_weight(s_mem.pairs, ixj, thread_idx);
                }
            }
            __syncthreads();
        }
    }
    // s_mem.pairs is now sorted by 't'.

    // --- STEP 3: PARALLEL CDF CALCULATION WITH CUB ---
    float thread_weight = (thread_idx < coarse_info.n_samples) ? (s_mem.pairs[thread_idx].weight + 1e-5f) : 0.0f;
    float exclusive_cdf_val;
    float total_weight_sum;

    BlockScan(s_mem.cub_temp_storage).ExclusiveSum(thread_weight, exclusive_cdf_val, total_weight_sum);
    __syncthreads();

    if (total_weight_sum < 1e-5f) {
        return; // Exit if the ray has no meaningful weights.
    }

    // --- STEP 4: PARALLEL NORMALIZATION ---
    // Create the final normalized CDF array in shared memory for the binary search.
    s_mem.cdf[thread_idx] = exclusive_cdf_val / total_weight_sum;
    if (thread_idx == coarse_info.n_samples - 1) {
        // The last valid sample should have a CDF value of 1.0 at the end of its interval.
        // We calculate this from the sum of its own weight plus the preceding ones.
        s_mem.cdf[coarse_info.n_samples] = (exclusive_cdf_val + thread_weight) / total_weight_sum;
    }
    __syncthreads();
    
    // --- STEP 5: PARALLEL FINE SAMPLING ---
    curandState state = d_rng_states[ray_idx];
    __shared__ int fine_samples_start_idx;
    __shared__ int fine_samples_generated_count;
    if (thread_idx == 0) {
        fine_samples_start_idx = -1;
        fine_samples_generated_count = 0;
    }
    __syncthreads();

    // Each thread processes a subset of fine samples using a grid-stride loop.
    for (int i = thread_idx; i < n_fine_samples; i += BLOCK_SIZE) {
        const float u = curand_uniform(&state);
        
        // Perform binary search on the shared CDF array.
        int bin = 0;
        {
            int low = 0;
            int high = coarse_info.n_samples;
            while (low < high) {
                const int mid = low + (high - low) / 2;
                if (u >= s_mem.cdf[mid + 1]) { // Search on the upper bound of the interval
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            bin = min(low, coarse_info.n_samples - 1);
        }

        const float t_start = s_mem.pairs[bin].t;
        const float t_end = (bin + 1 < coarse_info.n_samples) ? s_mem.pairs[bin + 1].t : d_far_bounds[ray_idx];
        const float t = t_start + (t_end - t_start) * curand_uniform(&state);

        const int global_sample_idx = atomicAdd(d_sample_counter, 1);
        if (global_sample_idx >= max_samples) {
            atomicSub(d_sample_counter, 1);
            break; // This thread stops generating samples.
        }

        // Atomically find the minimum start index and count the generated samples for this block.
        atomicMin(&fine_samples_start_idx, global_sample_idx);
        atomicAdd(&fine_samples_generated_count, 1);

        const float3 origin = d_ray_origins[ray_idx];
        const float3 direction = d_ray_directions[ray_idx];
        d_all_sample_positions[global_sample_idx] = origin + t * direction;
        d_ray_indices[global_sample_idx] = ray_idx;
        d_all_sample_ts_output[global_sample_idx] = t;
    }
    
    // --- STEP 6: FINALIZE AND SAVE STATE ---
    d_rng_states[ray_idx] = state;
    __syncthreads();

    if (thread_idx == 0 && fine_samples_generated_count > 0) {
        d_fine_ray_info[ray_idx] = {fine_samples_start_idx, fine_samples_generated_count};
    }
}


/**
 * @brief Step 2 (Forward): Takes all valid sample points and computes their
 * hash grid and spherical harmonics features in a single batch.
 */
__global__ void evaluate_samples_kernel(
    int n_total_samples,
    DeviceHashTableAccessor* table_acc,
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
    const float3 dir_opencv = d_ray_directions[ray_idx];

    // Compute and store hash features
    hash_encode_kernel(pos, table_acc, &d_all_hash_features[sample_idx * MLP::D_in]);

    // The relationship is: x_nerf=x_opencv, y_nerf=-y_opencv, z_nerf=-z_opencv
    float3 dir_nerf = make_float3(dir_opencv.x, -dir_opencv.y, -dir_opencv.z);

    // Normalize the NERF space vector to be a unit vector
    float norm_nerf = rsqrtf(
        dir_nerf.x * dir_nerf.x + dir_nerf.y * dir_nerf.y + dir_nerf.z * dir_nerf.z
    );

    dir_nerf.x *= norm_nerf;
    dir_nerf.y *= norm_nerf;
    dir_nerf.z *= norm_nerf;
    
    // Compute and store spherical harmonics features
    sh_encode(dir_nerf, &d_all_sh_features[sample_idx * SH_COEFS]);
}


/**
 * @brief Step 3 (Forward): Performs volumetric composition.
 * This version uses a **one-block-per-ray** strategy to enable fast parallel sorting
 * of samples using a bitonic sort in shared memory. This is critical for correct
 * dt calculation and stable rendering.
 */
__global__ void composite_rays_kernel(
    int n_rays,
    int max_samples_per_ray,
    const float* __restrict__ d_near_bounds,
    const float* __restrict__ d_far_bounds, 
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_ts, // Now contains 't' values
    float* __restrict__ d_all_sample_dts, // OUTPUT for correct dts
    const __half* __restrict__ d_all_densities, // Activated densities
    const __half* __restrict__ d_all_colors,    // Activated colors
    // OUTPUTS
    float4* d_out_pixels,
    float* d_all_weights, // To be used by backward pass
    float* d_all_alphas   // To be used by backward pass
) {
    const int ray_idx = blockIdx.x;
    if (ray_idx >= n_rays) return;

    
    const RayCompactInfo info = d_ray_info[ray_idx];
    if (info.n_samples <= 0) {
        d_out_pixels[ray_idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    // --- Parallel Sort using Shared Memory ---
    extern __shared__ TValuePair s_data[];

    // Load samples for this ray into shared memory
    if (threadIdx.x < info.n_samples) {
        s_data[threadIdx.x] = {d_all_sample_ts[info.sample_start_idx + threadIdx.x], (int)threadIdx.x};
    }
    __syncthreads();

    // Perform bitonic sort on the samples in shared memory
    for (int k = 2; k <= max_samples_per_ray; k <<= 1) { // k is the size of the sorted sub-arrays
        for (int j = k >> 1; j > 0; j >>= 1) { // j is the distance between elements to compare
            if (threadIdx.x < info.n_samples) {
                int ixj = threadIdx.x ^ j;
                if (ixj > threadIdx.x && ixj < info.n_samples) {
                    if ((threadIdx.x & k) == 0) {
                        bitonic_sort_step(s_data, threadIdx.x, ixj);
                    } else {
                        bitonic_sort_step(s_data, ixj, threadIdx.x);
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Let thread 0 perform the final composition using the sorted data
    if (threadIdx.x == 0) {
        float3 acc_color = make_float3(0.f, 0.f, 0.f);
        float acc_trans = 1.f;

        for (int i = 0; i < info.n_samples; ++i) {
            if (acc_trans < 1e-4f) break;

            const TValuePair current_sample = s_data[i];
            const int original_offset = current_sample.original_idx_offset;
            const int s_idx = info.sample_start_idx + original_offset;

            // --- Correct dt calculation ---
            const float t_curr = current_sample.t;
            const float t_next = (i + 1 < info.n_samples) ? s_data[i + 1].t : d_far_bounds[ray_idx];
            const float dt = t_next - t_curr + 1e-5f;
            d_all_sample_dts[s_idx] = dt; // Store correct dt for backward pass

            const float density = __half2float(d_all_densities[s_idx]);
            const float alpha = 1.f - expf(-density * dt);
            const float weight = acc_trans * alpha;

            const float3 color = make_float3(
                __half2float(d_all_colors[s_idx * 3 + 0]),
                __half2float(d_all_colors[s_idx * 3 + 1]),
                __half2float(d_all_colors[s_idx * 3 + 2])
            );

            acc_color += weight * color;
            acc_trans *= (1.f - alpha);

            d_all_weights[s_idx] = weight;
            d_all_alphas[s_idx] = alpha;
        }
        d_out_pixels[ray_idx] = make_float4(acc_color.x, acc_color.y, acc_color.z, 1.f - acc_trans);
    }
}

/**
 * @brief Step 4 (Backward): Computes initial gradients.
 * Now uses the correctly calculated and stored dt values.
 */
__global__ void compute_backward_initial_grads_kernel(
    int n_rays,
    const float3* __restrict__ dL_d_predicted_rgb,
    const RayCompactInfo* __restrict__ d_ray_info,
    const float* __restrict__ d_all_sample_dts,
    const __half* __restrict__ d_all_colors,
    const float* __restrict__ d_all_weights,
    const float* __restrict__ d_all_alphas,
    // OUTPUTS
    float* dL_d_density,
    float* dL_d_color
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
        const float dt = d_all_sample_dts[s_idx];

        // --- Gradient Calculations ---

        // 1. Gradient w.r.t. color_i (This was already correct)
        float3 dL_d_c = dL_d_pixel * weight;

        // 2. Transmittance T_i is needed for the alpha gradient.
        // We can recover it from the weight and alpha for this sample.
        const float T_i = weight / (alpha + 1e-10f);

        // 3. Gradient w.r.t this sample's weight (dL/dw_i)
        const float dL_d_w = dot(dL_d_pixel, color);

        // 4. Gradient w.r.t alpha_i (THE FIX IS HERE)
        // The formula is: dL/d_alpha = (dL/dw - dL/dT_out) * T_i
        // Your original code incorrectly calculated (dL/dw * T_i - dL/dT_out).
        const float dL_d_alpha = (dL_d_w - dL_d_transmittance) * T_i;

        // 5. Gradient w.r.t density_i (This was already correct)
        const float dL_d_sigma = dL_d_alpha * (1.f - alpha) * dt;

        // 6. Update transmittance gradient for the next iteration (i-1)
        // This line in your original code was already correct.
        dL_d_transmittance = dL_d_transmittance * (1.f - alpha) + dL_d_w * alpha;

        // Store the results
        dL_d_density[s_idx] = dL_d_sigma;
        dL_d_color[s_idx * 3 + 0] = dL_d_c.x;
        dL_d_color[s_idx * 3 + 1] = dL_d_c.y;
        dL_d_color[s_idx * 3 + 2] = dL_d_c.z;
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
    const float* __restrict__ dL_d_hash_features
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_total_samples) return;

    const float3 pos = d_all_sample_positions[sample_idx];
    
    // Convert __half gradient to float for accumulation
    float dL_d_feat_float[MLP::D_in];
    const float* dL_d_feat = &dL_d_hash_features[sample_idx * MLP::D_in];
    // for (int i = 0; i < MLP::D_in; ++i) {
    //     dL_d_feat_float[i] = dL_d_feat[i];
    // }

    // Call the device function from hashing.cuh
    accumulate_gradients_merged_kernel(table_acc, pos, dL_d_feat);
}


__global__ void calculate_loss_kernel(
    const float4* predicted_pixels, 
    const float3* ground_truth_colors, 
    const RayCompactInfo* d_ray_info,
    int n_rays,
    float3* dL_d_predicted_rgb, 
    float* d_per_ray_mse_loss,
    float gradient_scale_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;

    if (d_ray_info[i].n_samples > 0) {
        // MSE Loss
        float3 predicted_rgb = make_float3(predicted_pixels[i].x, predicted_pixels[i].y, predicted_pixels[i].z);
        float3 actual_rgb = ground_truth_colors[i];
        float3 diff = predicted_rgb - actual_rgb;

        dL_d_predicted_rgb[i] = 2.f * diff * gradient_scale_factor;
        d_per_ray_mse_loss[i] = dot(diff, diff);
    } else {
        // No samples for this ray - zero gradient
        dL_d_predicted_rgb[i] = make_float3(0.f, 0.f, 0.f);
        d_per_ray_mse_loss[i] = 0.f;
    }
}




__global__ void merge_ray_info_kernel(
    int n_rays,
    const RayCompactInfo* __restrict__ d_coarse_info,
    const RayCompactInfo* __restrict__ d_fine_info,
    RayCompactInfo* d_final_ray_info // Output
) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    const RayCompactInfo coarse = d_coarse_info[ray_idx];
    const RayCompactInfo fine = d_fine_info[ray_idx];

    // The starting index is always the one from the coarse pass
    // (or the fine pass if no coarse samples existed).
    int final_start_idx = coarse.sample_start_idx;
    if (coarse.n_samples <= 0) {
        final_start_idx = fine.sample_start_idx;
    }

    // The total number of samples is simply the sum.
    int final_n_samples = coarse.n_samples + fine.n_samples;

    d_final_ray_info[ray_idx] = {final_start_idx, final_n_samples};
}
////////////////////////////////////////////////////////////////////////////////
// RENDERER CLASS IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////////////

void Renderer::render_forward(
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
) {
    const int threads = 256;
    dim3 grid_gen_samples((n_rays + threads - 1) / threads);

    // ========================================================================
    // --- PASS 1: COARSE SAMPLING & WEIGHT GENERATION ---
    // ========================================================================

    // === NEW: Create dedicated buffers for coarse and fine ray info ===
    CudaDeviceBuffer<RayCompactInfo> d_coarse_ray_info(n_rays);
    CudaDeviceBuffer<RayCompactInfo> d_fine_ray_info(n_rays);

    CHECK_CUDA_THROW(cudaMemsetAsync(ws.d_sample_counter.get(), 0, sizeof(int), stream));
    
    generate_coarse_samples_kernel<<<grid_gen_samples, threads, 0, stream>>>(
        n_rays, n_coarse_samples, ws.max_samples, d_near_bounds, d_far_bounds, grid,
        d_ray_origins, d_ray_directions, ws.d_rng_states.get(),
        ws.d_sample_counter.get(), ws.d_all_sample_positions.get(), ws.d_all_sample_ts.get(),
        ws.d_ray_indices.get(), d_coarse_ray_info.get()
    );

    int n_coarse_total_samples = 0;
    CHECK_CUDA_THROW(cudaMemcpy(&n_coarse_total_samples, ws.d_sample_counter.get(), sizeof(int), cudaMemcpyDeviceToHost));

    if (n_coarse_total_samples == 0) {
        CHECK_CUDA_THROW(cudaMemsetAsync(d_out_pixels, 0, n_rays * sizeof(float4), stream));
        return;
    }

    // --- Evaluate Coarse Samples ---
    dim3 grid_eval_coarse((n_coarse_total_samples + threads - 1) / threads);
    evaluate_samples_kernel<<<grid_eval_coarse, threads, 0, stream>>>(
        n_coarse_total_samples, table_acc, ws.d_all_sample_positions.get(),
        ws.d_ray_indices.get(), d_ray_directions, n_rays, ws.d_all_hash_features.get(),
        ws.d_all_sh_features.get()
    );
    mlp->forward_cublas(
        handle, stream, n_coarse_total_samples,
        ws.d_all_hash_features.get(), ws.d_all_sh_features.get(), ws.d_mlp_hidden1.get(),
        ws.d_mlp_density_out_full.get(), ws.d_mlp_color_net_input.get(), ws.d_mlp_color_hidden1.get(),
        ws.d_mlp_color_hidden2.get(), ws.d_mlp_rgb_out.get(), ws.d_all_raw_densities.get(),
        ws.d_all_raw_colors.get()
    );

    // --- Composite Coarse & Get Weights ---
    const int max_samples_per_ray_pow2 = 1 << static_cast<int>(ceil(log2(n_coarse_samples + n_fine_samples)));
    dim3 grid_composite(n_rays);
    dim3 block_composite(max_samples_per_ray_pow2);
    size_t smem_size = max_samples_per_ray_pow2 * sizeof(TValuePair);

    composite_rays_kernel<<<grid_composite, block_composite, smem_size, stream>>>(
        n_rays, max_samples_per_ray_pow2, d_near_bounds, d_far_bounds, d_coarse_ray_info.get(),
        ws.d_all_sample_ts.get(), ws.d_all_sample_dts.get(), ws.d_all_raw_densities.get(),
        ws.d_all_raw_colors.get(), d_out_pixels, ws.d_all_weights.get(), ws.d_all_alphas.get()
    );

    
    // ========================================================================
    // --- PASS 2: FINE SAMPLING & FINAL RENDER ---
    // ========================================================================

    const int N_COARSE_SAMPLES = 64;

    // Set up the launch configuration
    dim3 grid_s(n_rays);
    dim3 block_s(N_COARSE_SAMPLES);

    // Calculate the required dynamic shared memory
    // Note: The union means we only need enough memory for the LARGEST member.
    size_t smem_size_w = 0;
    size_t pairs_size = sizeof(TWeightPair) * N_COARSE_SAMPLES;
    size_t cub_storage_size = sizeof(cub::BlockScan<float, N_COARSE_SAMPLES>::TempStorage);
    size_t cdf_size = sizeof(float) * (N_COARSE_SAMPLES + 1);

    // smem_size = max(pairs_size, cub_storage_size); But cdf is also needed after cub.
    // A simpler, safer approach is to not use a union if memory isn't tight
    // and just sum them. Let's assume the union is used correctly.
    smem_size_w = max(pairs_size, cdf_size); // pairs and cdf are the main contenders
    smem_size_w = max(smem_size_w, cub_storage_size);


    // Launch the kernel with the template argument
    generate_fine_samples_kernel_cub_complete<N_COARSE_SAMPLES><<<grid_s, block_s, smem_size, stream>>>(
        n_rays,
        n_fine_samples,
        ws.max_samples,
        d_coarse_ray_info.get(),
        ws.d_all_sample_ts.get(),
        ws.d_all_weights.get(),
        d_far_bounds,
        d_ray_origins,
        d_ray_directions,
        ws.d_rng_states.get(),
        ws.d_sample_counter.get(),
        ws.d_all_sample_positions.get(),
        ws.d_all_sample_ts.get(),
        ws.d_ray_indices.get(),
        d_fine_ray_info.get()
    );

    int n_total_samples = 0;
    CHECK_CUDA_THROW(cudaMemcpy(&n_total_samples, ws.d_sample_counter.get(), sizeof(int), cudaMemcpyDeviceToHost));
    const int n_fine_only_samples = n_total_samples - n_coarse_total_samples;

    // --- Evaluate *a_fine_only_samples* Samples ---
    if (n_fine_only_samples > 0) {
        dim3 grid_eval_fine((n_fine_only_samples + threads - 1) / threads);
        evaluate_samples_kernel<<<grid_eval_fine, threads, 0, stream>>>(
            n_fine_only_samples, table_acc, ws.d_all_sample_positions.get() + n_coarse_total_samples,
            ws.d_ray_indices.get() + n_coarse_total_samples, d_ray_directions, n_rays,
            ws.d_all_hash_features.get() + n_coarse_total_samples * MLP::D_in,
            ws.d_all_sh_features.get() + n_coarse_total_samples * SH_COEFS
        );
        mlp->forward_cublas(
            handle, stream, n_fine_only_samples,
            ws.d_all_hash_features.get() + n_coarse_total_samples * MLP::D_in,
            ws.d_all_sh_features.get() + n_coarse_total_samples * SH_COEFS,
            ws.d_mlp_hidden1.get() + n_coarse_total_samples * MLP::D_hidden,
            ws.d_mlp_density_out_full.get() + n_coarse_total_samples * MLP::D_density_out,
            ws.d_mlp_color_net_input.get() + n_coarse_total_samples * MLP::D_color_in,
            ws.d_mlp_color_hidden1.get() + n_coarse_total_samples * MLP::D_color_hidden,
            ws.d_mlp_color_hidden2.get() + n_coarse_total_samples * MLP::D_color_hidden,
            ws.d_mlp_rgb_out.get() + n_coarse_total_samples * MLP::D_color_out,
            ws.d_all_raw_densities.get() + n_coarse_total_samples,
            ws.d_all_raw_colors.get() + n_coarse_total_samples * 3
        );
    }
    
    merge_ray_info_kernel<<<grid_gen_samples, threads, 0, stream>>>(
        n_rays, 
        d_coarse_ray_info.get(), 
        d_fine_ray_info.get(), 
        ws.d_ray_info.get()
    );

    // --- Final Composite ---
    composite_rays_kernel<<<grid_composite, block_composite, smem_size, stream>>>(
        n_rays, max_samples_per_ray_pow2, d_near_bounds, d_far_bounds, ws.d_ray_info.get(),
        ws.d_all_sample_ts.get(), ws.d_all_sample_dts.get(), ws.d_all_raw_densities.get(),
        ws.d_all_raw_colors.get(), d_out_pixels, ws.d_all_weights.get(), ws.d_all_alphas.get()
    );
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
    mlp->backward_cublas(
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
        n_total_samples, table_acc,
        ws.d_all_sample_positions.get(), ws.dL_d_hash_features.get()
    );
}