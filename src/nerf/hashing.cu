#include "../../include/nerf/hashing.cuh"
#include <curand_kernel.h>
#include "../../include/nerf/occupancy_grid.cuh"




void HashTable::release_optimizer_states() {
    printf("Releasing HashTable optimizer states and gradient buffers...\n");

    // Release the Adam optimizer state buffers
    all_momentum.free();
    all_variance.free();

    // Release the buffers for gradients and other training-only helpers
    all_sum_grads.free();
    all_counts.free();
    all_locks.free();
}



__global__ void initialize_hash_entries_kernel(
    __half2* table, size_t n_elements, 
    unsigned long long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_half_elements = n_elements * 4;

    if (idx >= n_elements) return;


    // Initialize cuRAND state for each thread
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Uniform distribution    
    const float scale = 1e-4f;
    float val1 = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    float val2 = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    

    // Store as half-precision float
    table[idx] = __floats2half2_rn(val1, val2);
}




// =====================================================
//      TV LOSS KERNELS
// =====================================================


__global__ void calculate_density_tv_loss_kernel(
    const float* density_grid,  // 3D density grid
    int3 resolution,
    float* tv_loss
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= resolution.x - 1 || y >= resolution.y - 1 || z >= resolution.z - 1) return;
    
    int idx = x + y * resolution.x + z * resolution.x * resolution.y;
    
    float d_curr = density_grid[idx];
    float d_x = density_grid[idx + 1];
    float d_y = density_grid[idx + resolution.x];
    float d_z = density_grid[idx + resolution.x * resolution.y];
    
    float tv = pow(d_curr - d_x, 2) + pow(d_curr - d_y, 2) + pow(d_curr - d_z, 2);
    atomicAdd(tv_loss, tv);
}

__global__ void calculate_tv_grad_kernel(
    DeviceHashTableAccessor* table,
    float tv_loss_weight,
    float* d_total_tv_loss
) {
    const int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level_idx = blockIdx.y;

    if (level_idx >= table->L)
    {
        return;
    }

    hash_level* level = &(table->d_levels_array[level_idx]);
    const int n_float2_per_entry = F_val / 2;
    const int total_float2_in_level = level->T * n_float2_per_entry;
    
    
    const int idx = entry_idx;
    if (idx >= total_float2_in_level - 1) return;
    
    if (tv_loss_weight > 0.0f) {
        // const float max_tv_grad = 0.1;
        for (int i = 0; i < n_float2_per_entry; i++) {
            // Fetch the current feature vector's values (from the FP32 master parameters)
            float2 p_curr = level->master[idx];
            float2 p_next = level->master[idx + 1];

            float2 diff = p_curr - p_next;

            const float max_tv_grad = 0.1f;
            diff.x = fmaxf(-max_tv_grad, fminf(max_tv_grad, diff.x));
            diff.y = fmaxf(-max_tv_grad, fminf(max_tv_grad, diff.y));
            float loss = diff.x * diff.x + diff.y * diff.y;

            if (loss > 0.f) atomicAdd(d_total_tv_loss, loss);

            const float2 grad_update = make_float2(tv_loss_weight * diff.x, tv_loss_weight * diff.y);

            // Add the gradient of the TV loss to the rendering loss gradient
            atomicAdd(&level->sum_grad[idx].x, grad_update.x);
            atomicAdd(&level->sum_grad[idx].y, grad_update.y);

            // The neighbor gets the opposite gradient, applied atomically to prevent race conditions
            atomicAdd(&level->sum_grad[idx + 1].x, -grad_update.x);
            atomicAdd(&level->sum_grad[idx + 1].y, -grad_update.y);
        }
    }
}



// =====================================================
//      TV LOSS KERNELS
// =====================================================






__global__ void adam_update_kernel(
    DeviceHashTableAccessor* table,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int step,
    float grad_unscaler = 1.0f
) {
    const int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level_idx = blockIdx.y;

    if (level_idx >= table->L) return;
    
    hash_level* level = &(table->d_levels_array[level_idx]);
    
    if (entry_idx >= level->T) return;

    
    //const int F_per_entry = F_val; // let F be the constitent F in hash level
    const int n_float2_per_entry = F_val / 2;

    for (int i = 0; i < n_float2_per_entry; ++i) {
        int buffer_idx = entry_idx * n_float2_per_entry + i;


        float2 grad = level->sum_grad[buffer_idx] / grad_unscaler;


        if (grad.x == 0.0f && grad.y == 0.0f) {
            continue; // Skip if no gradient for this part
        }


        // Update moments
        float2 m_prev = level->momentum[buffer_idx];
        float2 v_prev = level->variance[buffer_idx];
        
        float2 m_new = beta1 * m_prev + (1.f - beta1) * grad;
        float2 v_new = beta2 * v_prev + (1.f - beta2) * (grad * grad);

        level->momentum[buffer_idx] = m_new;
        level->variance[buffer_idx] = v_new;

        // Bias correction
        float t = (float) step + 1.f;
        float m_hat_x = m_new.x / (1.f - powf(beta1, t));
        float m_hat_y = m_new.y / (1.f - powf(beta1, t));
        float v_hat_x = v_new.x / (1.f - powf(beta2, t));
        float v_hat_y = v_new.y / (1.f - powf(beta2, t));


        // Update parameters
        // 1. Get the current master parameters
        float2 p_curr = level->master[buffer_idx];

        // 2. Calculate the update step
        float2 update_step = make_float2(
            lr * m_hat_x / (sqrtf(v_hat_x) + epsilon),
            lr * m_hat_y / (sqrtf(v_hat_y) + epsilon)
        );

        // 3. Apply the update to the master parameters
        p_curr.x -= update_step.x;
        p_curr.y -= update_step.y;
        
        // 4. Write the updated parameters back to BOTH buffers to keep them in sync
        level->master[buffer_idx] = p_curr;
        level->entries[buffer_idx] = __floats2half2_rn(p_curr.x, p_curr.y);

    }
}


__global__ void zero_hash_gradients_kernel(DeviceHashTableAccessor* table) {
    const int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int level_idx = blockIdx.y;
    if (level_idx >= table->L) return;
    hash_level* level = &(table->d_levels_array[level_idx]);
    if (entry_idx >= level->T) return;

    const int n_float2_per_entry = F_val / 2;
    for (int i = 0; i < n_float2_per_entry; ++i) {
        int buffer_idx = entry_idx * n_float2_per_entry + i;
        level->sum_grad[buffer_idx] = make_float2(0.0f, 0.0f);
    }
}


__global__ void init_gradients_kernel(
    float2* d_sum_grad,
    int n_elements, 
    float val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread 0 attempts the write
    if (idx == 0) {
        printf("Test InitKernel[0]: Trying to write sum_grad {%f, %f} to address %p at index %d (n_elements=%d)\n",
               val, val, &d_sum_grad[idx], idx, n_elements);
    }
    if (idx < n_elements) {
        d_sum_grad[idx] = make_float2(val, val);
    }
};


__global__ void zero_grad_kernel(float* grad, int n_elements) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n_elements) {
        grad[i] = 0.0f;
    }
}




void HashTable::adam_update(
    float lr, float beta1, float beta2, float epsilon,
     int step, float tv_loss_weight, 
     const OccupancyGrid* grid,
     float* d_total_tv_loss,
     cudaStream_t stream,
     int grad_accumulation_steps
    ) {

    if (tv_loss_weight > 0.0f && grid != nullptr ){
        // zero out the buffer
        CHECK_CUDA_THROW(cudaMemsetAsync(d_total_tv_loss, 0, sizeof(float), stream));

        dim3 block_dim_tv(8, 8, 4);
        dim3 grid_dim_tv(
            (grid->resolution.x + block_dim_tv.x - 1) / block_dim_tv.x,
            (grid->resolution.y + block_dim_tv.y - 1) / block_dim_tv.y,
            (grid->resolution.z + block_dim_tv.z - 1) / block_dim_tv.z
        );
        calculate_density_tv_loss_kernel<<<grid_dim_tv, block_dim_tv, 0, stream>>>(
            grid->density_grid_data_host(),
            grid->resolution,
            d_total_tv_loss
        );
        CHECK_CUDA_THROW(cudaGetLastError());
        // CHECK_CUDA_THROW(cudaDeviceSynchronize());
    }

    dim3 block_dim_adam(256);
    dim3 grid_dim_adam(
        (this->T_per_level + block_dim_adam.x - 1) / block_dim_adam.x, 
        this->L
    ); 
    float grad_unscaler = (float) grad_accumulation_steps;

    adam_update_kernel<<< grid_dim_adam, block_dim_adam, 0, stream >>>(
        this->get_device_accessor(),
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        grad_unscaler
    );
    CHECK_CUDA_THROW(cudaGetLastError());
    // CHECK_CUDA_THROW(cudaDeviceSynchronize());

    // CHECK_CUDA_THROW(cudaStreamSynchronize(stream));
}


void HashTable::zero_grad(cudaStream_t stream) {
    dim3 block_dim_adam(256);
    dim3 grid_dim_adam(
        (this->T_per_level + block_dim_adam.x - 1) / block_dim_adam.x, 
        this->L
    );
    zero_hash_gradients_kernel<<<grid_dim_adam, block_dim_adam, 0, stream>>>(this->get_device_accessor());
    CHECK_CUDA_THROW(cudaGetLastError());
}


__global__ void tv_loss_kernel(
    const half2* __restrict__ grid, // The half-precision feature grid, as half2
    int T,                          // Number of entries in this grid (e.g., 524288)
    int F,                          // Number of features per entry (e.g. 4)
    float* __restrict__ d_per_level_loss // Output buffer
) {
    const int idx = threadIdx.x + blockIdx.x * blockIdx.x;
    if (idx >= T) return;

    // Since F=4, we have two half2 elements per grid location.
    const int n_half2_features = F / 2;

    float loss = 0.0f;
    for (int f_idx = 0; f_idx < n_half2_features; ++f_idx) {
        half2 val_curr = grid[idx * n_half2_features + f_idx];
        
        // Penalize difference with next element (if not at the edge)
        if (idx < T - 1) {
            half2 val_next = grid[(idx + 1) * n_half2_features + f_idx];

            // Calculate squared difference for both components of the half2 vector
            float diff_x = __half2float(val_curr.x) - __half2float(val_next.x);
            float diff_y = __half2float(val_curr.y) - __half2float(val_next.y);
            loss += diff_x * diff_x + diff_y * diff_y;
        }
    }
    
    if (loss > 0.0f) {
        atomicAdd(d_per_level_loss, loss);
    }
}





__global__ void resize_hash_table_kernel(
    hash_level* __restrict__ level,
    int new_size
) {

}