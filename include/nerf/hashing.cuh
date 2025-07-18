#pragma once

#define HASHING
#ifdef  HASHING


// UTILS libs
#include "nerf_config.cuh"
#include "../../UTIL/cpu_util.h"
#include "../../UTIL/err_handle.h"
#include "../common/cuda_wrappers.h"
#include "../common/math_utils.h"
#include "mlp.cuh"

// C/C++ libs
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

// CUDA libs
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <curand_kernel.h> // random values generation in parallel




/////////////////////////////////////////////////////////////////////////
// ------------- Hashing ------------- //
/////////////////////////////////////////////////////////////////////////



class OccupancyGrid;


//const unsigned long PRIMES[3] = {1, 2654435761, 805459861};

struct hash_level {
    int T;  // Number of entries in the hash table
    int F;  // Feature dimension
    int N_1;  // Resolution at this level
    unsigned long primes[3];  // Prime numbers for hashing
    __half2* entries;  // Hash table entries (FP16)
    float2* master;   // Master copy of parameters (FP32)
    float2* sum_grad; // Sum of gradients
    float2* momentum; // 1st moment estimate
    float2* variance; // 2nd moment estimate
    int* count;       // Count of updates
    int* locks;       // For collision resolution

    // Default constructor
    hash_level() : T(0), F(0), N_1(0) {
        primes[0] = 1;
        primes[1] = 2654435761UL;
        primes[2] = 805459861UL;
    }

    // Parameterized constructor
    hash_level(int t_val, int f_val) : T(t_val), F(f_val), N_1(0) {
        primes[0] = 1;
        primes[1] = 2654435761UL;
        primes[2] = 805459861UL;

    }

    // Rule of Five:
    hash_level(const hash_level&) = delete;
    hash_level& operator=(const hash_level&) = delete;
    hash_level(hash_level&& other) noexcept = default; // Relies on Cuda...Buffer move ops
    hash_level& operator=(hash_level&& other) noexcept = default; // Relies on Cuda...Buffer move ops
};



typedef struct DeviceHashTableAccessor {
    hash_level* d_levels_array; // <<< This will point to an ARRAY of hash_level structs ON THE DEVICE
    int L;
    int N_min; // If needed by kernels
    int N_max; // If needed by kernels
    int T_per_level; // If needed by kernels
    // Add any other global table parameters kernels might need
} DeviceHashTableAccessor;




__global__ void initialize_hash_entries_kernel(
    __half2* entries, size_t n_elements, 
    unsigned long long seed
);


class HashTable {

public:
    std::vector<hash_level> levels; // array of L hash levels
    int L;              // Number of levels (default 16)
    int N_min;          // Coarsest resolution
    int N_max;          // Finest resolution
    int T_per_level;
    std::vector<int> N_l_values;

    CudaDeviceBuffer<__half2> all_entries;
    CudaDeviceBuffer<float2> all_master_params;
    CudaManagedBuffer<float2> all_sum_grads;
    CudaManagedBuffer<float2> all_momentum;
    CudaManagedBuffer<float2> all_variance;
    CudaManagedBuffer<int> all_counts;
    CudaManagedBuffer<int> all_locks;


private:
    // RAII wrappers for the device-side representation
    mutable CudaDeviceBuffer<hash_level> d_levels_array_buffer_;
    mutable CudaDeviceBuffer<DeviceHashTableAccessor> d_accessor_buffer_;
    mutable bool device_representation_synced_;


public:
    // Constructor: move init hash table here
    HashTable(int nlevels, int n_min_val, int n_max_val, int t_val_per_level) :
        L(nlevels),
        N_min(n_min_val),
        N_max(n_max_val),
        T_per_level(t_val_per_level),
        device_representation_synced_(false)
    {
        // Conditions Check
        THROW_IF_ERROR_CONDITION(L <= 0, std::invalid_argument, "L must be > 0");
        THROW_IF_ERROR_CONDITION(N_min <= 0, std::invalid_argument, "N_min must be > 0");

        printf("Constructing HashTable L=%d, N_min=%d, N_max=%d, T_per_level=%d\n", L, N_min, N_max, T_per_level);

        // Calculate total size needed for all levels, assuming F=4
        const int F = F_val;
        size_t elements_per_level_float2 = (size_t)T_per_level * (F / 2);
        size_t total_float2_elements = (size_t)L * elements_per_level_float2;
        
        size_t elements_per_level_half2 = (size_t)T_per_level * (F / 2); // F/2 because a __half2 holds 2 halfs
        size_t total_half2_elements = (size_t)L * elements_per_level_half2;
        
        
        size_t total_int_elements = (size_t)L * T_per_level;


        // Allocate one large, contiguous buffer for each parameter type
        all_entries       = CudaDeviceBuffer<__half2>(total_half2_elements);        
        all_master_params = CudaDeviceBuffer<float2>(total_float2_elements);
        all_sum_grads     = CudaManagedBuffer<float2>(total_float2_elements);
        all_momentum      = CudaManagedBuffer<float2>(total_float2_elements);
        all_variance      = CudaManagedBuffer<float2>(total_float2_elements);
        all_counts        = CudaManagedBuffer<int>(total_int_elements);
        all_locks         = CudaManagedBuffer<int>(total_int_elements);

        // Initialize all allocated memory to zero to be safe
        const int block_size = 256;
        const int grid_size = (all_entries.size() + block_size - 1) / block_size;
        unsigned long long seed = 1337ULL;
        initialize_hash_entries_kernel<<<grid_size, block_size>>>(all_entries.get(), all_entries.size(), seed);
        CHECK_CUDA_THROW(cudaDeviceSynchronize());

       
        // CHECK_CUDA_THROW(cudaMemset(all_entries.get(), 0, all_entries.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_master_params.get(), 0, all_master_params.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_sum_grads.get(), 0, all_sum_grads.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_momentum.get(), 0, all_momentum.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_variance.get(), 0, all_variance.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_counts.get(), 0, all_counts.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(all_locks.get(), 0, all_locks.nbytes()));


        // ====================================================================
        // === ADD THIS NEW BLOCK TO PREFETCH MANAGED MEMORY TO THE GPU      ===
        // ====================================================================
        printf("  Prefetching managed optimizer states to GPU...\n");
        int device = -1;
        CHECK_CUDA_THROW(cudaGetDevice(&device));

        // Prefetch all managed buffers to the current GPU device.
        // This moves the memory physically to the GPU before any kernels run,
        // ensuring it's ready for device-side atomic operations.
        CHECK_CUDA_THROW(cudaMemPrefetchAsync(all_sum_grads.get(), all_sum_grads.nbytes(), device, 0));
        CHECK_CUDA_THROW(cudaMemPrefetchAsync(all_momentum.get(), all_momentum.nbytes(), device, 0));
        CHECK_CUDA_THROW(cudaMemPrefetchAsync(all_variance.get(), all_variance.nbytes(), device, 0));
        CHECK_CUDA_THROW(cudaMemPrefetchAsync(all_counts.get(), all_counts.nbytes(), device, 0));
        CHECK_CUDA_THROW(cudaMemPrefetchAsync(all_locks.get(), all_locks.nbytes(), device, 0));
        
        // It's good practice to wait for the prefetch to complete.
        CHECK_CUDA_THROW(cudaStreamSynchronize(0));


        // Calculate and store resolution N_l for each level
        N_l_values.resize(L);
        double b = (L > 1) ? std::exp((std::log((double)N_max) - std::log((double)N_min)) / (L - 1)) : 1.0;
        for (int l = 0; l < L; ++l) {
            N_l_values[l] = static_cast<int>(std::floor(N_min * std::pow(b, (double)l)));
            printf("  HashTable: Level %d configured with N_l=%d\n", l, N_l_values[l]);
        }
    };




    // Method to prepare and get the device-side accessor pointer
    // mad const, but modifies mutable members for caching behavior
    DeviceHashTableAccessor* get_device_accessor() const {
        if (!device_representation_synced_) {
            printf("HashTable: Device representation is not synced. Preparing device data...\n");

            std::vector<hash_level> host_levels_pod(L);
            const size_t elements_per_level_float2 = (size_t)T_per_level * (F_val / 2);
            const size_t elements_per_level_half2 = (size_t)T_per_level * (F_val / 2);
            const size_t elements_per_level_int = (size_t)T_per_level;

            for (int i = 0; i < L; ++i) {
                host_levels_pod[i].T = T_per_level;
                host_levels_pod[i].F = 4;
                host_levels_pod[i].N_1 = N_l_values[i];
                host_levels_pod[i].primes[0] = 1;
                host_levels_pod[i].primes[1] = 2654435761UL;
                host_levels_pod[i].primes[2] = 805459861UL;


                // --- ADD THIS DEBUG PRINT HERE ---
                // This will print the value of T for every single level before it's copied.
                printf("DEBUG HOST-SIDE: Configuring level %d with T = %u\n", i, host_levels_pod[i].T);


                // Set the raw device pointers by offsetting into the large, consolidated buffers
                host_levels_pod[i].entries   = all_entries.get() + i * elements_per_level_half2;
                host_levels_pod[i].master    = all_master_params.get() + i * elements_per_level_float2;
                host_levels_pod[i].sum_grad  = all_sum_grads.get() + i * elements_per_level_float2;
                host_levels_pod[i].momentum  = all_momentum.get() + i * elements_per_level_float2;
                host_levels_pod[i].variance  = all_variance.get() + i * elements_per_level_float2;
                host_levels_pod[i].count     = all_counts.get() + i * elements_per_level_int;
                host_levels_pod[i].locks     = all_locks.get() + i * elements_per_level_int;
            }

            d_levels_array_buffer_ = CudaDeviceBuffer<hash_level>(L);

            if (L > 0) {
                CHECK_CUDA_THROW(cudaMemcpy(
                    d_levels_array_buffer_.get(),
                    host_levels_pod.data(),
                    d_levels_array_buffer_.nbytes(),
                    cudaMemcpyHostToDevice
                ));
            }

            DeviceHashTableAccessor host_accessor_st;
            host_accessor_st.L = L;
            host_accessor_st.N_min = N_min;
            host_accessor_st.N_max = N_max;
            host_accessor_st.T_per_level = T_per_level;
            host_accessor_st.d_levels_array = d_levels_array_buffer_.get();

            d_accessor_buffer_ = CudaDeviceBuffer<DeviceHashTableAccessor>(1);
            CHECK_CUDA_THROW(cudaMemcpy(
                d_accessor_buffer_.get(),
                &host_accessor_st,
                sizeof(DeviceHashTableAccessor),
                cudaMemcpyHostToDevice
            ));
            printf("  HashTable: Device data prepared successfully.\n");

            device_representation_synced_ = true;
        }
        return d_accessor_buffer_.get();
    };

    // Destructor
    ~HashTable() {
        printf("HashTable Destructor: Cleaning up %d levels. std::vector will handle individual hash_level destructors.\n", L);
        // No explicit delete[] needed for levels_vec.
        // std::vector's destructor calls ~hash_level() for each element.
        // ~hash_level() calls destructors of Cuda...Buffer members.
        // ~Cuda...Buffer() calls cudaFree().
    };


    void adam_update(
        float lr, 
        float beta1, 
        float beta2, 
        float epsilon, 
        int step, 
        float tv_loss_weight,
        const OccupancyGrid* grid, 
        float* d_total_tv_loss,
        cudaStream_t stream,
        int grad_accumulation_steps
    );

    void zero_grad(cudaStream_t stream);

    // Calculates the total variation loss across all feature grids
    void tv_loss(float* d_loss_output, cudaStream_t stream = 0);



    // Rule of Five for HashTable:
    // For simplicity in this step, we'll delete copy operations.
    // If you need to copy/move HashTables, you'd implement these carefully.
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    // Move operations can be defaulted if std::vector's move is sufficient.
    HashTable(HashTable&& other) noexcept = default;
    HashTable& operator=(HashTable&& other) noexcept = default;

    // Accessor methods
    hash_level& get_level(int index) {
        THROW_IF_ERROR_CONDITION(index < 0 || index >= L,
                                 std::out_of_range,
                                 "HashTable Error: Level index %d out of bounds [0, %d).", index, L);
        return levels[index];
    }

    const hash_level& get_level(int index) const {
        THROW_IF_ERROR_CONDITION(index < 0 || index >= L,
                                 std::out_of_range,
                                 "HashTable Error: Level index %d out of bounds [0, %d).", index, L);
        return levels[index];
    }

    int n_levels() const { return L; }

    // --- MULTI-GPU: Add accessors for gradient synchronization ---
    float2* gradients() {
        return all_sum_grads.get();
    }

    size_t gradients_count_float2() const {
        return all_sum_grads.size();
    }

    size_t gradients_count_floats() const {
        return all_sum_grads.size() * 2;
    }
    // --- End of new methods ---

    void release_optimizer_states();
};


////////////////////// Device Kernels //////////////////////
__device__ __forceinline__ unsigned int spatial_hash(
    int3 grid_coord, 
    const hash_level* level,
    const int N,
    const int level_idx
) {
    /** 
     * Computes a spatial hash index for a 3D grid coordinate.
     * 
     * @param grid_coord Integer 3D coordinates of a voxel corner, typically computed as floor(scaled_xyz).
     * @param T Size of the hash table (number of entries per level).
     * @param primes[3] Large prime numbers used to decorrelate spatial dimensions.
     * 
     * @return A hash index in [0, T-1] mapping the 3D coordinate to a hash table entry.
     * 
     * @note 
     * - Must be called from device code (GPU).
     * - Uses prime number multiplication and XOR to minimize hash collisions.
     * - Does NOT include collision resolution in this base implementation.
     * 
     * Hash Formula:
     * h(x,y,z) = (x ⊕ (y * π₂) ⊕ (z * π₃)) % T
     * Where π₁, π₂, π₃ are large primes (~10⁹ range).
     */
    // First check: Is the 'level' pointer itself valid?
    
    

    
    if (level == nullptr) {
        printf("spatial_hash ERROR: level pointer is NULL! coord=(%d,%d,%d). ThreadIdx=(%d,%d,%d), BlockIdx=(%d,%d,%d)\n",
               grid_coord.x, grid_coord.y, grid_coord.z,
               threadIdx.x, threadIdx.y, threadIdx.z,
               blockIdx.x, blockIdx.y, blockIdx.z);
        return 0; // Return a "safe" index, though the test will likely still show issues
    }

    
    // convert each coordinate to 64-bit
    long long x = (long long)grid_coord.x;
    long long y = (long long)grid_coord.y;
    long long z = (long long)grid_coord.z;

    if (level->T == 0) {
        printf("FATAL ERROR in spatial_hash: level->T is ZERO! level_idx=%d, coord=(%d,%d,%d), ThreadIdx=%d, BlockIdx=%d\n",
               level_idx, 
               grid_coord.x, grid_coord.y, grid_coord.z,
               threadIdx.x, blockIdx.x);
        return 0; // Return a safe index to prevent the hang
    }
    
    // multiply by large primes as 64-bit
    unsigned long long hx = (unsigned long long)(x) * level->primes[0];
    unsigned long long hy = (unsigned long long)(y) * level->primes[1];
    unsigned long long hz = (unsigned long long)(z) * level->primes[2];

    unsigned long long hval = hx ^ hy ^ hz;

    return (unsigned int)(hval % level->T); // half precision
}



__device__ inline void init_entries_master_weight(
    hash_level* level,
    float scale,
    unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int n_float2_elements = level->T * level->F / 2;
    if (idx < n_float2_elements) {
        // init random state per thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random values from (-scale, scale)
        float val_x = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
        float val_y = (curand_uniform(&state) * 2.0f - 1.0f) * scale;

        level->entries[idx] = __floats2half2_rn(val_x, val_y);
        level->master[idx]  = make_float2(val_x, val_y);
    }
}






static constexpr int D_in = N_LEVELS * F_val;

__device__ inline void hash_encode_kernel(
    const float3& sample_pos, // [-1.5, 1.5] Coords
    const DeviceHashTableAccessor* table_acc,
    __half* output_features
) {
    for (int level_idx = 0; level_idx < N_LEVELS; ++level_idx) {
        const hash_level* level = &(table_acc->d_levels_array[level_idx]);
        const float3 scaled_pos = sample_pos * level->N_1;

        
        const float3 floor_coord = make_float3(floorf(scaled_pos.x), floorf(scaled_pos.y), floorf(scaled_pos.z));
        const float3 weights = make_float3(scaled_pos.x - floor_coord.x, scaled_pos.y - floor_coord.y, scaled_pos.z - floor_coord.z);

        float level_features[F_val] = {0.0f};

        for (int i = 0; i < 8; ++i) {
            const int c_i = i & 1;
            const int c_j = (i >> 1) & 1;
            const int c_k = (i >> 2) & 1;

            const float w = (c_i ? weights.x : 1.f - weights.x) *
                            (c_j ? weights.y : 1.f - weights.y) *
                            (c_k ? weights.z : 1.f - weights.z);

            const int3 corner = make_int3((int)floor_coord.x + c_i, (int)floor_coord.y + c_j, (int)floor_coord.z + c_k);
            const unsigned int hash_idx = spatial_hash(corner, level, (unsigned int)level->N_1, level_idx);

            // --- ADD THIS DEBUGGING CHECK ---
            // The table stores pairs of __half2, so the max hash_idx must be less than half the table size.
            // if (hash_idx * 2 >= level->T) { 
            //     // This printf will only work on the device if you compile with --ptxas-options=-v
            //     // But even without it, adding a guard will stop the crash if this is the issue.
            //     printf("FATAL OOB in hash_encode_kernel: level=%d hash_idx=%u table_size=%u\n", level_idx, hash_idx, level->T);
            //     // Set features to zero and exit to prevent crash
            //     for(int f=0; f < F_val; ++f) {
            //        level_features[f] = 0.0f;
            //     }
            //     continue; // Skip to the next corner
            // }
            // --- END OF CHECK ---

            const int n_half2_features = F_val / 2;
        
            const __half2 entry1 = level->entries[hash_idx * n_half2_features + 0];
            const __half2 entry2 = level->entries[hash_idx * n_half2_features + 1];

            level_features[0] += __half2float(entry1.x) * w;
            level_features[1] += __half2float(entry1.y) * w;
            level_features[2] += __half2float(entry2.x) * w;
            level_features[3] += __half2float(entry2.y) * w;
        }

        for(int f=0; f < F_val; ++f) {
            output_features[level_idx * F_val + f] = __float2half(level_features[f]);
        }
    }
};


__global__ void calculate_density_tv_loss_kernel(
    const float* density_grid,  // 3D density grid
    int3 resolution,
    float* tv_loss
);




__global__ void adam_update_kernel(
    DeviceHashTableAccessor* table,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int step
);

__global__ void calculate_tv_grad_kernel(
    DeviceHashTableAccessor* table,
    float tv_loss_weight,
    float* d_total_tv_loss
);

__global__ void init_gradients_kernel(
    float2* d_sum_grad,
    int n_elements, 
    float val
);

__global__ void zero_grad_kernel(
    float* grad, 
    int n_elements
);

__global__ void resize_hash_table_kernel(
    hash_level* __restrict__ level,
    int new_size
);
////////////////////// Device Kernels //////////////////////



////////////////////// Host Kernels //////////////////////


// ====================================================================
//  Step 1: Host Allocation (Inside init_hash_table)
// ====================================================================
// 
// [ HOST CPU MEMORY (RAM) ]                                     [ GPU DEVICE MEMORY ]
// -------------------------                                     ---------------------
// 
// 1. hash_table* table = malloc(sizeof(hash_table))
//    +-----------------+
//    | hash_table      | <---- 'table' pointer (Host address, e.g., 0x555...)
//    |-----------------|
//    | L               | (int, e.g., 16)
//    | N_min           | (int)
//    | N_max           | (int)
//    | levels          | (hash_level* pointer, initially garbage)
//    +-----------------+
// 
// 2. table->levels = malloc(L * sizeof(hash_level))
//    +-----------------+
//    | hash_table      |
//    |-----------------|
//    | L               |
//    | N_min           |
//    | N_max           |
//    | levels          | ---+
//    +-----------------+    |
//                         |
//                         V
//                       +-------------------+ <---- table->levels pointer (Host address, e.g., 0x555...)
//                       | hash_level[0]     | (sizeof(hash_level) bytes on HOST)
//                       |-------------------|
//                       | hash_level[1]     | (sizeof(hash_level) bytes on HOST)
//                       |-------------------|
//                       | ...               | (...)
//                       |-------------------|
//                       | hash_level[L-1]   | (sizeof(hash_level) bytes on HOST)
//                       +-------------------+
// 
// ====================================================================
//  Step 2: Per-Level GPU Allocation & Host Pointer Storage
//          (Inside loop in init_hash_table calling init_hash_level)
// ====================================================================
// 
// [ HOST CPU MEMORY (RAM) ]                                     [ GPU DEVICE MEMORY ]
// -------------------------                                     ---------------------
//                                                               (Allocated by cudaMalloc/cudaMallocManaged)
// 
//                       +-------------------+
//  For l = 0:           | hash_level[0]     | <--- Address passed to init_hash_level
//                       |-------------------|
//                       | entries  (ptr) ---|---------------------> [ GPU Buffer: FP16 Entries (Size T*F*half) ]
//                       | master   (ptr) ---|---------------------> [ GPU Buffer: FP32 Master (Size T*F*float) ]
//                       | sum_grad (ptr) ---|---------------------> [ GPU Buffer: float2 Gradients (Size T*float2) ] (Managed)
//                       | momentum (ptr) ---|---------------------> [ GPU Buffer: float2 Momentum (Size T*float2) ] (Managed)
//                       | variance (ptr) ---|---------------------> [ GPU Buffer: float2 Variance (Size T*float2) ] (Managed)
//                       | locks    (ptr) ---|---------------------> [ GPU Buffer: int Locks (Size T*int) ] (Managed)
//                       | T        (int)    | (e.g., 16384) <-- Set by init_hash_level
//                       | F        (int)    | (e.g., 2)     <-- Set by init_hash_level
//                       | N_1      (int)    | (e.g., 16)    <-- Set by init_hash_level
//                       | primes   (arr)    |               <-- Set by init_hash_level
//                       |-------------------|
//  For l = 1:           | hash_level[1]     | <--- Address passed to init_hash_level
//                       |-------------------|
//                       | entries  (ptr) ---|---------------------> [ GPU Buffer: FP16 Entries (Size T*F*half) ] (Separate Allocation)
//                       | master   (ptr) ---|---------------------> [ GPU Buffer: FP32 Master (Size T*F*float) ] (Separate Allocation)
//                       | sum_grad (ptr) ---|---------------------> [ GPU Buffer: float2 Gradients (Size T*float2) ] (Managed, Separate)
//                       | ... etc ...       | (Points to *different* GPU buffers than level 0)
//                       | T        (int)    | (e.g., 16384)
//                       | F        (int)    | (e.g., 2)
//                       | N_1      (int)    | (e.g., 22)
//                       |-------------------|
//                       | ...               | (Repeated for all L levels)
//                       |-------------------|
//                       | hash_level[L-1]   |
//                       +-------------------+
// 
// ====================================================================
//  Step 3: Usage (e.g., in Kernels or Host Code)
// ====================================================================
// 
// * **Host Code (`test_adam_update`)**:
//     * Accesses `table->levels[l]` (the host struct).
//     * Reads `table->levels[l].T` (host value).
//     * Gets `table->levels[l].sum_grad` (the *value* of the pointer, which is a GPU address) and passes it to `init_gradients_kernel`.
//     * Passes `table` (host pointer) to `adam_update_kernel`.
//     * Gets `table->levels[l].master` (GPU address) and passes it to `cudaMemcpy` DtoH.
// 
// * **Device Code (`adam_update_kernel`)**:
//     * Receives `table` pointer (points to host struct, requires Unified Memory or specific handling if not managed/mapped). **Correction:** Often, the relevant *device pointers* from the table are passed directly, or the table struct is copied to device memory if small enough. Assuming Unified Memory or appropriate handling here based on your code running. Let's assume the kernel can access `table->levels[level_idx]`.
//     * Reads `level = table->levels[level_idx]` (makes a local copy of the struct, including the GPU pointer values).
//     * Uses `level.sum_grad` (the GPU address) to read gradient data from GPU memory.
//     * Uses `level.master` (the GPU address) to read/write master parameters in GPU memory.
//     * Uses `level.T`, `level.N_1` (scalar values copied into the kernel's local state).



__device__ __host__ inline int gcd(int a, int b) {
    /**
     * Greatest Common Devisor
     */
    while(b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

inline void validate_primes(HashTable* table) {
    for(int l=0; l<table->L; l++) {
        // Check pairwise coprime
        assert(gcd(table->levels[l].primes[1], table->levels[l].primes[2]) == 1);
    }
}


static __device__ void accumulate_gradients_merged_kernel(
    DeviceHashTableAccessor* table,
    const float3& sample_pos,          // Input the sample position directly
    const float* __restrict__ dL_d_feat // This is the gradient dL/dy
) {
    const int N_LEVELS = table->L;

    // A single thread now loops over all feature dimensions.
    // It no longer gets its index from threadIdx.x.
    for (int feat_idx = 0; feat_idx < N_LEVELS * F_val; ++feat_idx) {
        
        int level_idx = feat_idx / F_val;
        int feature_dim_in_level = feat_idx % F_val;

        hash_level* level = &table->d_levels_array[level_idx];
        const float grad_component = dL_d_feat[feat_idx];

        // If the gradient for this feature is zero, we can skip the expensive calculations.
        if (grad_component == 0.0f) {
            continue;
        }

        // Calculate resolution for this level
        float b = expf((logf(table->N_max) - logf(table->N_min)) / (N_LEVELS > 1 ? N_LEVELS - 1 : 1));
        float N = floorf(table->N_min * powf(b, level_idx));

        // Calculate grid position and fractions
        float3 pos_grid = sample_pos * N;
        int3 p0 = make_int3(floorf(pos_grid.x), floorf(pos_grid.y), floorf(pos_grid.z));
        float3 fracts = make_float3(pos_grid.x - p0.x, pos_grid.y - p0.y, pos_grid.z - p0.z);

        // Loop over the 8 corners of the voxel
        for (int corner_idx = 0; corner_idx < 8; ++corner_idx) {
            int3 corner_offset = make_int3((corner_idx & 1), (corner_idx & 2) >> 1, (corner_idx & 4) >> 2);
            int3 corner_pos = make_int3(p0.x + corner_offset.x, p0.y + corner_offset.y, p0.z + corner_offset.z);
            
            unsigned int hash_idx = spatial_hash(corner_pos, level, N, level_idx);
            
            float weight = (corner_offset.x ? fracts.x : 1.f - fracts.x) *
                           (corner_offset.y ? fracts.y : 1.f - fracts.y) *
                           (corner_offset.z ? fracts.z : 1.f - fracts.z);

            int buffer_base_idx = hash_idx * 2;
            float* target_grad;

            // This branching is a bit slow. A different data layout for sum_grad could optimize this.
            // But for correctness, this is fine.
            if (feature_dim_in_level == 0) {
                target_grad = &level->sum_grad[buffer_base_idx + 0].x;
            } else if (feature_dim_in_level == 1) {
                target_grad = &level->sum_grad[buffer_base_idx + 0].y;
            } else if (feature_dim_in_level == 2) {
                target_grad = &level->sum_grad[buffer_base_idx + 1].x;
            } else { // feature_dim_in_level == 3
                target_grad = &level->sum_grad[buffer_base_idx + 1].y;
            }
            
            atomicAdd(target_grad, grad_component * weight);
        }
    }
}





////////////////////// Host Kernels //////////////////////

/////////////////////////////////////////////////////////////////////////
// ------------- Hashing ------------- //
/////////////////////////////////////////////////////////////////////////












/////////////////////////////////////////////////////////////////////////
// ------------- Input Encoding Buffers ------------- //
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// ------------- Input Encoding Buffers ------------- //
/////////////////////////////////////////////////////////////////////////











/////////////////////////////////////////////////////////////////////////
// ------------- Gradient Accumulation Buffers ------------- //
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// ------------- Gradient Accumulation Buffers ------------- //
/////////////////////////////////////////////////////////////////////////








/////////////////////////////////////////////////////////////////////////
// ------------- Training Metadata ------------- //
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// ------------- Training Metadata ------------- //
/////////////////////////////////////////////////////////////////////////



#endif