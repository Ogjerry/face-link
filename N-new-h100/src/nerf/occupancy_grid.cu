
// The .cu file needs the full definitions to call functions from these classes.
#include "../../include/nerf/occupancy_grid.cuh"
#include "../../include/nerf/hashing.cuh"
#include "../../include/nerf/mlp.cuh"
#include "../../include/common/math_utils.h"


__global__ void prepare_grid_mlp_inputs_kernel(
    int n_voxels,
    const OccupancyGrid* grid,
    const DeviceHashTableAccessor* table_acc,
    __half* d_batched_hash_features
) {
    const int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= n_voxels) return;

    // per-voxel position
    int3 grid_coords = make_int3(
        voxel_idx % grid->resolution.x,
        (voxel_idx / grid->resolution.x) % grid->resolution.y,
        voxel_idx / (grid->resolution.x * grid->resolution.y) 
    );
    float3 sample_pos = grid->world_pos_from_grid_coords(grid_coords);

    // feature encode && write to large batch buffer
    hash_encode_kernel(
        sample_pos,
        table_acc,
        d_batched_hash_features + voxel_idx * MLP::D_in
    );
}

__global__ void update_grid_from_densities_kernel(
    int n_voxels,
    OccupancyGrid* grid,
    const __half* d_batched_densities
) {
    const int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= n_voxels) return;

    float density = __half2float(d_batched_densities[voxel_idx]);

    float* density_grid_ptr = &grid->density_grid_data()[voxel_idx];
    float old_density_val = *density_grid_ptr;
    float new_density = fmaxf(0.f, grid->decay * old_density_val + (1.f - grid->decay) * density);
    *density_grid_ptr = new_density;

    int bitfield_idx = voxel_idx / 32;
    uint32_t bit_mask = 1u << (voxel_idx % 32);

    if (new_density > grid->threshold) {
        atomicOr(&grid->bitfield_data()[bitfield_idx], bit_mask);
    } else {
        atomicAnd(&grid->bitfield_data()[bitfield_idx], ~bit_mask);
    }
}


void OccupancyGrid::update(
    cublasHandle_t handle,
    cudaStream_t stream,
    const DeviceHashTableAccessor* table_acc, 
    const MLP* mlp, 
    int step
) {
    // update occupancy every 16 steps
    if (step<= 256 || step % 16 != 0) return;

    printf("updating Occupancy Grid at step %d\n", step);

    // --- 1. Allocate workspace for the batched operation ---
    CudaDeviceBuffer<__half> d_batched_hash_features(n_voxels * MLP::D_in);
    CudaDeviceBuffer<__half> d_batched_sh_features(n_voxels * SH_COEFS);
    CudaDeviceBuffer<__half> d_batched_density(n_voxels);

    CudaDeviceBuffer<__half> d_hidden1(n_voxels * MLP::D_hidden);
    CudaDeviceBuffer<__half> d_density_out_full(n_voxels * MLP::D_density_out);

    cudaMemset(d_batched_sh_features.get(), 0, d_batched_sh_features.nbytes());
    CHECK_CUDA_THROW(cudaGetLastError());

    // --- 2. Launch Kernel to Prepare All MLP Inputs ---
    dim3 block_size(256);
    dim3 grid_size((n_voxels + block_size.x - 1) / block_size.x);
    
    prepare_grid_mlp_inputs_kernel<<<grid_size, block_size, 0, stream>>>(
        n_voxels, this, table_acc, d_batched_hash_features.get()
    );

    // --- 3. Call the Batched MLP Forward Pass ---
    mlp->forward_density(
        handle, stream, n_voxels,
        d_batched_hash_features.get(),
        d_batched_density.get()
    );

    CHECK_CUDA_THROW(cudaGetLastError());
    CHECK_CUDA_THROW(cudaStreamSynchronize(stream));
    printf("DEBUG: MLP forward_density completed successfully\n");

    // --- 4. Launch Kernel to Update Grid with Batched Results ---
    update_grid_from_densities_kernel<<<grid_size, block_size, 0, stream>>>(
        n_voxels, this, d_batched_density.get()
    );

    CHECK_CUDA_THROW(cudaGetLastError());
    CHECK_CUDA_THROW(cudaDeviceSynchronize());
}