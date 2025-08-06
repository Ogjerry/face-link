#pragma once

#define OCCUPANCY_GRID_h
#ifdef  OCCUPANCY_GRID_h

#include <cublas_v2.h>
#include "hashing.cuh"

/////////////////////////////////////////////////////////////////////////
// ------------- Occupancy Grid ------------- //
/////////////////////////////////////////////////////////////////////////


// FORWARD DECLARATIONS:
// Instead of including the full headers, we just tell the compiler these types exist.
// This breaks the dependency cycle and solves the error.
class MLP;
struct DeviceHashTableAccessor;
struct hash_level;

////////////////////// Device Kernels //////////////////////

__device__ __host__ inline int linearize3D(int3 coord, int3 grid_resolution) {
    /**
     * @note: linearize seperate dimensions is quite different from
     *        uniform resolution. The z, y, x order matters for cache
     *        effeciency since z major keeps x/y neighbors in cache. 
     * 
     * @example:
     * 
     * Correct Cache:
     * 
     *      Layer 0 (z=0):
     *      [ (0,0,0) (1,0,0) (2,0,0) ]
     *      [ (0,1,0) (1,1,0) (2,1,0) ]
     *      [ (0,2,0) (1,2,0) (2,2,0) ]
     *      
     *      Layer 1 (z=1):
     *      [ (0,0,1) (1,0,1) (2,0,1) ]...
     * 
     * 
     * Inefficient Cache:
     * 
     *      X=0:
     *      [ (0,0,0) (0,1,0) (0,2,0) ]
     *      [ (0,0,1) (0,1,1) (0,2,1) ]...
     *      
     *      X=1:
     *      [ (1,0,0) (1,1,0) (1,2,0) ]...
     * 
     * 
     * 
     * Type                Data Structure       Linearization formula 
     * ______________________________________________________________
     * Uniform Resolution	int resolution;	    z*N² + y*N + x
     * ______________________________________________________________
     * Separate Dimensions	int3 resolution;	z*W*H + y*W + x     #
     * 
     */
    // Row-major order: z*width*height + y*width + x
    return coord.z * grid_resolution.y * grid_resolution.x
    + coord.y * grid_resolution.x 
    + coord.x;
};





////////////////////// Device Kernels //////////////////////

class OccupancyGrid {
private:
    CudaManagedBuffer<uint32_t> m_bitfield; // 1 bit per voxel
    CudaManagedBuffer<float>    m_density_grid;  // density estimates

public:
    const float decay;
    float threshold;                     // Occupancy threshold
    int3 resolution;                     // Grid resolution
    int n_voxels;                        // Total number of voxels

    // --- Constructor ---
    OccupancyGrid(int3 res, float occ_decay, float occ_threashold) :
        resolution(res),
        decay(occ_decay),
        threshold(occ_threashold),
        n_voxels(res.x * res.y * res.z)
    {
        printf("Constructing Occupancy Grid with resolution %d x %d x %d\n", resolution.x, resolution.y, resolution.z);
        const int n_cells = resolution.x * resolution.y * resolution.z;

        // The bitfield needs one bit per cell. Allocate enough uint32_t to cover all cells.
        const int bitfield_size = (n_cells + 31) / 32;
        m_bitfield = CudaManagedBuffer<uint32_t>(bitfield_size);
        
        // The density grid stores one float per cell.
        m_density_grid = CudaManagedBuffer<float>(n_cells);
        
        // Initialize memory to zero upon creation.
        CHECK_CUDA_THROW(cudaMemset(m_bitfield.get(), 0, m_bitfield.nbytes()));
        CHECK_CUDA_THROW(cudaMemset(m_density_grid.get(), 0, m_density_grid.nbytes()));
        printf("Occupancy Grid constructed and memory allocated.\n");
    };


    // --- Public Host-side Accessor Methods ---
    __host__ float* density_grid_data_host() const { return m_density_grid.get(); }
    
    // --- Methods for Multi-GPU Grid Synchronization ---
    /**
     * @brief Returns a void pointer to the raw bitfield data for NCCL operations.
     */
    __host__ void* get_grid_data_ptr() const {
        return (void*)m_bitfield.get();
    }

    /**
     * @brief Returns the size of the bitfield data in bytes for NCCL operations.
     */
    __host__ size_t get_grid_data_size() const {
        return m_bitfield.nbytes();
    }
    
    
    // --- Public Device-side Accessor Methods ---
    // These methods provide controlled, read-only access to the raw pointers for our kernels.
    __device__ uint32_t* bitfield_data() const { return m_bitfield.get(); }
    __device__ float* density_grid_data() const { return m_density_grid.get(); }



    __device__ bool is_occupied(const float3& pos) const { // const here adds none modifiation for later use
        // 1. Convert to grid coordinates with flooring
        
        int3 g = make_int3(   // Position in [0,1]
            floor((pos.x) * resolution.x),
            floor((pos.y) * resolution.y),
            floor((pos.z) * resolution.z)
        );
        // 2. Boundary check
        if(g.x <0 || g.x >= resolution.x || 
           g.y <0 || g.y >= resolution.y ||
           g.z <0 || g.z >= resolution.z) return false;

        // 3. Linearize with strides (see linearize3D)
        int idx = linearize3D(g, resolution);
        // check bitfield with unsigned bit shift (32 bits per uint32_t, 1 bit per voxel, 32 voxels per uint32_t)
        uint32_t mask = 1u << (idx % 32);
        return (m_bitfield.get()[idx/32] & mask) != 0;
    };

    __device__ float3 world_pos_from_grid_coords(const int3& grid_coords) const {
        
        // Convert integer grid coords to world coords in [-1, 1] range (approx)
        return make_float3(
            (grid_coords.x + 0.5f) / resolution.x,
            (grid_coords.y + 0.5f) / resolution.y,
            (grid_coords.z + 0.5f) / resolution.z
        );
    }


    // --- Host-side Functions ---
    void update(
        cublasHandle_t handle,
        cudaStream_t stream,
        const DeviceHashTableAccessor* table_acc, 
        const MLP* mlp, 
        int step
    );
};


__global__ void prepare_grid_mlp_inputs_kernel(
    int n_voxels,
    const OccupancyGrid* grid,
    const DeviceHashTableAccessor* table_acc,
    __half* d_batched_hash_features
);


__global__ void update_grid_from_densities_kernel(
    int n_voxels,
    OccupancyGrid* grid,
    const __half* d_batched_densities
);


////////////////////// Host Kernels //////////////////////
////////////////////// Host Kernels //////////////////////


/////////////////////////////////////////////////////////////////////////
// ------------- Occupancy Grid ------------- //
/////////////////////////////////////////////////////////////////////////



#endif