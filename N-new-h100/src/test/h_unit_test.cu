#include "../../include/nerf/hashing.cuh"
#include <iostream>
#include <vector>

// This is the wrapper we need to add to hashing.cuh
__global__ void accumulate_hash_gradients_kernel_test(
    DeviceHashTableAccessor* table,
    const int* all_corners_data,
    const float* dL_d_feat);

// Add this global wrapper to make the device function callable from the host for testing.
__global__ void accumulate_hash_gradients_kernel_test(
    DeviceHashTableAccessor* table,
    const int* all_corners_data,
    const float* dL_d_feat)
{
    accumulate_hash_gradients_kernel(table, all_corners_data, dL_d_feat);
}


int main(int argc, char* argv[]) {
    std::cout << "--- Starting Isolation Test for accumulate_hash_gradients_kernel ---" << std::endl;

    // ====================================================================
    // 1. Set up the HashTable object, just like in the main application.
    // ====================================================================
    try {
        const int L = 16;
        const int N_min = 16;
        const int N_max = 2048;
        const int T_per_level = 524288;
        const int F = 4;

        HashTable table(L, N_min, N_max, T_per_level);
        DeviceHashTableAccessor* d_table_acc = table.get_device_accessor();
        std::cout << "[Step 1] HashTable and DeviceAccessor created successfully." << std::endl;

        // ====================================================================
        // 2. Create MOCK input data on the GPU.
        // ====================================================================

        const int n_features = L * F;
        CudaDeviceBuffer<float> d_mock_dL_d_feat(n_features);
        std::vector<float> h_mock_dL_d_feat(n_features, 1.0f);
        
        CHECK_CUDA_THROW(cudaMemcpy(
            d_mock_dL_d_feat.get(),
            h_mock_dL_d_feat.data(),
            d_mock_dL_d_feat.nbytes(),
            cudaMemcpyHostToDevice
        ));
        std::cout << "[Step 2a] Mock dL_d_feat created and copied to device." << std::endl;

        const int n_corners_data = L * 8 * 2;
        CudaDeviceBuffer<int> d_mock_corners_data(n_corners_data);
        std::vector<int> h_mock_corners_data(n_corners_data);
        for (int level = 0; level < L; ++level) {
            for (int corner = 0; corner < 8; ++corner) {
                int offset = (level * 8 + corner) * 2;
                h_mock_corners_data[offset] = 0; 
                float weight = 1.0f / 8.0f;
                
                // --- CORRECTED LINE ---
                // Perform the bit-level cast on the host using C++ reinterpret_cast
                h_mock_corners_data[offset + 1] = *reinterpret_cast<int*>(&weight);
            }
        }
        
        CHECK_CUDA_THROW(cudaMemcpy(
            d_mock_corners_data.get(),
            h_mock_corners_data.data(),
            d_mock_corners_data.nbytes(),
            cudaMemcpyHostToDevice
        ));
        std::cout << "[Step 2b] Mock all_corners_data created and copied to device." << std::endl;

        // ====================================================================
        // 3. Launch the kernel with the mock data.
        // ====================================================================
        std::cout << "[Step 3] Launching accumulate_hash_gradients_kernel..." << std::endl;
        
        // Note: Make sure you have added the __global__ wrapper for this kernel in hashing.cuh
        accumulate_hash_gradients_kernel_test<<<1, L*F>>>(
            d_table_acc,
            d_mock_corners_data.get(),
            d_mock_dL_d_feat.get()
        );

        CHECK_CUDA_THROW(cudaGetLastError());
        CHECK_CUDA_THROW(cudaDeviceSynchronize());

        std::cout << "[Step 4] Kernel completed WITHOUT crashing." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "--- TEST FAILED ---" << std::endl;
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "--- TEST SUCCEEDED ---" << std::endl;
    return 0;
}