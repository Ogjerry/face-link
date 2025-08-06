#include <cstdio>
#include <cuda_fp16.h>
#include "../../include/nerf/mlp.cuh" // Make sure this path is correct

// This kernel now performs a write, making the test valid.
__global__ void access_test_kernel(__half* bias_ptr, __half* write_ptr) {
    // A thread will only write if its index is one of our test cases.
    if (threadIdx.x == 0) bias_ptr[0]   = __float2half(0.0f);
    if (threadIdx.x == 1) bias_ptr[63]  = __float2half(0.0f);
    if (threadIdx.x == 2) bias_ptr[64]  = __float2half(0.0f); // The index that was crashing
    if (threadIdx.x == 3) bias_ptr[127] = __float2half(0.0f); // The last valid index
}

int main() {
    printf("--- Starting MLP Isolation Test ---\n");
    cudaSetDevice(0);

    // 1. Construct the MLP
    MLP* mlp = new MLP(1337);
    CHECK_CUDA_THROW(cudaDeviceSynchronize());
    printf("MLP constructor finished.\n");

    // 2. Get the raw pointer using the corrected getter
    const __half* bias_ptr = mlp->density_biases1();
    printf("Bias buffer pointer is: %p\n", bias_ptr);

    if (!bias_ptr) {
         printf("FATAL: Bias pointer is null!\n");
         return -1;
    }

    // 3. Launch a simple kernel to access the pointer
    printf("Launching test kernel to access indices 0, 63, 64, and 127...\n");
    access_test_kernel<<<1, 4>>>(bias_ptr);
    CHECK_CUDA_THROW(cudaDeviceSynchronize());

    printf("\n--- TEST PASSED ---\n");
    printf("Successfully accessed bias buffer.\n");

    delete mlp;
    return 0;
}