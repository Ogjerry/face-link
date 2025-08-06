#define CPU_UTIL
#ifdef  CPU_UTIL

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cassert>


#define CHECK(call)                                                          \
{                                                                                 \
    const cudaError_t error = call;                                               \
    if (error != cudaSuccess)                                                     \
    {                                                                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                             \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                  \
    }                                                                             \
}

#define ASSERT_PTR_VALID(ptr) \
{ \
    if ((ptr) == nullptr) { \
        fprintf(stderr, "FATAL ERROR: Pointer check failed for '%s' (%s:%d)\n", \
                #ptr, __FILE__, __LINE__); /*'#' sign will extract the variable name for print */ \
        abort(); \
    } else { \
        /* Print pointer value on success for debugging */ \
        printf("Debug: Pointer '%s' is valid at %p (%s:%d)\n", #ptr, (void*)(ptr), __FILE__, __LINE__); \
    } \
}

#define CHECK_CUSOLVER(call)                              \
    {                                                     \
        cusolverStatus_t status = (call);                 \
        if (status != CUSOLVER_STATUS_SUCCESS) {          \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n" ,__FILE__, __LINE__, status);       \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    }


#define CHECK_CUBLAS(call)                                \
    {                                                     \
        cublasStatus_t status = (call);                   \
        if (status != CUBLAS_STATUS_SUCCESS) {            \
            fprintf(stderr, "cuBLAS error at %s:%d: %d \n", __FILE__, __LINE__, status);         \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    }








// void check_result(float *host_ref, float *gpu_ref, const int N) {
//     double epsilon = 1.0E-8;
//     bool match = 1;
//     for (int i = 0; i < N; i++) {
//         if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
//             match = 0;
//             printf("Arrays do not match!\n");
//             printf("CPU %lf \n GPU %lf \n at current %d\n", host_ref[i], gpu_ref[i], i);
//             break;
//         }
//     }
//     if (match) printf("Arrays match.\n\n");
// }
// 
// 
// 
// // Recursive Implementation of Interleaved Pair Approach
// long int cpusum(int *data, int const size)
// {
//     if (size == 0) return 0; // Handle empty array
// 
//     long int *temp = (long int*) malloc( sizeof(long int) * size);
//     if (!temp) return 0; // Allocation failed
// 
//     // Initialize temp array with input data
//     for (int i = 0; i < size; i++) {
//         temp[i] = data[i];
//     }
// 
//     int isize = size;
//     while (isize > 1) {
//         int const stride = isize / 2;
//         int remainder = isize % 2;
// 
//         for (int i = 0; i < stride; i++) {
//             temp[i] = temp[i] + temp[i + stride];
//         }
// 
//         // If the array size is odd, add the last element to the first element
//         if (remainder != 0) {
//             temp[0] += temp[isize - 1];
//         }
// 
//         isize = stride + remainder;
//     }
// 
//     long int result = temp[0]; // Final result
//     free(temp); // Free the allocated memory
//     return result;
// }





#endif
