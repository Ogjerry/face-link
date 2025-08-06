#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <utility>
#include <cassert>
#include <iostream>

// Basic CUDA Error Checking Macro
#ifndef CHECK_CUDA_THROW
#define CHECK_CUDA_THROW(call)              \
    do {                                    \
        cudaError_t err = call;             \
        if (err != cudaSuccess) {           \
            char errMsg[2048];              \
            snprintf(errMsg, sizeof(errMsg), "CUDA error in %s:%d : %s (%d) for call %s", \
                    __FILE__, __LINE__, cudaGetErrorString(err), err, #call);  \
            fprintf(stderr, "%s\n", errMsg);                                   \
            throw std::runtime_error(errMsg);                                  \
        }                                                                      \
    } while (0)
#endif

// Simplified CUDA check for non-critical places like destructors/move assignments
#ifndef LOG_CUDA_ERROR
#define LOG_CUDA_ERROR(call, context_msg)                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "RAII Warning: %s - CUDA call %s failed in %s:%d : %s (%d)\n", \
                    context_msg, #call, __FILE__, __LINE__, cudaGetErrorString(err), err); \
        }                                                                      \
    } while (0)
#endif

// RAII Wrapper for cudaMalloc / cudaFree (Device Memory)
template <typename T>
class CudaDeviceBuffer {
private:
    T* device_ptr;
    size_t count;

public:
    CudaDeviceBuffer() : device_ptr(nullptr), count(0) {}

    explicit CudaDeviceBuffer(size_t size) : device_ptr(nullptr), count(0) {
        if (size > 0) {
            resize(size);
        }
    }

    ~CudaDeviceBuffer() {
        free();
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : 
        device_ptr(other.device_ptr), count(other.count) {
        other.device_ptr = nullptr;
        other.count = 0;
    }

    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            device_ptr = other.device_ptr;
            count = other.count;
            other.device_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    // NEW: Explicitly free memory
    void free() {
        if (device_ptr) {
            LOG_CUDA_ERROR(cudaFree(device_ptr), "CudaDeviceBuffer::free()");
            device_ptr = nullptr;
            count = 0;
        }
    }
    
    // NEW: Resize the buffer
    void resize(size_t new_size) {
        if (count == new_size) return;
        free();
        count = new_size;
        if (count > 0) {
            CHECK_CUDA_THROW(cudaMalloc(&this->device_ptr, nbytes()));
        }
    }

    __host__ __device__ T* get() const { return device_ptr; }
    __host__ __device__ size_t size() const { return count; }
    __host__ __device__ size_t nbytes() const { return count * sizeof(T); }

    __device__ __host__ T& operator[](size_t idx) {
        #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
        assert(idx < count && "Index out of bounds");
        #endif
        return device_ptr[idx];
    }

    __device__ __host__ const T& operator[](size_t idx) const {
        #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
        assert(idx < count && "Index out of bounds");
        #endif
        return device_ptr[idx];
    }
};

// RAII Wrapper for cudaMallocManaged / cudaFree (Managed Memory)
template <typename T>
class CudaManagedBuffer {
private:
    T* managed_ptr;
    size_t count;

public:
    CudaManagedBuffer() : managed_ptr(nullptr), count(0) {}

    explicit CudaManagedBuffer(size_t size) : managed_ptr(nullptr), count(0) {
        if (size > 0) {
            resize(size);
        }
    }

    ~CudaManagedBuffer() {
        free();
    }

    CudaManagedBuffer(const CudaManagedBuffer&) = delete;
    CudaManagedBuffer& operator=(const CudaManagedBuffer&) = delete;

    CudaManagedBuffer(CudaManagedBuffer&& other) noexcept : 
        managed_ptr(other.managed_ptr), count(other.count) {
        other.managed_ptr = nullptr;
        other.count = 0;
    }

    CudaManagedBuffer& operator=(CudaManagedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            managed_ptr = other.managed_ptr;
            count = other.count;
            other.managed_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    // NEW: Explicitly free memory
    void free() {
        if (managed_ptr) {
            LOG_CUDA_ERROR(cudaFree(managed_ptr), "CudaManagedBuffer::free()");
            managed_ptr = nullptr;
            count = 0;
        }
    }
    
    // NEW: Resize the buffer
    void resize(size_t new_size) {
        if (count == new_size) return;
        free();
        count = new_size;
        if (count > 0) {
            CHECK_CUDA_THROW(cudaMallocManaged(&this->managed_ptr, nbytes()));
        }
    }

    __host__ __device__ T* get() const { return managed_ptr; }
    __host__ __device__ size_t size() const { return count; }
    __host__ __device__ size_t nbytes() const { return count * sizeof(T); }

    __device__ __host__ T& operator[](size_t idx) {
        #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
        assert(idx < count && "Index out of bounds");
        #endif
        return managed_ptr[idx];
    }

    __device__ __host__ const T& operator[](size_t idx) const {
        #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
        assert(idx < count && "Index out of bounds");
        #endif
        return managed_ptr[idx];
    }
};