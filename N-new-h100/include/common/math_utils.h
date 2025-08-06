#pragma once
#include <vector_types.h>
#include <cmath>
#include "../dataset/vec3.h"
#include <cuda_fp16.h>

// Helper function to convert a quaternion (w, x, y, z) to a 3x3 rotation matrix (row-major)
inline void quaternion_to_rotation_matrix(const double q[4], float R[9]) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double n = w * w + x * x + y * y + z * z;
    const double s = (n > 0) ? (2.0 / n) : 0.0;

    const double xs = x * s,  ys = y * s,  zs = z * s;
    const double wx = w * xs, wy = w * ys, wz = w * zs;
    const double xx = x * xs, xy = x * ys, xz = x * zs;
    const double yy = y * ys, yz = y * zs, zz = z * zs;

    R[0] = 1.0 - (yy + zz); R[1] = xy - wz;        R[2] = xz + wy;
    R[3] = xy + wz;        R[4] = 1.0 - (xx + zz); R[5] = yz - wx;
    R[6] = xz - wy;        R[7] = yz + wx;        R[8] = 1.0 - (xx + yy);
};


// Vector dot product
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
};

// Scalar multiplication for float3
__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
};

__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
};

__device__ inline float3 operator+(const float3&a, const float3&b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
};


__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
};

__host__ __device__ inline float3 operator/(const float3& a, int n) {
    return make_float3(a.x / n, a.y / n, a.z / n);
};


// --- Math Operations ---
__device__ inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
};

__device__ inline float sigmoid_derivative(float x) {
    return x * (1.0f - x);
};




// --- Full Precision Activation Functions ---
__host__ __device__ __forceinline__ float relu(float x) { 
    return fmaxf(0.f, x); 
};

__host__ __device__ __forceinline__ float sigmoid(float x) { 
    return 1.f / (1.f + expf(-x)); 
};


__host__ __device__ __forceinline__ float softplus(float x) {
    return logf(1.f + expf(x));
}

__host__ __device__ __forceinline__ float softplus_derivative(float x) {
    return sigmoid(x);
}
// --- Full Precision Activation Functions ---


// --- Math Operations ---



// --- float3 Operators ---

// --- float2 Operators ---


__host__ __device__ inline float2 operator/(const float2& a, const float& b) {
    return make_float2(a.x / b, a.y / b);
};

__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
};

// In-place add
__host__ __device__ inline void operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
};

// Subtract two float2 vectors
__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
};

// Multiply float2 by a scalar
__host__ __device__ inline float2 operator*(const float2& a, float s) {
    return make_float2(a.x * s, a.y * s);
};

// Multiply float2 by a scalar (commutative)
__host__ __device__ inline float2 operator*(float s, const float2& a) {
    return make_float2(a.x * s, a.y * s);
};

// Component-wise multiply two float2 vectors
__host__ __device__ inline float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
};

// --- float2 Operators ---



// In-place add
__host__ __device__ inline void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
};



// --- Half Precision Activation Functions ---

// A numerically stable softplus implementation for __half precision.
// It avoids overflow for large inputs.
__device__ __forceinline__ __half hsoftplus(const __half& x) {
    // For large x, softplus(x) approx x.
    // For small x, softplus(x) approx 0.
    // A threshold of 20 is safe for __half. exp(20) would overflow.
    const float x_f = __half2float(x);
    if (x_f > 20.0f) {
        return x;
    }
    if (x_f < -20.0f) {
        return __float2half(0.0f);
    }
    return hlog(__hadd(__float2half(1.0f), hexp(x)));
}

// A numerically stable sigmoid implementation for __half precision.
// This is the GRADIENT of softplus.
__device__ __forceinline__ __half hsigmoid(const __half& x) {
    const float x_f = __half2float(x);
    // For large positive x, sigmoid -> 1
    // For large negative x, sigmoid -> 0
    // We use a different formula for positive vs negative x to avoid exp(large_positive).
    if (x_f >= 0.0f) {
        // Use formula: 1 / (1 + exp(-x))
        const __half neg_x = __hneg(x);
        return __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(neg_x)));
    } else {
        // Use formula: exp(x) / (1 + exp(x))
        const __half exp_x = hexp(x);
        return __hdiv(exp_x, __hadd(__float2half(1.0f), exp_x));
    }
}

// Derivative of Sigmoid for the backward pass
__device__ __forceinline__ float hsigmoid_derivative(const float& x) {
    float sig = 1.f / (1.f + expf(-x));
    return sig * (1.f - sig);
};

// Overload the multiplication operator for float * __half
__device__ __forceinline__ float operator*(const float& f, const __half& h) {
    return f * __half2float(h);
};
// --- Half Precision Activation Functions ---
