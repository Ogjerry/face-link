#pragma once

#include <opencv2/opencv.hpp> // Main OpenCV header
#include <vector_types.h>     // For CUDA's float3
#include <vector>


// device version of CV::Mat
struct Mat3x3 {
    float m[9];

    __host__ __device__ float get(int r, int c) const {
        return m[r * 3 + c];
    }
};

struct Mat4x4 {
    float m[16];
};

struct CameraParameters {
    int width, height;
    Mat3x3 K;      // Intrinsics (Focal Length, Principal Point)
    Mat4x4 c2w;    // Extrinsics (Camera-to-World transformation)

    // Default constructor for placeholder creation
    CameraParameters() : width(0), height(0) {}

    // HOST-ONLY Constructor:
    // Takes CPU-only cv::Mat objects and extracts their data into our
    // simple, GPU-friendly float arrays.
    __host__ CameraParameters(
        const cv::Mat& K_cv,
        const cv::Mat& c2w_cv,
        int w, int h
    ) : width(w), height(h) {
        
        // Ensure matrices have the correct size and type before copying
        assert(K_cv.rows == 3 && K_cv.cols == 3 && K_cv.type() == CV_32F);
        assert(c2w_cv.rows == 4 && c2w_cv.cols == 4 && c2w_cv.type() == CV_32F);

        // Copy data from cv::Mat to our simple float arrays
        memcpy(K.m, K_cv.data, 9 * sizeof(float));
        memcpy(c2w.m, c2w_cv.data, 16 * sizeof(float));
    }

    // DEVICE-COMPATIBLE Ray Generation Function:
    // This function can be called from a CUDA kernel because it only uses
    // simple types (int, float, float3) and performs its own math.
    __host__ __device__ void generate_ray(
        int px, int py,
        float3& out_origin,
        float3& out_direction
    ) const {
        // Get camera intrinsics from our device-friendly struct
        const float fx = K.get(0, 0);
        const float fy = K.get(1, 1);
        const float cx = K.get(0, 2);
        const float cy = K.get(1, 2);

        // 1. Unproject pixel coordinates (px, py) to a direction in camera space
        float3 dir_c;
        dir_c.x = (static_cast<float>(px) + 0.5f - cx) / fx;
        dir_c.y = (static_cast<float>(py) + 0.5f - cy) / fy;
        dir_c.z = 1.0f;

        // 2. Transform the direction from camera space to world space using the rotation part of c2w
        // This is a manual 3x3 matrix-vector multiplication
        float3 dir_w;
        dir_w.x = dir_c.x * c2w.m[0] + dir_c.y * c2w.m[1] + dir_c.z * c2w.m[2];
        dir_w.y = dir_c.x * c2w.m[4] + dir_c.y * c2w.m[5] + dir_c.z * c2w.m[6];
        dir_w.z = dir_c.x * c2w.m[8] + dir_c.y * c2w.m[9] + dir_c.z * c2w.m[10];

        // 3. Normalize the final world-space direction vector
        float norm = sqrtf(dot(dir_w, dir_w));
        if (norm > 1e-6f) {
            out_direction = dir_w / norm;
        } else {
            out_direction = make_float3(0.f, 0.f, 0.f);
        }

        // 4. The ray origin is the camera's position (translation part of c2w)
        out_origin.x = c2w.m[3];
        out_origin.y = c2w.m[7];
        out_origin.z = c2w.m[11];
    }
};