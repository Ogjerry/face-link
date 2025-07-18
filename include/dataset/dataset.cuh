#pragma once

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>


#include "camera.cuh"
#include "../common/cuda_wrappers.h"
#include "../common/math_utils.h"
#include "../common/json.hpp"
#include "vec3.h"

struct TrainingRayBatch {
    int batch_size;
    CudaManagedBuffer<float3>* d_ray_origins;
    CudaManagedBuffer<float3>* d_ray_directions;
    CudaManagedBuffer<float3>* d_ground_truth_colors;
    CudaManagedBuffer<float>* d_near_bounds;
    CudaManagedBuffer<float>* d_far_bounds;
};




__global__ void calculate_ray_bounds_kernel(
    int n_rays,
    const float3* d_ray_origins,
    const float3* d_ray_directions,
    float3 aabb_min,
    float3 aabb_max,
    float* d_near_bounds,
    float* d_far_bounds  
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_idx >= n_rays) return;

    const float3& ray_o = d_ray_origins[ray_idx];
    const float3& ray_d = d_ray_directions[ray_idx];

    float t_near, t_far;

    if (get_ray_aabb_intersection(
        ray_o, ray_d, 
        aabb_min, 
        aabb_max, 
        t_near, t_far
    )) {
        d_near_bounds[ray_idx] = t_near;
        d_far_bounds[ray_idx] = t_far;
    } else {
        // If the ray misses the box, set near > far to flag it for dismissal.
        d_near_bounds[ray_idx] = 1.0f;
        d_far_bounds[ray_idx] = 0.0f;
    }
};

void calculate_ray_bounds(
    cublasHandle_t handle,
    int n_rays,
    const CudaManagedBuffer<float3>& d_ray_origins,
    const CudaManagedBuffer<float3>& d_ray_directions,
    const Vec3& aabb_min,
    const Vec3& aabb_max,
    CudaManagedBuffer<float>& d_near_bounds,
    CudaManagedBuffer<float>& d_far_bounds,
    cudaStream_t stream = 0
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n_rays + threads_per_block - 1) / threads_per_block;

    dim3 gridDim(blocks_per_grid);
    dim3 blockDim(threads_per_block);

    calculate_ray_bounds_kernel<<<gridDim, blockDim, 0, stream>>>(
        n_rays,
        d_ray_origins.get(),
        d_ray_directions.get(),
        make_float3(aabb_min.x, aabb_min.y, aabb_min.z),
        make_float3(aabb_max.x, aabb_max.y, aabb_max.z),
        d_near_bounds.get(),
        d_far_bounds.get()
    );
}


__global__ void precompute_all_rays_kernel(
    int n_images,
    int width,
    int height,
    const CameraParameters* __restrict__ d_camera_params,
    const float3* __restrict__ d_all_image_data,
    float3* __restrict__ out_all_ray_origins,
    float3* __restrict__ out_all_ray_directions,
    float3* __restrict__ out_all_ray_colors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = n_images * width * height;
    if (i >= total_pixels) return;

    int img_idx = i / (width * height);
    int pixel_idx = i % (width * height);
    int x = pixel_idx % width;
    int y = pixel_idx / width;

    const CameraParameters& cam = d_camera_params[img_idx];
    cam.generate_ray(x, y, out_all_ray_origins[i], out_all_ray_directions[i]);
    out_all_ray_colors[i] = d_all_image_data[i];
};


__global__ void generate_batch_on_the_fly_kernel(
    int n_images,
    int width,
    int height,
    const CameraParameters* __restrict__ d_camera_params,
    int batch_size,
    float3* __restrict__ out_batch_ray_origins,
    float3* __restrict__ out_batch_ray_directions,
    float3* __restrict__ out_batch_ray_colors,
    const float3* __restrict__ d_all_image_data,
    unsigned int seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    curandState state;
    curand_init(seed + i, 0, 0, &state);

    int img_idx = (int) (curand_uniform(&state) * n_images);
    if (img_idx >= n_images) img_idx = n_images - 1;

    int x = (int) (curand_uniform(&state) * width);
    int y = (int) (curand_uniform(&state) * height);

    if (x >= width) x = width - 1;
    if (y >= height) y = height - 1;

    const CameraParameters& cam = d_camera_params[img_idx];
    cam.generate_ray(x, y, out_batch_ray_origins[i], out_batch_ray_directions[i]);

    int pixel_offset = img_idx * (width * height) + y * width + x;
    out_batch_ray_colors[i] = d_all_image_data[pixel_offset];
}


class DatasetLoader {
public:
    std::vector<CameraParameters> camera_params;
    int image_width = 0, image_height = 0, total_rays = 0;
    int current_ray_idx = 0;



    CudaManagedBuffer<CameraParameters> d_camera_params;
    CudaManagedBuffer<float3> d_all_image_data;

    std::mt19937 random_generator;

    DatasetLoader(const std::string& base_path) : random_generator(std::random_device{}()) {
        printf("Initializing DatasetLaoder form path: %s\n", base_path.c_str());
        load_nerf_colmap_dataset(base_path);

        printf("Copying camera parameters to device ... \n");
        d_camera_params.resize(camera_params.size());
        for ( size_t i = 0; i < camera_params.size(); i++) {
            CameraParameters device_cam;
            device_cam.width = camera_params[i].width;
            device_cam.height = camera_params[i].height;
            memcpy(device_cam.K.m, camera_params[i].K.m, 9 * sizeof(float));
            memcpy(device_cam.c2w.m, camera_params[i].c2w.m, 16 * sizeof(float));
            d_camera_params.get()[i] = device_cam;
        }
        CHECK_CUDA_THROW(cudaDeviceSynchronize());
        printf("Camera parameters are on the device.\n");
    };



    void get_next_batch(TrainingRayBatch& gpu_batch, int gpu_idx) {
        dim3 block_dim(256);
        dim3 grid_dim((gpu_batch.batch_size + block_dim.x - 1) / block_dim.x);

        generate_batch_on_the_fly_kernel<<<grid_dim, block_dim>>>(
            camera_params.size(),
            image_width,
            image_height,
            d_camera_params.get(),
            gpu_batch.batch_size,
            gpu_batch.d_ray_origins->get(),
            gpu_batch.d_ray_directions->get(),
            gpu_batch.d_ground_truth_colors->get(),
            d_all_image_data.get(),
            random_generator() + gpu_idx
        );
    };

private:

    void load_nerf_colmap_dataset(const std::string& base_path) {
       // Try to open transforms.json, fallback to transforms_train.json
       std::string json_path = base_path + "/transforms.json";
       std::ifstream f(json_path);
       if (!f.is_open()) {
           json_path = base_path + "/transforms_train.json";
           f.open(json_path);
           if (!f.is_open()) {
               throw std::runtime_error("FATAL: Cannot open transforms.json or transforms_train.json in " + base_path);
           }
       }

       nlohmann::json data = nlohmann::json::parse(f);

       // BUG FIX: Removed the unsafe, misspelled direct access of data["camera_angile_x"].
       // The fallback logic below handles this case safely.

       // --- Robustly parse camera intrinsics ---
       float fx = data.value("fl_x", 0.0f);
       float fy = data.value("fl_y", 0.0f);
       float cx = data.value("cx", 0.0f);
       float cy = data.value("cy", 0.0f);
       image_width = data.value("w", 0);
       image_height = data.value("h", 0);

       // Fallback for older NeRF synthetic datasets if intrinsics are not specified directly
       if (fx == 0 && data.contains("camera_angle_x")) {
           printf("Found 'camera_angle_x', calculating focal length.\n");
           if (image_width == 0) image_width = data.value("w", 800);
           if (image_height == 0) image_height = data.value("h", 800);
           fx = fy = 0.5f * image_width / tanf(0.5f * float(data["camera_angle_x"]));
           cx = image_width / 2.0f;
           cy = image_height / 2.0f;
       }

       // DEBUG
       printf("=== DATASET DEBUG INFO ===\n");
       printf("Parsed intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f\n", fx, fy, cx, cy);
       printf("Image dimensions: %dx%d\n", image_width, image_height);
       printf("Number of frames in JSON: %zu\n", data["frames"].size());

       // Final check to ensure we have valid camera parameters
       if (fx == 0 || image_width == 0) {
            throw std::runtime_error("FATAL: Camera intrinsics (fl_x, w, h) could not be found or calculated from JSON.");
       }

       std::vector<float3> h_all_image_data;
       h_all_image_data.reserve(data["frames"].size() * image_width * image_height);

       int frame_count = 0;
       for (const auto& frame : data["frames"]) {
           // DEBUG
           printf("Processing frame %d...\n", frame_count++);

        
           // BUG FIX: Added safety checks for keys inside each frame.
           // This prevents a crash if a frame is missing data.
           if (!frame.contains("file_path") || !frame.contains("transform_matrix")) {
               fprintf(stderr, "Warning: A frame is missing 'file_path' or 'transform_matrix'. Skipping this frame.\n");
               continue; // Skip this malformed frame
           }

           std::string file_path = base_path + "/" + std::string(frame["file_path"]) + ".png";
           cv::Mat raw_image = cv::imread(file_path, cv::IMREAD_UNCHANGED);
           if (raw_image.empty()) {
               fprintf(stderr, "Warning: Failed to load image, skipping: %s\n", file_path.c_str());
               continue;
           }

           if(raw_image.cols != image_width || raw_image.rows != image_height) {
               cv::resize(raw_image, raw_image, cv::Size(image_width, image_height));
           }

           cv::Mat source_image;
           if (raw_image.channels() == 4) cv::cvtColor(raw_image, source_image, cv::COLOR_BGRA2BGR);
           else source_image = raw_image;

           cv::Mat float_image;
           source_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

           for (int y = 0; y < image_height; ++y) {
               for (int x = 0; x < image_width; ++x) {
                   cv::Vec3f bgr_pixel = float_image.at<cv::Vec3f>(y, x);
                   // BUG FIX: Corrected typo from b_pixel to bgr_pixel
                   h_all_image_data.push_back(make_float3(bgr_pixel[2], bgr_pixel[1], bgr_pixel[0]));
               }
           }

           std::vector<std::vector<float>> c2w_nested = frame["transform_matrix"];
           cv::Mat c2w_nerf(4, 4, CV_32F);
           for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) c2w_nerf.at<float>(r,c) = c2w_nested[r][c];

           cv::Mat nerf_to_opencv = (cv::Mat_<float>(4, 4) << 1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1);
           cv::Mat c2w_opencv = c2w_nerf * nerf_to_opencv;

           cv::Mat K_cv = cv::Mat::zeros(3, 3, CV_32F);
           K_cv.at<float>(0, 0) = fx;
           K_cv.at<float>(1, 1) = fy;
           K_cv.at<float>(0, 2) = cx;
           K_cv.at<float>(1, 2) = cy;
           K_cv.at<float>(2, 2) = 1.0f;

           camera_params.emplace_back(K_cv, c2w_opencv, image_width, image_height);
       }

       d_all_image_data.resize(h_all_image_data.size());
       memcpy(d_all_image_data.get(), h_all_image_data.data(), h_all_image_data.size() * sizeof(float3));
       CHECK_CUDA_THROW(cudaDeviceSynchronize());
       printf("Successfully loaded %zu camera poses and aggregated pixel data.\n", camera_params.size());


       printf("=== FINAL DATASET STATS ===\n");
       printf("Total camera poses loaded: %zu\n", camera_params.size());
       printf("Total pixels loaded: %zu\n", h_all_image_data.size());
       printf("Expected total rays: %zu\n", camera_params.size() * image_width * image_height);
       printf("========================\n");
    }   
};