#include "include/nerf/hashing.cuh"
#include "include/nerf/mlp.cuh"
#include "include/nerf/renderer.cuh"
#include "include/nerf/occupancy_grid.cuh"
#include "include/dataset/dataset.cuh"
#include "include/dataset/vec3.h"
#include "include/dataset/camera.cuh"
#include "include/common/global_norm.h"
#include "include/common/distortion_loss.h"

// MULTI-GPU
#include <vector>
#include <omp.h>

#include <cooperative_groups.h>
#include <nccl.h>
#include <cublas_v2.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",              \
        __FILE__,__LINE__,ncclGetErrorString(r));    \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// Add this __global__ kernel to main.cu
__global__ void debug_hash_encode_kernel_wrapper(
    const float3 pos,
    const DeviceHashTableAccessor* table_acc,
    __half* output_features)
{
    // This global kernel's only job is to call the device function
    hash_encode_kernel(pos, table_acc, output_features);
};

// In main.cu
void render_image(
    cublasHandle_t handle,
    cudaStream_t stream,
    Renderer* renderer,
    DeviceHashTableAccessor* d_table_acc,
    MLP* d_mlp,
    OccupancyGrid* d_grid,
    const DatasetLoader& dataset_loader,
    const Vec3& scene_aabb_min,
    const Vec3& scene_aabb_max
) {
    printf("\n--- Rendering Novel View ---\n");

    // In main.cu, inside render_image()

    // --- START DEBUG BLOCK ---
    printf("\n--- DEBUG: Querying single point ---\n");

    // A point that should be inside the Lego bulldozer
    float3 test_pos = {0.1f, 0.1f, 0.1f}; 

    // Create a buffer for the hash features of our single point
    CudaDeviceBuffer<__half> d_test_hash_features(MLP::D_in);

    // Run the hash encoding kernel for just this one point.
    // Note the use of 'd_table_acc' which is passed into this function.
    debug_hash_encode_kernel_wrapper<<<1, 1>>>(
        test_pos,
        d_table_acc, // Correct variable
        d_test_hash_features.get()
    );

    // Create a buffer for the single density output
    CudaDeviceBuffer<__half> d_test_density_output(1);

    // Call the density-only forward pass using the 'd_mlp' passed into this function.
    d_mlp->forward_density(
        handle, stream, 1, // n_points = 1
        d_test_hash_features.get(),
        d_test_density_output.get()
    );

    // Copy the result back to the CPU to print it
    __half h_test_density;
    CHECK_CUDA_THROW(cudaStreamSynchronize(stream)); // Wait for the kernel to finish
    CHECK_CUDA_THROW(cudaMemcpy(&h_test_density, d_test_density_output.get(), sizeof(__half), cudaMemcpyDeviceToHost));

    printf("Density for point (0.1, 0.1, 0.1): %f\n", __half2float(h_test_density));
    printf("--- END DEBUG BLOCK ---\n\n");
    // --- You can delete or comment out this block after testing ---

    const int width = 800;
    const int height = 800;
    const int n_total_rays = width * height;

    // --- 1. Generate ALL rays for the image ---
    std::vector<float3> h_all_ray_origins(n_total_rays);
    std::vector<float3> h_all_ray_directions(n_total_rays);
    
    // Camera and ray generation logic
    const auto& ref_cam = dataset_loader.camera_params[0];
    float fx = ref_cam.K.get(0, 0), fy = ref_cam.K.get(1, 1), cx = ref_cam.K.get(0, 2), cy = ref_cam.K.get(1, 2);
    float angle = 25.0f * 3.14159265f / 180.0f;
    float c2w_data[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 4.0f,  // Looking from z=4
        0.0f, 0.0f, 0.0f, 1.0f
    };
    cv::Mat nerf_to_opencv = (
        cv::Mat_<float>(4, 4) << 
        1, 0, 0, 0, 
        0, -1, 0, 0, 
        0, 0, -1, 0, 
        0, 0, 0, 1
    );
    cv::Mat c2w_novel_nerf = cv::Mat(4, 4, CV_32F, c2w_data).clone();
    cv::Mat c2w_novel_opencv = c2w_novel_nerf * nerf_to_opencv;
    cv::Mat K_render = cv::Mat::zeros(3, 3, CV_32F);
    K_render.at<float>(0, 0) = fx; K_render.at<float>(1, 1) = fy; K_render.at<float>(0, 2) = cx; K_render.at<float>(1, 2) = cy; K_render.at<float>(2, 2) = 1.0f;
    CameraParameters render_cam(K_render, c2w_novel_opencv, width, height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            render_cam.generate_ray(x, y, h_all_ray_origins[y * width + x], h_all_ray_directions[y * width + x]);
        }
    }
    printf("Generated %d total rays for final render.\n", n_total_rays);

    // --- 2. Allocate output buffer ---
    CudaDeviceBuffer<float4> d_all_predicted_pixels(n_total_rays);

    // --- 3. Process image in smaller batches to avoid OOM ---
    const int render_batch_size = 16384; // Smaller batch size to avoid OOM
    const int num_batches = (n_total_rays + render_batch_size - 1) / render_batch_size;
    
    // --- ADDED --- Define sampling configuration for final render (can be higher quality)
    const int render_coarse_samples = 128;
    const int render_fine_samples = 256;
    const int render_samples_per_ray = render_coarse_samples + render_fine_samples;

    // Create workspace for batch processing
    RendererWorkspace workspace(render_batch_size, render_samples_per_ray);
    
    // Allocate managed buffers for batch processing
    CudaManagedBuffer<float3> d_batch_origins(render_batch_size);
    CudaManagedBuffer<float3> d_batch_directions(render_batch_size);
    CudaManagedBuffer<float> d_near_bounds(render_batch_size);
    CudaManagedBuffer<float> d_far_bounds(render_batch_size);


    

    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = batch_idx * render_batch_size;
        int batch_end = std::min(batch_start + render_batch_size, n_total_rays);
        int n_rays_in_batch = batch_end - batch_start;
        
        if (batch_idx % 10 == 0) {
            printf("  Rendering batch %d/%d (rays %d-%d)\n", 
                   batch_idx + 1, num_batches, batch_start, batch_end - 1);
        }
        
        // Copy this batch's rays to managed buffers
        memcpy(d_batch_origins.get(), 
               &h_all_ray_origins[batch_start], 
               n_rays_in_batch * sizeof(float3));
        
        memcpy(d_batch_directions.get(), 
               &h_all_ray_directions[batch_start], 
               n_rays_in_batch * sizeof(float3));

        // Calculate bounds for this batch
        calculate_ray_bounds(
            handle,
            n_rays_in_batch, 
            d_batch_origins, 
            d_batch_directions, 
            scene_aabb_min, 
            scene_aabb_max, 
            d_near_bounds, 
            d_far_bounds, 
            stream
        );

        // Render this batch directly to the correct position in the output buffer
        float4* d_batch_output = d_all_predicted_pixels.get() + batch_start;

        renderer->render_forward(
            handle, stream, d_table_acc, d_mlp, d_grid, n_rays_in_batch,
            d_batch_origins.get(), d_batch_directions.get(),
            d_near_bounds.get(), d_far_bounds.get(), 
            render_coarse_samples,
            render_fine_samples,
            workspace, 
            d_batch_output
        );
        
        // Synchronize after each batch to ensure completion
        CHECK_CUDA_THROW(cudaStreamSynchronize(stream));
    }
    
    printf("All render batches complete.\n");

    // --- 4. Save Image ---
    printf("Saving image to 'final_render.png'...\n");
    cv::Mat output_image_rgba(height, width, CV_32FC4);
    CHECK_CUDA_THROW(cudaMemcpy(
        output_image_rgba.data, 
        d_all_predicted_pixels.get(), 
        n_total_rays * sizeof(float4), 
        cudaMemcpyDeviceToHost
    ));
    
    cv::Mat output_image_bgr;
    output_image_rgba.convertTo(output_image_rgba, CV_8UC4, 255.0);
    cv::cvtColor(output_image_rgba, output_image_bgr, cv::COLOR_RGBA2BGR);
    
    if (!cv::imwrite("final_render.png", output_image_bgr)) {
        printf("ERROR: Failed to save rendered image!\n");
    } else {
        printf("Image saved successfully!\n");
    }
}




// Gentler learning rate schedule
float get_learning_rate(int step, float initial_lr, int warmup_steps, int total_steps) {
    // --- STAGE 1: Linear Warm-up ---
    // Same as before. This is a best practice.
    if (step < warmup_steps) {
        return initial_lr * (float)(step + 1) / (float)warmup_steps;
    }

    // --- NEW: Define Training Stages ---
    // After this step, we enter the refinement phase by dropping the LR.
    const int refinement_start_step = 5000; 
    
    // The factor by which to reduce the LR during refinement. 0.1 means a 10x drop.
    const float refinement_lr_factor = 0.1f; 

    // Determine the effective LR for the current stage
    float stage_lr = initial_lr;
    if (step >= refinement_start_step) {
        stage_lr = initial_lr * refinement_lr_factor;
    }

    // --- STAGE 2 & 3: Cosine Decay within the current stage ---
    // The decay now happens much faster and "resets" if we enter a new stage.
    const int decay_start_step = (step < refinement_start_step) ? warmup_steps : refinement_start_step;
    const int decay_end_step = (step < refinement_start_step) ? refinement_start_step : total_steps;
    
    // Ensure we don't divide by zero if start and end are the same
    if (decay_end_step <= decay_start_step) {
        return stage_lr;
    }
    
    float progress = (float)(step - decay_start_step) / (float)(decay_end_step - decay_start_step);
    progress = std::min(1.0f, progress); // Clamp progress to 1.0 to handle steps beyond total_steps

    float cosine_decay = 0.5f * (1.0f + cosf(M_PI * progress)); // Goes from 1 to 0

    // The final LR is now based on the stage_lr, not the initial_lr
    const float final_lr_factor = 0.01f; // Decay to 1% of the STAGE learning rate
    return stage_lr * (final_lr_factor + (1.0f - final_lr_factor) * cosine_decay);
}



int main() {
    // --- Training & NeRF Parameters ---
    const int BATCH_SIZE = 1<<18; //2^20
    const int FGPU_BATCH_SIZE = 1<<14; //2^20
    const int GRAD_ACCUMULATION_STEPS = BATCH_SIZE / FGPU_BATCH_SIZE;
    const int MAX_STEPS = 30000;

    // --- Hierarchical Sampling Configuration for Training
    const int N_COARSE_SAMPLES = 64;
    const int N_FINE_SAMPLES = 128;
    const int N_SAMPLES_PER_RAY_WORKSPACE = N_COARSE_SAMPLES + N_FINE_SAMPLES;

    // --- Training Schedule Constants ---
    const int WARMUP_STEPS = 1000;
    const int PROPOSAL_STEPS = 2000;
    const int OCCUPANCY_GRID_ACTIVATION_STEPS = 4000;
    const int TV_LOSS_END_STEP = 20000;
    const int TV_LOSS_DECAY_STEPS = 2000;
    const int TV_DECAY_START_STEP = TV_LOSS_END_STEP - TV_LOSS_DECAY_STEPS;

    // --- ADAM Optimizer Hyperparameters ---
    const float LR_MLP = 1e-4f;
    const float LR_HASH = 1e-3f;
    const float LR_PROPOSAL = 1e-4f;
    const float GRAD_CLIP_VAL = 0.1f;
    const float BETA1 = 0.9f;
    const float BETA2 = 0.99f;
    const float EPSILON = 1e-10f;
    const float TV_LOSS_WEIGHT = 1e-8f;
    const float L2_REG_WEIGHT = 1e-6f;
    const float DISTORTION_LOSS_WEIGHT = 0.01f;


    // --- Multi-GPU Setup ---
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
        printf("FATAL: No GPUs found.\n");
        return -1;
    }
    printf("--- Found %d GPUs ---\n", n_gpus);
    const int SUB_BATCH_SIZE = FGPU_BATCH_SIZE / n_gpus;

    // --- Initialization ---
    printf("--- Initializing Components ---\n");
    DatasetLoader dataset_loader("./lego");
    if (dataset_loader.camera_params.empty()) return -1;

    std::vector<MLP*> d_mlps(n_gpus);
    std::vector<HashTable*> host_hash_tables(n_gpus);
    std::vector<DeviceHashTableAccessor*> d_table_accs(n_gpus);
    std::vector<OccupancyGrid*> host_grids(n_gpus);
    std::vector<Renderer*> renderers(n_gpus); // NEW: Vector of Renderer objects

    // Buffer vectors...
    std::vector<CudaManagedBuffer<float3>> d_batch_ray_origins_vec(n_gpus);
    std::vector<CudaManagedBuffer<float3>> d_batch_ray_directions_vec(n_gpus);
    std::vector<CudaManagedBuffer<float3>> d_batch_gt_colors_vec(n_gpus);
    std::vector<CudaManagedBuffer<float>> d_batch_near_bounds_vec(n_gpus);
    std::vector<CudaManagedBuffer<float>> d_batch_far_bounds_vec(n_gpus);
    std::vector<CudaManagedBuffer<float4>> d_predicted_pixels_vec(n_gpus);
    std::vector<CudaManagedBuffer<float3>> dL_d_predicted_rgb_vec(n_gpus);
    std::vector<CudaManagedBuffer<float>> d_per_ray_mse_loss_vec(n_gpus);
    std::vector<CudaManagedBuffer<float>> d_total_tv_loss_vec(n_gpus);
    std::vector<CudaManagedBuffer<float>> d_total_dist_loss_vec(n_gpus);

    std::vector<cudaStream_t> compute_streams(n_gpus);
    std::vector<cudaStream_t> comm_streams(n_gpus);
    std::vector<cudaEvent_t> compute_done_events(n_gpus);

    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&compute_streams[i]);
        cudaStreamCreate(&comm_streams[i]);
        cudaEventCreate(&compute_done_events[i]);
    }

    for (int i = 0; i < n_gpus; ++i) {
        printf("--- Initializing GPU %d ---\n", i);
        cudaSetDevice(i);

        // Models
        host_hash_tables[i] = new HashTable(N_LEVELS, N_min, N_max, T_per_level);
        d_table_accs[i] = host_hash_tables[i]->get_device_accessor();
        
        cudaMallocManaged(&d_mlps[i], sizeof(MLP));
        new (d_mlps[i]) MLP(1337ULL + i);

        host_grids[i] = new OccupancyGrid(make_int3(128, 128, 128), 0.95f, 0.01f);
        renderers[i] = new Renderer(); // NEW: Initialize renderer

        // Buffers
        d_batch_ray_origins_vec[i].resize(SUB_BATCH_SIZE);
        d_batch_ray_directions_vec[i].resize(SUB_BATCH_SIZE);
        d_batch_gt_colors_vec[i].resize(SUB_BATCH_SIZE);
        d_batch_near_bounds_vec[i].resize(SUB_BATCH_SIZE);
        d_batch_far_bounds_vec[i].resize(SUB_BATCH_SIZE);
        d_predicted_pixels_vec[i].resize(SUB_BATCH_SIZE);
        dL_d_predicted_rgb_vec[i].resize(SUB_BATCH_SIZE);
        d_per_ray_mse_loss_vec[i].resize(SUB_BATCH_SIZE);
        d_total_tv_loss_vec[i].resize(1);
        d_total_dist_loss_vec[i].resize(1);
    }

    // +++ START OF WARM-UP BLOCK +++
    printf("\n--- Pre-warming device accessors to prevent race conditions ---\n");
    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(i);
        // This call ensures the thread-unsafe, one-time setup is done serially.
        host_hash_tables[i]->get_device_accessor(); 
    }
    printf("--- Warm-up complete ---\n\n");
    // +++ END OF WARM-UP BLOCK +++
    
    // --- Init NCCL and cuBLAS ---
    ncclComm_t comms[n_gpus];
    int gpus_ids[n_gpus];
    for(int i = 0; i < n_gpus; ++i) gpus_ids[i] = i;
    NCCLCHECK(ncclCommInitAll(comms, n_gpus, gpus_ids));

    std::vector<cublasHandle_t> cublas_handles(n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(i);
        cublasCreate(&cublas_handles[i]);
    }

    // --- Scene Bounds ---
    const Vec3 scene_aabb_min = {-1.5f, -1.5f, -1.5f};
    const Vec3 scene_aabb_max = {1.5f, 1.5f, 1.5f};

    std::vector<RendererWorkspace*> workspaces(n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(i);
        workspaces[i] = new RendererWorkspace(SUB_BATCH_SIZE, N_SAMPLES_PER_RAY_WORKSPACE);
    }



    printf("\n--- Starting Training Loop ---\n");

    const float grad_scale_factor = 1.0f / GRAD_ACCUMULATION_STEPS;


    //float n_lr_mlp = LR_MLP;
    //float n_lr_hash = LR_HASH;

    for (int step = 1; step <= MAX_STEPS; ++step) {

        float n_lr_mlp = get_learning_rate(step, LR_MLP, WARMUP_STEPS, MAX_STEPS);
        float n_lr_hash = get_learning_rate(step, LR_HASH, WARMUP_STEPS, MAX_STEPS);

        float current_tv_weight = 0.0f; // Default to 0 after the schedule is done
        if (step <= TV_DECAY_START_STEP) {
            // Before the decay period, use the full weight
            current_tv_weight = TV_LOSS_WEIGHT;
        } else if (step < TV_LOSS_END_STEP) {
            // During the decay period, linearly interpolate the weight from initial to zero
            float progress = (float)(step - TV_DECAY_START_STEP) / (float)TV_LOSS_DECAY_STEPS;
            current_tv_weight = TV_LOSS_WEIGHT * (1.0f - progress);
        }
        // --- Gradient Accumulation Loop ---
        // 1. Zero gradients ONCE before the accumulation loop
        #pragma omp parallel for
        for (int i = 0; i < n_gpus; ++i) {
            cudaSetDevice(i);
            d_mlps[i]->zero_grad(compute_streams[i]);
            host_hash_tables[i]->zero_grad(compute_streams[i]);
        }



        for (int accum_step = 0; accum_step < GRAD_ACCUMULATION_STEPS; ++accum_step) {
            #pragma omp parallel for
            for (int i = 0; i < n_gpus; ++i) {
                cudaSetDevice(i);
                cudaStream_t compute_stream = compute_streams[i];
                cudaStream_t comm_stream = comm_streams[i];
                cublasHandle_t handle = cublas_handles[i];
                cudaEvent_t compute_done_event = compute_done_events[i];

                // --- NEW: Create the workspace for the current batch ---
                RendererWorkspace& workspace = *workspaces[i];


                // Get batch data
                TrainingRayBatch batch = {
                    SUB_BATCH_SIZE, &d_batch_ray_origins_vec[i], &d_batch_ray_directions_vec[i],
                    &d_batch_gt_colors_vec[i], &d_batch_near_bounds_vec[i], &d_batch_far_bounds_vec[i]
                };
                dataset_loader.get_next_batch(batch, i, compute_stream);

                // Calculate ray bounds
                calculate_ray_bounds(
                    handle, SUB_BATCH_SIZE, d_batch_ray_origins_vec[i], d_batch_ray_directions_vec[i],
                    scene_aabb_min, scene_aabb_max, d_batch_near_bounds_vec[i],
                    d_batch_far_bounds_vec[i], compute_stream
                );

                OccupancyGrid* grid_to_use = (step > OCCUPANCY_GRID_ACTIVATION_STEPS) ? host_grids[i] : nullptr;

                renderers[i]->render_forward(
                    handle, compute_stream, 
                    d_table_accs[i], d_mlps[i], grid_to_use,
                    SUB_BATCH_SIZE, d_batch_ray_origins_vec[i].get(), d_batch_ray_directions_vec[i].get(),
                    d_batch_near_bounds_vec[i].get(), d_batch_far_bounds_vec[i].get(),
                    N_COARSE_SAMPLES, N_FINE_SAMPLES, 
                    workspace, d_predicted_pixels_vec[i].get()
                );

                

                cudaMemsetAsync(dL_d_predicted_rgb_vec[i].get(), 0, 
                SUB_BATCH_SIZE * sizeof(float3), compute_stream);
                cudaMemsetAsync(d_per_ray_mse_loss_vec[i].get(), 0, 
                SUB_BATCH_SIZE * sizeof(float), compute_stream);
                // --- Calculate Loss ---
                dim3 grid_dim((SUB_BATCH_SIZE + 255) / 256), block_dim(256);
                calculate_loss_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                    d_predicted_pixels_vec[i].get(), d_batch_gt_colors_vec[i].get(),
                    workspace.d_ray_info.get(), SUB_BATCH_SIZE, dL_d_predicted_rgb_vec[i].get(), 
                    d_per_ray_mse_loss_vec[i].get(),
                    grad_scale_factor
                );


                if (DISTORTION_LOSS_WEIGHT > 0.f) {
                    // 1. Get a pointer to your size-1 buffer and zero it out for this batch.
                    float* d_total_dist_loss_ptr = d_total_dist_loss_vec[i].get();
                    CHECK_CUDA_THROW(cudaMemsetAsync(d_total_dist_loss_ptr, 0, sizeof(float), compute_stream));

                    // 2. Calculate shared memory and launch parameters.
                    const int max_samples_per_ray = N_COARSE_SAMPLES + N_FINE_SAMPLES;
                    size_t shared_mem_size = 4 * max_samples_per_ray * sizeof(float);
                    dim3 grid_dim_fwd(SUB_BATCH_SIZE);
                    dim3 block_dim_fwd(max_samples_per_ray);

                    // 3. Call the EFFICIENT forward kernel.
                    // It will atomically add the loss from all rays into your single float buffer.
                    distortion_loss_fwd_kernel_efficient<<<grid_dim_fwd, block_dim_fwd, shared_mem_size, compute_stream>>>(
                        SUB_BATCH_SIZE, 
                        workspace.d_ray_info.get(), 
                        workspace.d_all_sample_ts.get(),
                        workspace.d_all_sample_dts.get(), 
                        workspace.d_all_weights.get(),
                        d_total_dist_loss_ptr // Correctly passing the pointer to the single float.
                    );
                
                    // --- BACKWARD PASS ---
                    // 4. Call the parallelized backward kernel.
                    dim3 grid_dim_bwd(SUB_BATCH_SIZE);
                    dim3 block_dim_bwd(max_samples_per_ray);
                
                    distortion_loss_bwd_kernel_parallel<<<grid_dim_bwd, block_dim_bwd, 0, compute_stream>>>(
                        SUB_BATCH_SIZE, 
                        workspace.d_ray_info.get(), 
                        workspace.d_all_sample_ts.get(),
                        workspace.d_all_sample_dts.get(), 
                        workspace.d_all_weights.get(),
                        workspace.d_all_raw_densities.get(), 
                        DISTORTION_LOSS_WEIGHT * grad_scale_factor,
                        workspace.dL_d_density.get()
                    );
                }



                // --- High-level backward pass call ---
                renderers[i]->render_backward(
                    handle, compute_stream, d_table_accs[i], d_mlps[i], SUB_BATCH_SIZE,
                    dL_d_predicted_rgb_vec[i].get(), workspace
                );
            }
        }


        #pragma omp parallel for
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            cudaStream_t compute_stream = compute_streams[i];
            cudaStream_t comm_stream = comm_streams[i];
            cublasHandle_t handle = cublas_handles[i];
            cudaEvent_t compute_done_event = compute_done_events[i];

            clip_gradients_by_global_norm(
                d_mlps[i], 
                host_hash_tables[i], 
                GRAD_CLIP_VAL,
                compute_stream
            );
            // --- Synchronize Gradients via NCCL ---
            CHECK_CUDA_THROW(cudaEventRecord(compute_done_event, compute_stream));
            CHECK_CUDA_THROW(cudaStreamWaitEvent(comm_stream, compute_done_event, 0));
            
            // --- 1. Synchronize Gradients via NCCL (EFFICIENT) ---
    
            // Get pointers and sizes for the consolidated buffers
            void* d_mlp_grads_ptr = d_mlps[i]->all_gradients_ptr();
            size_t mlp_grads_count = d_mlps[i]->all_gradients_size_in_elements();

            void* d_hash_grads_ptr = host_hash_tables[i]->gradients();
            size_t hash_grads_count = host_hash_tables[i]->gradients_count_floats();
            
            NCCLCHECK(ncclGroupStart());
            // Call 1: Synchronize the entire MLP gradient buffer
            if (mlp_grads_count > 0) {
                NCCLCHECK(ncclAllReduce((const void*)d_mlp_grads_ptr, (void*)d_mlp_grads_ptr,
                    mlp_grads_count, ncclFloat, ncclSum, comms[i], comm_stream));
            }
        
            // Call 2: Synchronize the entire HashTable gradient buffer
            if (hash_grads_count > 0) {
                NCCLCHECK(ncclAllReduce((const void*)d_hash_grads_ptr, (void*)d_hash_grads_ptr,
                    hash_grads_count, ncclFloat, ncclSum, comms[i], comm_stream));
            }
            NCCLCHECK(ncclGroupEnd());

            // --- Optimizer Step ---
            d_mlps[i]->adam_update(
                n_lr_mlp, BETA1, BETA2, EPSILON, 
                step + 1, L2_REG_WEIGHT, 1,
                GRAD_CLIP_VAL, comm_stream
            );
            host_hash_tables[i]->adam_update(
                n_lr_hash, BETA1, BETA2, EPSILON, 
                step + 1, current_tv_weight, 
                host_grids[i], 
                d_total_tv_loss_vec[i].get(), 
                comm_stream, 1,
                GRAD_CLIP_VAL
            );
        }



        // --- Update Occupancy Grid (on GPU 0) ---
        if (step > OCCUPANCY_GRID_ACTIVATION_STEPS && step % 256 == 0) {
            // 1. Update the grid on GPU 0 as before.
            //    The hard sync here is okay, but can be optimized later with events.
            cudaSetDevice(0);
            CHECK_CUDA_THROW(cudaStreamSynchronize(comm_streams[0]));
            host_grids[0]->update(cublas_handles[0], compute_streams[0], d_table_accs[0], d_mlps[0], step);
            CHECK_CUDA_THROW(cudaStreamSynchronize(compute_streams[0]));

            // 2. Broadcast the updated grid from GPU 0 to all other GPUs.
            //    We use a parallel region to issue the broadcast command to all devices.
            #pragma omp parallel for
            for (int i = 0; i < n_gpus; i++) {
                void* grid_data = host_grids[i]->get_grid_data_ptr();
                size_t grid_size = host_grids[i]->get_grid_data_size();

                NCCLCHECK(ncclBroadcast(
                    grid_data,
                    grid_data,
                    grid_size,
                    ncclChar,
                    0,
                    comms[i],
                    comm_streams[i]
                ));
            }
        };

        // --- Logging ---
        if (step % 10 == 0) {
            cudaSetDevice(0);
            // Wait for all work on GPU 0 to finish before copying data to host for logging
            CHECK_CUDA_THROW(cudaStreamSynchronize(compute_streams[0]));
            CHECK_CUDA_THROW(cudaStreamSynchronize(comm_streams[0]));


            // --- START: DEBUG MLP GRADIENTS ---
            if (step % 10 == 0) { // Only print this extended log once at the first check
                std::vector<float> h_mlp_grads(10);
                CHECK_CUDA_THROW(cudaMemcpy(
                    h_mlp_grads.data(),
                    d_mlps[0]->all_gradients_ptr(), // Pointer to the consolidated MLP gradients
                    10 * sizeof(float),
                    cudaMemcpyDeviceToHost
                ));
            
                printf("\n--- DEBUG: MLP Gradients (First 10) at Step %d ---\n", step);
                for (int k = 0; k < 10; ++k) {
                    printf("  MLP grad[%d] = %f\n", k, h_mlp_grads[k]);
                }
                printf("--------------------------------------------------\n");
            }
            // --- END: DEBUG MLP GRADIENTS ---


            // --- START: CORRECTED DEBUG HASH TABLE GRADIENTS ---
            if (step % 10 == 0) {
                CHECK_CUDA_THROW(cudaStreamSynchronize(compute_streams[0]));
                // Copy a LARGER sample of the gradient buffer to increase our chances of seeing a non-zero value.
                const int n_grads_to_check = 20000;
                std::vector<float2> h_hash_grads(n_grads_to_check);
                CHECK_CUDA_THROW(cudaMemcpy(
                    h_hash_grads.data(),
                    host_hash_tables[0]->gradients(),
                    n_grads_to_check * sizeof(float2),
                    cudaMemcpyDeviceToHost
                ));
            
                printf("\n--- DEBUG: Hash Table Gradients (NON-ZERO ONLY) at Step %d ---\n", step);
                int non_zero_count = 0;
                for (int k = 0; k < n_grads_to_check; ++k) {
                    // FIX: Print ONLY if the gradient is actually non-zero.
                    if (h_hash_grads[k].x != 0.0f || h_hash_grads[k].y != 0.0f) {
                        if (non_zero_count < 10) { // Print the first 10 non-zero grads we find
                            printf("  Hash grad[%d] = (%f, %f)\n", k, h_hash_grads[k].x, h_hash_grads[k].y);
                        }
                        non_zero_count++;
                    }
                }
            
                if (non_zero_count == 0) {
                    printf("  All checked hash gradients in the first %d entries are zero.\n", n_grads_to_check);
                } else {
                    printf("  Found %d non-zero gradients in the first %d entries.\n", non_zero_count, n_grads_to_check);
                }
                printf("----------------------------------------------------------------\n\n");
            }
            // --- END: CORRECTED DEBUG HASH TABLE GRADIENTS ---

            // --- Calculate MSE loss from per-ray buffer ---
            std::vector<float> h_per_ray_mse_loss(SUB_BATCH_SIZE);
            CHECK_CUDA_THROW(cudaMemcpy(
                h_per_ray_mse_loss.data(),
                d_per_ray_mse_loss_vec[0].get(),
                d_per_ray_mse_loss_vec[0].nbytes(),
                cudaMemcpyDeviceToHost
            ));


            float mse_loss = 0.f;
            for (int k = 0; k < SUB_BATCH_SIZE; ++k) {
                mse_loss += h_per_ray_mse_loss[k];
            }
            mse_loss /= SUB_BATCH_SIZE;

            // --- Get TV loss ---
            float h_tv_loss = 0.0f;
            CHECK_CUDA_THROW(cudaMemcpy(
                &h_tv_loss,
                d_total_tv_loss_vec[0].get(),
                sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            // --- Calculate and print total loss ---
            float total_loss = mse_loss + current_tv_weight * h_tv_loss;

            printf("Step %d/%d, Loss: %f (MSE: %f, TV: %.2e), LR(hash, mlp): (%.2e, %.2e)\n",
               step, MAX_STEPS, total_loss, mse_loss, h_tv_loss, n_lr_hash, n_lr_mlp);

            if (isnan(total_loss) || isinf(total_loss)) {
                printf("FATAL: NaN or Inf detected in loss at step %d!\n", step);
                break; // Exit training if loss is unstable
            }
        }
    }

    printf("\nTraining finished.\n");



    for (int i = 0; i < n_gpus; i++ ) {
        d_mlps[i]->release_optimizer_states();
        host_hash_tables[i]->release_optimizer_states();
    }
    cudaSetDevice(0);
    printf("\nPerforming final occupancy grid update for render...\n");
    host_grids[0]->update(cublas_handles[0], compute_streams[0], d_table_accs[0], d_mlps[0], MAX_STEPS + 1);
    CHECK_CUDA_THROW(cudaDeviceSynchronize());

    render_image(
        cublas_handles[0],
        compute_streams[0],
        renderers[0],
        d_table_accs[0],
        d_mlps[0],
        host_grids[0],
        dataset_loader,
        scene_aabb_min,
        scene_aabb_max
    );

    // Cleanup logic remains the same...
    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(i);
        delete renderers[i];
        d_mlps[i]->~MLP();
        cudaFree(d_mlps[i]);
        delete host_hash_tables[i];
        delete host_grids[i];
        ncclCommDestroy(comms[i]);
        cublasDestroy(cublas_handles[i]);
        cudaStreamDestroy(compute_streams[i]);
        cudaStreamDestroy(comm_streams[i]);
        cudaEventDestroy(compute_done_events[i]);
    };

    return 0;
}