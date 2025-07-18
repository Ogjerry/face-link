#include "include/nerf/hashing.cuh"
#include "include/nerf/mlp.cuh"
#include "include/nerf/renderer.cuh"
#include "include/nerf/occupancy_grid.cuh"
#include "include/dataset/dataset.cuh"
#include "include/dataset/vec3.h"
#include "include/dataset/camera.cuh"


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
    
    // Create workspace for batch processing
    RendererWorkspace workspace(render_batch_size, N_MAIN_SAMPLES);
    
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
            workspace, d_batch_output
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
float get_learning_rate(int step, float initial_lr) {
    if (step < 20000) {
        // Warmup
        return initial_lr * (float)step / 500.0f;
    } else if (step < 25000) {
        return initial_lr;
    } else {
        // Cosine decay
        float progress = (float)(step - 5000) / 5000.0f;
        return initial_lr * 0.1f + initial_lr * 0.9f * (1.0f + cosf(M_PI * progress)) / 2.0f;
    }
}



int main() {
    // --- Training & NeRF Parameters ---
    const int BATCH_SIZE = 1<<18; //2^20
    const int FGPU_BATCH_SIZE = 1<<14; //2^20
    const int GRAD_ACCUMULATION_STEPS = BATCH_SIZE / FGPU_BATCH_SIZE;
    const int MAX_STEPS = 30000;

    // --- Training Schedule Constants ---
    const int WARMUP_STEPS = 512;
    const int PROPOSAL_STEPS = 2000;
    const int OCCUPANCY_GRID_ACTIVATION_STEPS = 1024;

    // --- ADAM Optimizer Hyperparameters ---
    const float LR_MLP = 1e-2f;
    const float LR_PROPOSAL = 1e-4f;
    const float LR_HASH = 1e-2f;
    // const float FINAL_LR_FACTOR = 0.1f; // Factor to reduce LR for fine-tuning
    const float BETA1 = 0.9f;
    const float BETA2 = 0.99f;
    const float EPSILON = 1e-15f;
    const float TV_LOSS_WEIGHT = 0.0f;
    const float L2_REG_WEIGHT = 1e-6f;


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

        host_grids[i] = new OccupancyGrid(make_int3(128, 128, 128), 0.95f, 0.5f);
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
    }
    
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
        workspaces[i] = new RendererWorkspace(SUB_BATCH_SIZE, N_MAIN_SAMPLES);
    }

    // DEBUG
    // for (int i = 0; i < n_gpus; ++i) {
    //     printf("POST-ALLOCATION [GPU %d]: ws->d_all_hash_features pointer is %p\n", i, workspaces[i]->d_all_hash_features.get());
    // }
    // DEBUG

    printf("\n--- Starting Training Loop ---\n");
    float n_lr_mlp = LR_MLP;
    float n_lr_hash = LR_HASH;

    for (int step = 1; step <= MAX_STEPS; ++step) {
        if (step == 15000 || step == 22500 || step == 27500) {
            n_lr_mlp *= 0.33f;
            n_lr_hash *= 0.33f;
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

                // DEBUG
                // printf("[GPU %d] PRE-KERNEL LAUNCH: ws.d_all_hash_features pointer is %p\n", i, workspace.d_all_hash_features.get());
                // DEBUG

                // Get batch data
                TrainingRayBatch batch = {
                    SUB_BATCH_SIZE, &d_batch_ray_origins_vec[i], &d_batch_ray_directions_vec[i],
                    &d_batch_gt_colors_vec[i], &d_batch_near_bounds_vec[i], &d_batch_far_bounds_vec[i]
                };
                dataset_loader.get_next_batch(batch, i);

                // Calculate ray bounds
                calculate_ray_bounds(
                    handle, SUB_BATCH_SIZE, d_batch_ray_origins_vec[i], d_batch_ray_directions_vec[i],
                    scene_aabb_min, scene_aabb_max, d_batch_near_bounds_vec[i],
                    d_batch_far_bounds_vec[i], compute_stream
                );

                OccupancyGrid* grid_to_use = (step > OCCUPANCY_GRID_ACTIVATION_STEPS) ? host_grids[i] : nullptr;


                // DEBUG
                // printf("[GPU %d, Step %d] DEBUG: Starting forward pass.\n", i, step); // ADD THIS LINE
                // CHECK_CUDA_THROW(cudaDeviceSynchronize()); // Add a sync to be sure
                // DEBUG

                // --- NEW: High-level forward pass call ---
                renderers[i]->render_forward(
                    handle, compute_stream, d_table_accs[i], d_mlps[i], grid_to_use,
                    SUB_BATCH_SIZE, d_batch_ray_origins_vec[i].get(), d_batch_ray_directions_vec[i].get(),
                    d_batch_near_bounds_vec[i].get(), d_batch_far_bounds_vec[i].get(),
                    workspace, d_predicted_pixels_vec[i].get()
                );

                // DEBUG
                // printf("[GPU %d, Step %d] DEBUG: Forward pass finished. Starting NCCL sync.\n", i, step); // ADD THIS LINE
                // CHECK_CUDA_THROW(cudaDeviceSynchronize()); // Add a sync to be sure
                // DEBUG

                // --- Calculate Loss ---
                dim3 grid_dim((SUB_BATCH_SIZE + 255) / 256), block_dim(256);
                calculate_loss_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                    d_predicted_pixels_vec[i].get(), d_batch_gt_colors_vec[i].get(), SUB_BATCH_SIZE,
                    dL_d_predicted_rgb_vec[i].get(), d_per_ray_mse_loss_vec[i].get()
                );

                // DEBUG
                // int device_id;
                // cudaGetDevice(&device_id);
                // printf("GPU %d: Calling render_forward with %d rays\n", device_id, SUB_BATCH_SIZE);
                // DEBUG



                // --- NEW: High-level backward pass call ---
                renderers[i]->render_backward(
                    handle, compute_stream, d_table_accs[i], d_mlps[i], SUB_BATCH_SIZE,
                    dL_d_predicted_rgb_vec[i].get(), workspace
                );

                // --- Synchronize Gradients via NCCL ---
                CHECK_CUDA_THROW(cudaEventRecord(compute_done_event, compute_stream));
                CHECK_CUDA_THROW(cudaStreamWaitEvent(comm_stream, compute_done_event, 0));

                NCCLCHECK(ncclGroupStart());
                // MLP Gradients
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->density_weights1_grad_ptr(), (void*)d_mlps[i]->density_weights1_grad_ptr(), d_mlps[i]->density_weights1_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->density_biases1_grad_ptr(), (void*)d_mlps[i]->density_biases1_grad_ptr(), d_mlps[i]->density_biases1_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->density_weights2_grad_ptr(), (void*)d_mlps[i]->density_weights2_grad_ptr(), d_mlps[i]->density_weights2_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->density_biases2_grad_ptr(), (void*)d_mlps[i]->density_biases2_grad_ptr(), d_mlps[i]->density_biases2_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_weights1_grad_ptr(), (void*)d_mlps[i]->color_weights1_grad_ptr(), d_mlps[i]->color_weights1_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_biases1_grad_ptr(), (void*)d_mlps[i]->color_biases1_grad_ptr(), d_mlps[i]->color_biases1_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_weights2_grad_ptr(), (void*)d_mlps[i]->color_weights2_grad_ptr(), d_mlps[i]->color_weights2_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_biases2_grad_ptr(), (void*)d_mlps[i]->color_biases2_grad_ptr(), d_mlps[i]->color_biases2_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_weights3_grad_ptr(), (void*)d_mlps[i]->color_weights3_grad_ptr(), d_mlps[i]->color_weights3_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclAllReduce((const void*)d_mlps[i]->color_biases3_grad_ptr(), (void*)d_mlps[i]->color_biases3_grad_ptr(), d_mlps[i]->color_biases3_grad_size(), ncclFloat, ncclSum, comms[i], comm_stream));
                // Hash Table Gradients
                NCCLCHECK(ncclAllReduce((const void*)host_hash_tables[i]->gradients(), (void*)host_hash_tables[i]->gradients(), host_hash_tables[i]->gradients_count_floats(), ncclFloat, ncclSum, comms[i], comm_stream));
                NCCLCHECK(ncclGroupEnd());

                // DEBUG
                // printf("[GPU %d, Step %d] DEBUG: NCCL sync finished. Starting MLP optimizer.\n", i, step); // ADD THIS LINE
                // CHECK_CUDA_THROW(cudaDeviceSynchronize()); // Add a sync to be sure
                // DEBUG

                // --- Optimizer Step ---
                d_mlps[i]->adam_update(
                    n_lr_mlp, BETA1, BETA2, EPSILON, 
                    step + 1, L2_REG_WEIGHT, GRAD_ACCUMULATION_STEPS, comm_stream
                );

                // DEBUG
                // printf("[GPU %d, Step %d] DEBUG: MLP optimizer finished. Starting Hash Table optimizer.\n", i, step); // ADD THIS LINE
                // CHECK_CUDA_THROW(cudaDeviceSynchronize()); // Add a sync to be sure
                // DEBUG

                host_hash_tables[i]->adam_update(
                    n_lr_hash, BETA1, BETA2, EPSILON, 
                    step + 1, TV_LOSS_WEIGHT, 
                    host_grids[i], 
                    d_total_tv_loss_vec[i].get(), 
                    comm_stream, GRAD_ACCUMULATION_STEPS
                );
                // DEBUG
                // printf("[GPU %d, Step %d] DEBUG: Hash Table optimizer finished. End of step.\n", i, step); // ADD THIS LINE
                // CHECK_CUDA_THROW(cudaDeviceSynchronize()); // The final and most important check
                // DEBUG
            }
        }



        // --- Update Occupancy Grid (on GPU 0) ---
        if (step > OCCUPANCY_GRID_ACTIVATION_STEPS && step % 16 == 0) {
            cudaSetDevice(0);
            CHECK_CUDA_THROW(cudaStreamSynchronize(comm_streams[0]));
            host_grids[0]->update(cublas_handles[0], compute_streams[0], d_table_accs[0], d_mlps[0], step);
        };

        // --- Logging ---
        if (step % 10 == 0) {
            cudaSetDevice(0);
            // Wait for all work on GPU 0 to finish before copying data to host for logging
            CHECK_CUDA_THROW(cudaStreamSynchronize(compute_streams[0]));
            CHECK_CUDA_THROW(cudaStreamSynchronize(comm_streams[0]));

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
            float total_loss = mse_loss + TV_LOSS_WEIGHT * h_tv_loss;

            printf("Step %d/%d, Loss: %f (MSE: %f, TV: %.2e), LR(hash): %.2e\n",
               step, MAX_STEPS, total_loss, mse_loss, h_tv_loss, n_lr_hash);

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