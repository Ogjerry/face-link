#include "../../include/nerf/mlp.cuh"
#include "../../include/nerf/hashing.cuh"
#include <curand_kernel.h>
#include <random>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



// ---------------------------
// ------- CUBLAS HLPER ------
// ---------------------------


__global__ void add_bias_and_relu_kernel(
    __half* matrix, 
    const __half* bias, 
    int total_elements,
    int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) { // Check against total elements
        int col_idx = idx % n_cols;
        matrix[idx] = hrelu(__hadd(matrix[idx], bias[col_idx]));
    }
};

__global__ void add_bias_kernel(
    __half* matrix, 
    const __half* bias, 
    int total_elements,
    int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col_idx = idx % n_cols; 
        matrix[idx] = __hadd(matrix[idx], bias[col_idx]);
    }
};
// ---------------------------
// ------- CUBLAS HLPER ------
// ---------------------------








// Kernel to zero out a float buffer. This is a generic utility.
__global__ void zero_float_buffer_kernel(float* buffer, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buffer[i] = 0.0f;
    }
}



// Adam optimizer kernel specifically for the MLP's parameters.
// It handles FP16 weights (params) and FP32 gradients and momentum buffers.
__global__ void mlp_adam_update_kernel(
    __half* __restrict__ params,      // Network parameters in FP16
    float* __restrict__ grads,// Gradients in FP32
    float* __restrict__ m,          // 1st moment (Adam) in FP32
    float* __restrict__ v,          // 2nd moment (Adam) in FP32
    int n_elements,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int t,
    float l2_reg_weight,
    float grad_unscaler,
    float grad_clip_val
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    // Adam Update logic
    float grad = grads[i] / grad_unscaler;

    float p = __half2float(params[i]);
    float old_p_debug = p;

    // --- L2 Regularization (Weight Decay) ---
    if (l2_reg_weight > 0.0f) {
        grad += l2_reg_weight * p;
    }
    // --- End of L2 Regularization ---

    if (isnan(grad) || isinf(grad)) {
        grad = 0.0f;
    }

    if (isnan(p) || isinf(p)) p = 0.0f;
    if (isnan(m[i]) || isinf(m[i])) m[i] = 0.0f;
    if (isnan(v[i]) || isinf(v[i])) v[i] = 0.0f;


    m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;


    // Bias correction
    float m_hat = m[i] / (1.0f - powf(beta1, t));
    float v_hat = v[i] / (1.0f - powf(beta2, t));


    if (isnan(m_hat) || isinf(m_hat)) m_hat = 0.0f;
    if (isnan(v_hat) || isinf(v_hat)) v_hat = epsilon;


    float update = 0.0f;
    if (v_hat > 0.f) { // a small safe-guard
        update = lr * m_hat / (sqrtf(v_hat) + epsilon);
    }

    p -= update;
    params[i] = __float2half(p);


    // ======================= DEBUGGING CODE =======================
    // Check for NaN/Inf after all calculations are done.
    // This will tell us if this kernel produced a bad value.
    if (isnan(p) || isinf(p)) {
        printf(
            "!!!!!!!! KERNEL ERROR (thread %d) !!!!!!!!\n"
            "PARAM_IDX: %d\n"
            "  - grad:    %f\n"
            "  - m_hat:   %f\n"
            "  - v_hat:   %f\n"
            "  - update:  %f\n"
            "  - old_p:   %f\n"
            "  - new_p:   %f\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
            i, i, grad, m_hat, v_hat, update, old_p_debug, p
        );
    }
    // =============================================================
}






__global__ void init_mlp_weights_kernel(
    __half* weights_ptr,       // Direct pointer to the weights to initialize
    int fan_in,                // The input dimension for the Kaiming init formula
    size_t n_elements,         // Total number of weights in this layer
    unsigned long long seed
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_elements) return;

    // Initialize the random number generator state for this thread
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Kaiming initialization
    const float std_dev = sqrtf(2.0f / (float)fan_in);
    float random_val = curand_normal(&state) * std_dev;

    weights_ptr[idx] = __float2half(random_val);
};



__global__ void init_mlp_biases_kernel(
    __half* biases,
    size_t n_elements,
    float val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        biases[idx] = __float2half(val);
    }
}



/////////////////////////////////////////////////////////////
///////////////// ---------- MLP ---------- /////////////////
/////////////////////////////////////////////////////////////



// ---------------------------
// ------- FUSED KERNELS ------
// ---------------------------
/**
 * @brief Fuses matrix multiplication (input * weights), bias addition, and optional ReLU activation.
 * Each thread computes one element of the output matrix.
 * M: Batch size (n_points)
 * N: Input features
 * K: Output features
 */
__global__ void fused_forward_fully_connected_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int M, int N, int K,
    bool has_relu
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index (M)
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index (K)

    if (row < M && col < K) {
        float sum_fp32 = 0.0f;
        for (int i = 0; i < N; i++) {
            sum_fp32 += __half2float(input[row * N + i]) * __half2float(weights[i * K + col]);
        }

        // Add bias
        sum_fp32 += __half2float(bias[col]);
        __half sum_h = __float2half(sum_fp32);

        if (has_relu) {
            sum_h = hrelu(sum_h);
        }
        output[row * K + col] = sum_h;
    }
}

/**
 * @brief Fuses the entire backward pass for a single fully connected layer.
 * Computes gradients for input, weights, and biases in one go.
 */
__global__ void fused_backward_fully_connected_kernel(
    const __half* __restrict__ grad_output,
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    const __half* __restrict__ activations,
    __half* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    int M, int N, int K,
    bool has_relu
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        __half grad_out_h = grad_output[row * K + col];

        if (has_relu && __half2float(activations[row * K + col]) <= 0.f) {
            grad_out_h = __float2half(0.f);
        }

        float grad_out_f = __half2float(grad_out_h);

        // --- Calculate gradients for weights and biases ---
        // Each thread in the batch (M) contributes to the same weight/bias gradients.
        // atomics are required to prevent race conditions.
        for (int i = 0; i < N; ++i) { // Iterate over input features
            float input_val_f = __half2float(input[row * N + i]);
            // dL/dW_ij = dL/dOut_j * input_i
            atomicAdd(&grad_weights[i * K + col], input_val_f * grad_out_f);
        }
        // dL/dB_j = dL/dOut_j
        atomicAdd(&grad_bias[col], grad_out_f);

        // --- Calculate gradient for the input to this layer ---
        // dL/dIn_i = sum_j(dL/dOut_j * W_ij)
        // Each thread computes one dL/dIn element's contribution from one output_j
        for (int i = 0; i < N; ++i) { // Iterate over input features
            float weight_val_f = __half2float(weights[i * K + col]);
            atomicAdd(&grad_input[row * N + i], __float2half(weight_val_f * grad_out_f));
        }
    }
}


// ---------------------------
// ------- FUSED KERNELS ------
// ---------------------------

__global__ void set_bias_kernel(__half* bias_ptr, float bias_value, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        bias_ptr[i] = __float2half(bias_value);
    }
}

MLP::MLP(unsigned long long seed) {
    printf("Constructing MLP with consolidated buffers...\n");

    // --- 1. Calculate Total Memory Sizes ---
    const size_t size_dw1 = D_in * D_hidden;
    const size_t size_db1 = D_hidden;
    const size_t size_dw2 = D_hidden * D_density_out;
    const size_t size_db2 = D_density_out;
    const size_t size_cw1 = D_color_in * D_color_hidden;
    const size_t size_cb1 = D_color_hidden;
    const size_t size_cw2 = D_color_hidden * D_color_hidden;
    const size_t size_cb2 = D_color_hidden;
    const size_t size_cw3 = D_color_hidden * D_color_out;
    const size_t size_cb3 = D_color_out;

    const size_t total_param_size = size_dw1 + size_db1 + size_dw2 + size_db2 +
                                    size_cw1 + size_cb1 + size_cw2 + size_cb2 +
                                    size_cw3 + size_cb3;
    const size_t total_grad_size = total_param_size;

    // --- 2. Allocate Single Contiguous Buffers ---
    m_params_buffer.resize(total_param_size);
    m_grads_buffer.resize(total_grad_size);
    m_optimizer_m_buffer.resize(total_grad_size);
    m_optimizer_v_buffer.resize(total_grad_size);

    printf("  Allocated %.2f MB for params, %.2f MB for grads, %.2f MB for optimizer states (m+v).\n",
           m_params_buffer.nbytes() / (1024.f * 1024.f),
           m_grads_buffer.nbytes() / (1024.f * 1024.f),
           (m_optimizer_m_buffer.nbytes() + m_optimizer_v_buffer.nbytes()) / (1024.f * 1024.f));

    // --- 3. Slice Buffers by Assigning Pointers ---
    size_t offset = 0;
    
    // Slice Parameter Buffer (__half)
    __half* params_ptr = m_params_buffer.get();
    offset = 0;
    p_density_w1 = params_ptr + offset; offset += size_dw1;
    p_density_b1 = params_ptr + offset; offset += size_db1;
    p_density_w2 = params_ptr + offset; offset += size_dw2;
    p_density_b2 = params_ptr + offset; offset += size_db2;
    p_color_w1   = params_ptr + offset; offset += size_cw1;
    p_color_b1   = params_ptr + offset; offset += size_cb1;
    p_color_w2   = params_ptr + offset; offset += size_cw2;
    p_color_b2   = params_ptr + offset; offset += size_cb2;
    p_color_w3   = params_ptr + offset; offset += size_cw3;
    p_color_b3   = params_ptr + offset;

    // Slice Gradient Buffer (float)
    float* grads_ptr = m_grads_buffer.get();
    offset = 0;
    g_density_w1 = grads_ptr + offset; offset += size_dw1;
    g_density_b1 = grads_ptr + offset; offset += size_db1;
    g_density_w2 = grads_ptr + offset; offset += size_dw2;
    g_density_b2 = grads_ptr + offset; offset += size_db2;
    g_color_w1   = grads_ptr + offset; offset += size_cw1;
    g_color_b1   = grads_ptr + offset; offset += size_cb1;
    g_color_w2   = grads_ptr + offset; offset += size_cw2;
    g_color_b2   = grads_ptr + offset; offset += size_cb2;
    g_color_w3   = grads_ptr + offset; offset += size_cw3;
    g_color_b3   = grads_ptr + offset;

    // Slice Optimizer 'm' Buffer (float)
    float* m_ptr = m_optimizer_m_buffer.get();
    offset = 0;
    m_dw1_m = m_ptr + offset; offset += size_dw1;
    m_db1_m = m_ptr + offset; offset += size_db1;
    m_dw2_m = m_ptr + offset; offset += size_dw2;
    m_db2_m = m_ptr + offset; offset += size_db2;
    m_cw1_m = m_ptr + offset; offset += size_cw1;
    m_cb1_m = m_ptr + offset; offset += size_cb1;
    m_cw2_m = m_ptr + offset; offset += size_cw2;
    m_cb2_m = m_ptr + offset; offset += size_cb2;
    m_cw3_m = m_ptr + offset; offset += size_cw3;
    m_cb3_m = m_ptr + offset;

    // Slice Optimizer 'v' Buffer (float)
    float* v_ptr = m_optimizer_v_buffer.get();
    offset = 0;
    m_dw1_v = v_ptr + offset; offset += size_dw1;
    m_db1_v = v_ptr + offset; offset += size_db1;
    m_dw2_v = v_ptr + offset; offset += size_dw2;
    m_db2_v = v_ptr + offset; offset += size_db2;
    m_cw1_v = v_ptr + offset; offset += size_cw1;
    m_cb1_v = v_ptr + offset; offset += size_cb1;
    m_cw2_v = v_ptr + offset; offset += size_cw2;
    m_cb2_v = v_ptr + offset; offset += size_cb2;
    m_cw3_v = v_ptr + offset; offset += size_cw3;
    m_cb3_v = v_ptr + offset;

    // --- 4. Initialize Buffers ---
    // Zero out all gradient and optimizer state buffers efficiently.
    CHECK_CUDA_THROW(cudaMemset(m_grads_buffer.get(), 0, m_grads_buffer.nbytes()));
    CHECK_CUDA_THROW(cudaMemset(m_optimizer_m_buffer.get(), 0, m_optimizer_m_buffer.nbytes()));
    CHECK_CUDA_THROW(cudaMemset(m_optimizer_v_buffer.get(), 0, m_optimizer_v_buffer.nbytes()));

    printf("Initializing MLP weights and biases...\n");
    const int n_threads = 256;

    // --- 5. Initialize Weights using the New Pointers ---
    
    /// --- Density Network Initialization ---
    init_mlp_weights_kernel<<< (size_dw1 + n_threads - 1) / n_threads, n_threads >>>(
        p_density_w1, D_in, size_dw1, seed);

    init_mlp_biases_kernel<<< (size_db1 + n_threads - 1) / n_threads, n_threads >>>(
        p_density_b1, size_db1, 0.1f);

    init_mlp_weights_kernel<<< (size_dw2 + n_threads - 1) / n_threads, n_threads >>>(
        p_density_w2, D_hidden, size_dw2, seed + 1);

    init_mlp_biases_kernel<<< (size_db2 + n_threads - 1) / n_threads, n_threads >>>(
        p_density_b2, size_db2, 0.0f);


    // --- Color Network Initialization ---
    init_mlp_weights_kernel<<< (size_cw1 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_w1, D_color_in, size_cw1, seed + 2);

    init_mlp_biases_kernel<<< (size_cb1 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_b1, size_cb1, 0.1f);

    init_mlp_weights_kernel<<< (size_cw2 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_w2, D_color_hidden, size_cw2, seed + 3);

    init_mlp_biases_kernel<<< (size_cb2 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_b2, size_cb2, 0.1f);

    init_mlp_weights_kernel<<< (size_cw3 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_w3, D_color_hidden, size_cw3, seed + 4);

    init_mlp_biases_kernel<<< (size_cb3 + n_threads - 1) / n_threads, n_threads >>>(
        p_color_b3, size_cb3, 0.0f);

    
    CHECK_CUDA_THROW(cudaDeviceSynchronize());

    // +++ ADD THIS BLOCK TO SET THE DENSITY BIAS +++
    printf("Applying initial positive bias to density output...\n");
    // Specifically set the bias for the DENSITY output (the first neuron of the last layer) to 0.1
    set_bias_kernel<<<1, 1>>>(
        p_density_b2, // Pointer to the biases of the final density layer
        0.1f,         // The bias value
        1             // Only modify the first element
    );
    // +++ END OF BIAS BLOCK +++    
    
    CHECK_CUDA_THROW(cudaDeviceSynchronize());
    printf("MLP initialization complete.\n");
};





__global__ void prepare_color_input_and_activate_density_kernel(
    int n_points,
    const __half* d_density_out_full,
    const __half* d_input_sh_features,
    __half* d_color_net_input,
    __half* d_out_density
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    // Always update density if the buffer is provided
    if (d_out_density != nullptr) {
        d_out_density[idx] = hsoftplus(d_density_out_full[idx * MLP::D_density_out + 0]);
    }

    // Only update color input if the buffer is provided
    if (d_color_net_input != nullptr && d_input_sh_features != nullptr) {
        for (int i = 0; i < MLP::D_geo_feat; i++) {
            d_color_net_input[idx * MLP::D_color_in + i] = 
                d_density_out_full[idx * MLP::D_density_out + (i + 1)];
        }
        
        for (int i = 0; i < SH_COEFS; i++) {
            d_color_net_input[idx * MLP::D_color_in + MLP::D_geo_feat + i] = 
                d_input_sh_features[idx * SH_COEFS + i];
        }
    }
}

__global__ void apply_sigmoid_kernel(int n_elements, const __half* d_logits, __half* d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        d_output[idx] = hsigmoid(d_logits[idx]);
    }
}




void MLP::forward_cublas(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const __half* d_input_hash_features,
    const __half* d_input_sh_features,
    __half* d_hidden1_density,
    __half* d_density_out_full,
    __half* d_color_net_input,
    __half* d_hidden1_color,
    __half* d_hidden2_color,
    __half* d_rgb_out,
    __half* d_out_density,
    __half* d_out_color
) const {
    cublasSetStream(handle, stream);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    const int threads = 256;

    // --- 1. Density Net Layer 1 ---
    // Row-major: C[n_points, D_hidden] = A[n_points, D_in] * B[D_in, D_hidden]
    // Column-major: C^T[D_hidden, n_points] = B^T[D_hidden, D_in] * A^T[D_in, n_points]
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_hidden, n_points, D_in,
        &alpha,
        p_density_w1, D_hidden,    // B^T with ldb = D_hidden
        d_input_hash_features, D_in,           // A^T with lda = D_in
        &beta,
        d_hidden1_density, D_hidden                    // C^T with ldc = D_hidden
    );

    dim3 grid_b1(((size_t)n_points * D_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b1, threads, 0, stream>>>(
        d_hidden1_density, p_density_b1, (size_t)n_points * D_hidden, D_hidden
    );

    // --- 2. Density Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_density_out, n_points, D_hidden,
        &alpha,
        p_density_w2, D_density_out,
        d_hidden1_density, D_hidden,
        &beta,
        d_density_out_full, D_density_out
    );

    dim3 grid_b2(((size_t)n_points * D_density_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b2, threads, 0, stream>>>(
        d_density_out_full, p_density_b2, (size_t)n_points * D_density_out, D_density_out
    );

    // --- 3. Prepare Color Input & Activate Density ---
    dim3 grid_prep(((size_t)n_points + threads - 1) / threads);
    prepare_color_input_and_activate_density_kernel<<<grid_prep, threads, 0, stream>>>(
        n_points, d_density_out_full, d_input_sh_features,
        d_color_net_input, d_out_density
    );

    // --- 4. Color Net Layer 1 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_color_hidden, n_points, D_color_in,
        &alpha,
        p_color_w1, D_color_hidden,
        d_color_net_input, D_color_in,
        &beta,
        d_hidden1_color, D_color_hidden
    );

    dim3 grid_b3(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b3, threads, 0, stream>>>(
        d_hidden1_color, p_color_b1, (size_t)n_points * D_color_hidden, D_color_hidden
    );

    // --- 5. Color Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_color_hidden, n_points, D_color_hidden,
        &alpha,
        p_color_w2, D_color_hidden,
        d_hidden1_color, D_color_hidden,
        &beta,
        d_hidden2_color, D_color_hidden
    );

    dim3 grid_b4(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    add_bias_kernel<<<grid_b4, threads, 0, stream>>>(
        d_hidden2_color, p_color_b2, (size_t)n_points * D_color_hidden, D_color_hidden
    );


    // --- 6. Color Net Layer 3 (Output)
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_color_out, n_points, D_color_hidden,
        &alpha,
        p_color_w3, D_color_out,
        d_hidden2_color, D_color_hidden,
        &beta,
        d_rgb_out, D_color_out
    );
    dim3 grid_b5(((size_t)n_points * D_color_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b5, threads, 0, stream>>>(
        d_rgb_out, p_color_b3, (size_t)n_points * D_color_out, D_color_out
    );

    // --- 7. Final Color Activation ---
    dim3 grid_sigmoid(((size_t)n_points * D_color_out + threads - 1) / threads);
    apply_sigmoid_kernel<<<grid_sigmoid, threads, 0, stream>>>(
        (size_t)n_points * D_color_out, d_rgb_out, d_out_color
    );
}




__global__ void activate_density_kernel(int n_points, const __half* d_density_out_full, __half* d_out_density) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    // The first element of the density output is the density logit
    d_out_density[idx] = hsoftplus(d_density_out_full[idx * MLP::D_density_out]);
}


void MLP::forward_density(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const __half* d_input_hash_features,
    __half* d_out_density
) const {
    cublasSetStream(handle, stream);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    const int threads = 256;

    // Allocate temporary buffers needed for this pass
    CudaDeviceBuffer<__half> d_hidden1(n_points * MLP::D_hidden);
    CudaDeviceBuffer<__half> d_density_out_full(n_points * MLP::D_density_out);

    // --- 1. Density Net Layer 1 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_hidden, n_points, D_in,
        &alpha,
        p_density_w1, D_hidden,
        d_input_hash_features, D_in,
        &beta,
        d_hidden1.get(), D_hidden);

    dim3 grid_b1(((size_t)n_points * D_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b1, threads, 0, stream>>>(
        d_hidden1.get(), p_density_b1, (size_t)n_points * D_hidden, D_hidden);

    // --- 2. Density Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_density_out, n_points, D_hidden,
        &alpha,
        p_density_w2, D_density_out,
        d_hidden1.get(), D_hidden,
        &beta,
        d_density_out_full.get(), D_density_out);

    dim3 grid_b2(((size_t)n_points * D_density_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b2, threads, 0, stream>>>(
        d_density_out_full.get(), p_density_b2, (size_t)n_points * D_density_out, D_density_out);

    // --- 3. Final Density Activation ---
    // A simplified kernel is needed to just get the density
    // This is a variation of prepare_color_input_and_activate_density_kernel
    const int n_elements = n_points;
    const dim3 grid_size((n_elements + threads - 1) / threads);
    
    
    activate_density_kernel<<<grid_size, threads, 0, stream>>>(
        n_points, d_density_out_full.get(), d_out_density
    );
}







// ---------------------------
// ------- BACKWARD HLP ------
// ---------------------------

__global__ void backprop_relu_kernel(
    float* grad_in_out, 
    const __half* activations, 
    size_t n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (__hle(activations[idx], __float2half(0.0f))) {
            grad_in_out[idx] = 0.0f;
        }
    }
};


__global__ void sum_bias_gradients_kernel(
    const float* d_output_grads, // Incoming gradients [n_points, n_cols]
    float* d_bias_grads,          // Output bias gradients [n_cols]
    int n_points, 
    int n_cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n_cols) {
        float sum = 0.0f;
        for (int i = 0; i < n_points; i++) {
            sum += d_output_grads[i * n_cols + col];
        }
        atomicAdd(&d_bias_grads[col], sum);
    }
};

__global__ void backprop_output_activations_kernel(
    int n_points,
    const float* dL_d_density,       // from loss function
    const float* dL_d_color,         // from loss function
    const __half* d_raw_density_out,  // raw output from density net (before softplus)
    const __half* d_raw_rgb_out,      // raw output from color net (before sigmoid)
    float* d_grad_density,           // output grad for density net
    float* d_grad_color              // output grad for color net
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    float logit_d = __half2float(d_raw_density_out[idx * MLP::D_density_out]);
    float grad_d = dL_d_density[idx];
    d_grad_density[idx * MLP::D_density_out] = grad_d * sigmoid(logit_d);

    for (int i = 1; i < MLP::D_density_out; ++i) {
        d_grad_density[idx * MLP::D_density_out + i] = 0.0f;
    }

    for (int i = 0; i < 3; i++) {
        int grad_idx = idx * 3 + i;
        float logit_c = __half2float(d_raw_rgb_out[grad_idx]);
        float grad_c = dL_d_color[grad_idx];
        float sig_c = sigmoid(logit_c); // half precision notice
        d_grad_color[grad_idx] = grad_c * sig_c * (1.0f - sig_c);
    }
};

__global__ void split_and_add_grads_kernel(
    int n_points,
    const float* d_grad_color_input, // Gradient coming back from the color network
    float* d_grad_density_output,    // Gradient for the density network's output (additive)
    float* d_grad_sh_features        // Gradient for the SH features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    for (int i = 0; i < MLP::D_geo_feat; i++) {
        int density_idx = idx * MLP::D_density_out + (i + 1);
        int color_idx = idx * MLP::D_color_in + i;
        atomicAdd(&d_grad_density_output[density_idx], d_grad_color_input[color_idx]);
    }

    for (int i = 0; i < SH_COEFS; ++i) {
        int sh_idx = idx * SH_COEFS + i;
        int color_idx = idx * MLP::D_color_in + MLP::D_geo_feat + i;
        d_grad_sh_features[sh_idx] = d_grad_color_input[color_idx];
    }
};


// ---------------------------
// ------- BACKWARD HLP ------
// ---------------------------

__global__ void convert_half_to_float_kernel(const __half* in, float* out, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        out[i] = __half2float(in[i]);
    }
}


__global__ void convert_float_to_half_kernel(const float* in, __half* out, size_t n_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        out[i] = __float2half(in[i]);
    }
}

void MLP::backward_cublas(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const float* dL_d_density,
    const float* dL_d_color,
    const __half* d_input_hash_features,
    const __half* d_input_sh_features,
    const __half* d_hidden1_density,
    const __half* d_density_out_full,
    const __half* d_color_net_input,
    const __half* d_hidden1_color,
    const __half* d_hidden2_color,
    const __half* d_rgb_out,
    float* d_grad_hash_features, // Final output
    float* d_grad_sh_features   // Final output
) {
    cublasSetStream(handle, stream);
    const float alpha_fp32 = 1.0f;
    const float beta_fp32_accumulate = 1.0f;
    const float beta_fp32_zero = 0.0f;
    const int threads = 256;

    // --- Workspace buffers for intermediate gradients (float) ---
    float *d_grad_rgb_out, *d_grad_hidden2, *d_grad_hidden1_color, 
          *d_grad_color_input, *d_grad_density_out, *d_grad_density_hidden;
    
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_rgb_out, (size_t)n_points * D_color_out * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_hidden2, (size_t)n_points * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_hidden1_color, (size_t)n_points * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_color_input, (size_t)n_points * D_color_in * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_density_out, (size_t)n_points * D_density_out * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_grad_density_hidden, (size_t)n_points * D_hidden * sizeof(float)));

    // --- Temporary FLOAT buffers for converting HALF inputs ---
    float *p_color_w3_f, *d_hidden2_color_f, *p_color_w2_f, *d_hidden1_color_f,
          *p_color_w1_f, *d_color_net_input_f, *p_density_w2_f, *d_hidden1_density_f,
          *p_density_w1_f, *d_input_hash_features_f;

    CHECK_CUDA_THROW(cudaMalloc(&p_color_w3_f, (size_t)D_color_hidden * D_color_out * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_hidden2_color_f, (size_t)n_points * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&p_color_w2_f, (size_t)D_color_hidden * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_hidden1_color_f, (size_t)n_points * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&p_color_w1_f, (size_t)D_color_in * D_color_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_color_net_input_f, (size_t)n_points * D_color_in * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&p_density_w2_f, (size_t)D_hidden * D_density_out * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_hidden1_density_f, (size_t)n_points * D_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&p_density_w1_f, (size_t)D_in * D_hidden * sizeof(float)));
    CHECK_CUDA_THROW(cudaMalloc(&d_input_hash_features_f, (size_t)n_points * D_in * sizeof(float)));


    // --- 1. Backprop through final activations ---
    dim3 grid_out(((size_t)n_points + threads - 1) / threads);
    backprop_output_activations_kernel<<<grid_out, threads, 0, stream>>>(
        n_points, dL_d_density, dL_d_color, d_density_out_full, d_rgb_out,
        d_grad_density_out, d_grad_rgb_out
    );

    // --- 2. Backprop Color Layer 3 ---
    convert_half_to_float_kernel<<<((size_t)n_points * D_color_hidden + threads - 1)/threads, threads, 0, stream>>>(d_hidden2_color, d_hidden2_color_f, (size_t)n_points * D_color_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        D_color_out, D_color_hidden, n_points, 
        &alpha_fp32,
        d_grad_rgb_out, CUDA_R_32F, D_color_out, 
        d_hidden2_color_f, CUDA_R_32F, D_color_hidden,
        &beta_fp32_accumulate, 
        g_color_w3, CUDA_R_32F, D_color_out, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    
    sum_bias_gradients_kernel<<<(D_color_out + threads - 1) / threads, threads, 0, stream>>>(d_grad_rgb_out, g_color_b3, n_points, D_color_out);

    convert_half_to_float_kernel<<<((size_t)D_color_hidden * D_color_out + threads - 1)/threads, threads, 0, stream>>>(p_color_w3, p_color_w3_f, (size_t)D_color_hidden * D_color_out);
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        D_color_hidden, n_points, D_color_out, 
        &alpha_fp32,
        p_color_w3_f, CUDA_R_32F, D_color_out, 
        d_grad_rgb_out, CUDA_R_32F, D_color_out,
        &beta_fp32_zero, 
        d_grad_hidden2, CUDA_R_32F, D_color_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // --- 3. Backprop Color Layer 2 ---
    dim3 grid_h2(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    backprop_relu_kernel<<<grid_h2, threads, 0, stream>>>(d_grad_hidden2, d_hidden2_color, (size_t)n_points * D_color_hidden);
    
    convert_half_to_float_kernel<<<grid_h2, threads, 0, stream>>>(d_hidden1_color, d_hidden1_color_f, (size_t)n_points * D_color_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        D_color_hidden, D_color_hidden, n_points, 
        &alpha_fp32,
        d_grad_hidden2, CUDA_R_32F, D_color_hidden, 
        d_hidden1_color_f, CUDA_R_32F, D_color_hidden,
        &beta_fp32_accumulate, 
        g_color_w2, CUDA_R_32F, D_color_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    sum_bias_gradients_kernel<<<grid_h2, threads, 0, stream>>>(d_grad_hidden2, g_color_b2, n_points, D_color_hidden);
    
    convert_half_to_float_kernel<<<((size_t)D_color_hidden * D_color_hidden + threads - 1)/threads, threads, 0, stream>>>(p_color_w2, p_color_w2_f, (size_t)D_color_hidden * D_color_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        D_color_hidden, n_points, D_color_hidden, 
        &alpha_fp32,
        p_color_w2_f, CUDA_R_32F, D_color_hidden, 
        d_grad_hidden2, CUDA_R_32F, D_color_hidden,
        &beta_fp32_zero, 
        d_grad_hidden1_color, CUDA_R_32F, D_color_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // --- 4. Backprop Color Layer 1 ---
    backprop_relu_kernel<<<grid_h2, threads, 0, stream>>>(d_grad_hidden1_color, d_hidden1_color, (size_t)n_points * D_color_hidden);

    convert_half_to_float_kernel<<<((size_t)n_points * D_color_in + threads - 1)/threads, threads, 0, stream>>>(d_color_net_input, d_color_net_input_f, (size_t)n_points * D_color_in);
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        D_color_hidden, D_color_in, n_points, 
        &alpha_fp32,
        d_grad_hidden1_color, CUDA_R_32F, D_color_hidden, 
        d_color_net_input_f, CUDA_R_32F, D_color_in,
        &beta_fp32_accumulate, 
        g_color_w1, CUDA_R_32F, D_color_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    sum_bias_gradients_kernel<<<grid_h2, threads, 0, stream>>>(d_grad_hidden1_color, g_color_b1, n_points, D_color_hidden);

    convert_half_to_float_kernel<<<((size_t)D_color_in * D_color_hidden + threads - 1)/threads, threads, 0, stream>>>(p_color_w1, p_color_w1_f, (size_t)D_color_in * D_color_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        D_color_in, n_points, D_color_hidden, 
        &alpha_fp32,
        p_color_w1_f, CUDA_R_32F, D_color_hidden, 
        d_grad_hidden1_color, CUDA_R_32F, D_color_hidden,
        &beta_fp32_zero, 
        d_grad_color_input, CUDA_R_32F, D_color_in, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // --- 5. Backprop through Concatenation ---
    split_and_add_grads_kernel<<<grid_out, threads, 0, stream>>>(
        n_points, d_grad_color_input, d_grad_density_out, d_grad_sh_features);

    // --- 6. Backprop Density Layer 2 ---
    dim3 grid_h1_density(((size_t)n_points * D_hidden + threads - 1) / threads);
    convert_half_to_float_kernel<<<grid_h1_density, threads, 0, stream>>>(d_hidden1_density, d_hidden1_density_f, (size_t)n_points * D_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        D_density_out, D_hidden, n_points, 
        &alpha_fp32,
        d_grad_density_out, CUDA_R_32F, D_density_out, 
        d_hidden1_density_f, CUDA_R_32F, D_hidden,
        &beta_fp32_accumulate, 
        g_density_w2, CUDA_R_32F, D_density_out, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    sum_bias_gradients_kernel<<<(D_density_out + threads - 1) / threads, threads, 0, stream>>>(d_grad_density_out, g_density_b2, n_points, D_density_out);

    convert_half_to_float_kernel<<<((size_t)D_hidden * D_density_out + threads - 1)/threads, threads, 0, stream>>>(p_density_w2, p_density_w2_f, (size_t)D_hidden * D_density_out);
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        D_hidden, n_points, D_density_out, 
        &alpha_fp32,
        p_density_w2_f, CUDA_R_32F, D_density_out, 
        d_grad_density_out, CUDA_R_32F, D_density_out,
        &beta_fp32_zero, 
        d_grad_density_hidden, CUDA_R_32F, D_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    // --- 7. Backprop Density Layer 1 ---
    backprop_relu_kernel<<<grid_h1_density, threads, 0, stream>>>(d_grad_density_hidden, d_hidden1_density, (size_t)n_points * D_hidden);
    
    convert_half_to_float_kernel<<<((size_t)n_points * D_in + threads - 1)/threads, threads, 0, stream>>>(d_input_hash_features, d_input_hash_features_f, (size_t)n_points * D_in);
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        D_hidden, D_in, n_points, 
        &alpha_fp32,
        d_grad_density_hidden, CUDA_R_32F, D_hidden, 
        d_input_hash_features_f, CUDA_R_32F, D_in,
        &beta_fp32_accumulate, 
        g_density_w1, CUDA_R_32F, D_hidden, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    sum_bias_gradients_kernel<<<(D_hidden + threads - 1) / threads, threads, 0, stream>>>(d_grad_density_hidden, g_density_b1, n_points, D_hidden);

    convert_half_to_float_kernel<<<((size_t)D_in * D_hidden + threads - 1)/threads, threads, 0, stream>>>(p_density_w1, p_density_w1_f, (size_t)D_in * D_hidden);
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        D_in, n_points, D_hidden, 
        &alpha_fp32,
        p_density_w1_f, CUDA_R_32F, D_hidden, 
        d_grad_density_hidden, CUDA_R_32F, D_hidden,
        &beta_fp32_zero, 
        d_grad_hash_features, CUDA_R_32F, D_in, 
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                 
    // --- Cleanup: Manually free all allocated workspace and temporary buffers ---
    CHECK_CUDA_THROW(cudaFree(d_grad_rgb_out));
    CHECK_CUDA_THROW(cudaFree(d_grad_hidden2));
    CHECK_CUDA_THROW(cudaFree(d_grad_hidden1_color));
    CHECK_CUDA_THROW(cudaFree(d_grad_color_input));
    CHECK_CUDA_THROW(cudaFree(d_grad_density_out));
    CHECK_CUDA_THROW(cudaFree(d_grad_density_hidden));
    
    CHECK_CUDA_THROW(cudaFree(p_color_w3_f));
    CHECK_CUDA_THROW(cudaFree(d_hidden2_color_f));
    CHECK_CUDA_THROW(cudaFree(p_color_w2_f));
    CHECK_CUDA_THROW(cudaFree(d_hidden1_color_f));
    CHECK_CUDA_THROW(cudaFree(p_color_w1_f));
    CHECK_CUDA_THROW(cudaFree(d_color_net_input_f));
    CHECK_CUDA_THROW(cudaFree(p_density_w2_f));
    CHECK_CUDA_THROW(cudaFree(d_hidden1_density_f));
    CHECK_CUDA_THROW(cudaFree(p_density_w1_f));
    CHECK_CUDA_THROW(cudaFree(d_input_hash_features_f));
}


void MLP::adam_update(
    float lr, 
    float beta1, 
    float beta2, 
    float epsilon, 
    int step,
    float l2_reg_weight,
    int grad_accumulation_steps,
    float grad_clip_val,
    cudaStream_t stream
) {
    auto update_buf = [&](
        CudaManagedBuffer<__half>& params, CudaManagedBuffer<float>& grads,
        CudaManagedBuffer<float>& m, CudaManagedBuffer<float>& v
    ) {
        if (params.size() == 0) return;
        const int block_size = 256;
        const int grid_size = (params.size() + block_size - 1) / block_size;


        mlp_adam_update_kernel<<< grid_size, block_size, 0, stream >>>(
            params.get(),
            grads.get(),
            m.get(),
            v.get(),
            params.size(),
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            l2_reg_weight,
            1.0f * grad_accumulation_steps,
            grad_clip_val
        );
        //CHECK_CUDA_THROW(cudaDeviceSynchronize());
    };

}



void MLP::zero_grad(cudaStream_t stream) {
    auto zero_out_buffer = [&](CudaManagedBuffer<float>& buf){
        if (buf.size() > 0) {
            const int block_size = 256;
            const int grid_size = (buf.size() + block_size - 1) / block_size;
            zero_float_buffer_kernel<<<grid_size, block_size, 0, stream>>>(buf.get(), buf.size());
        }
    };

}


void MLP::release_optimizer_states() {
    printf("Releasing MLP optimizer states and gradient buffers...\n");

    // --- Release Adam Optimizer States (m) ---
    m_grads_buffer.free();
    // --- Release Adam Optimizer States (v) ---
    m_optimizer_m_buffer.free();
    // --- Release Gradient Buffers ---
    m_optimizer_v_buffer.free();
}




/////////////////////////////////////////////////////////////
///////////////// ---------- MLP ---------- /////////////////
/////////////////////////////////////////////////////////////



// ===================================================================
// ================= PROPOSAL NETWORK IMPLEMENTATION =================
// ===================================================================



__global__ void init_proposal_weights_kernel(__half* weights, int fan_in, int fan_out, unsigned long long seed) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= fan_in * fan_out) return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);

    const float std_dev = sqrtf(2.0f / (float)fan_in);
    float val = curand_normal(&state) * std_dev;
    weights[idx] = __float2half(val);
}




ProposalNetwork::ProposalNetwork(unsigned long long seed) :
    m_weights1(D_in * D_hidden), m_biases1(D_hidden),
    m_weights2(D_hidden * D_out), m_biases2(D_out),
    m_weights1_grad(D_in * D_hidden), m_biases1_grad(D_hidden),
    m_weights2_grad(D_hidden * D_out), m_biases2_grad(D_out),
    m_w1_m(D_in * D_hidden), m_w1_v(D_in * D_hidden),
    m_b1_m(D_hidden), m_b1_v(D_hidden),
    m_w2_m(D_hidden * D_out), m_w2_v(D_hidden * D_out),
    m_b2_m(D_out), m_b2_v(D_out)
{
    printf("Proposal Network constructed.\n");

    auto zero_out = [](CudaManagedBuffer<float>& buf) {
        if (buf.size() > 0) cudaMemset(buf.get(), 0, buf.nbytes());
    };

    // Zero out all gradient and optimizer state buffers
    zero_out(m_weights1_grad); zero_out(m_biases1_grad);
    zero_out(m_weights2_grad); zero_out(m_biases2_grad);
    zero_out(m_w1_m); zero_out(m_w1_v);
    zero_out(m_b1_m); zero_out(m_b1_v);
    zero_out(m_w2_m); zero_out(m_w2_v);
    zero_out(m_b2_m); zero_out(m_b2_v);

    printf("Initializing Proposal Network weights and biases...\n");
    const int n_threads = 256;
    
    // Init Layer 1
    dim3 grid_dim_w1((D_in * D_hidden + n_threads - 1) / n_threads);
    init_proposal_weights_kernel<<<grid_dim_w1, n_threads>>>(m_weights1.get(), D_in, D_hidden, seed);
    cudaMemset(m_biases1.get(), 0, m_biases1.nbytes());

    // Init Layer 2
    dim3 grid_dim_w2((D_hidden * D_out + n_threads - 1) / n_threads);
    init_proposal_weights_kernel<<<grid_dim_w2, n_threads>>>(m_weights2.get(), D_hidden, D_out, seed + 1);
    cudaMemset(m_biases2.get(), 0, m_biases2.nbytes());

    CHECK_CUDA_THROW(cudaDeviceSynchronize());
    printf("Proposal Network initialization complete.\n");
}

void ProposalNetwork::adam_update_and_zero_grad(
    float lr, 
    float beta1, float beta2, 
    float epsilon, int step, float l2_reg_weight, float grad_unscaler, cudaStream_t stream
) {
    auto update_buf = [&]( CudaManagedBuffer<__half>& params, CudaManagedBuffer<float>& grads,
                           CudaManagedBuffer<float>& m, CudaManagedBuffer<float>& v) {
        if (params.size() == 0) return;
        const int n_threads = 256;
        const int grid_size = (params.size() + n_threads - 1) / n_threads;
        mlp_adam_update_kernel<<<grid_size, n_threads, 0, stream>>>(
            params.get(), grads.get(), m.get(), v.get(),
            params.size(), lr, beta1, beta2, epsilon, step, l2_reg_weight, grad_unscaler
        );
    };

    update_buf(m_weights1, m_weights1_grad, m_w1_m, m_w1_v);
    update_buf(m_biases1, m_biases1_grad, m_b1_m, m_b1_v);
    update_buf(m_weights2, m_weights2_grad, m_w2_m, m_w2_v);
    update_buf(m_biases2, m_biases2_grad, m_b2_m, m_b2_v);
}


__global__ void forwardprop_softplus_kernel(int n_elements, __half* d_in_out)
 {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        d_in_out[idx] = hsoftplus(d_in_out[idx]);
    }
};



// In mlp.cu
void ProposalNetwork::forward(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const __half* d_inputs,
    __half* d_outputs
) const {
    cublasSetStream(handle, stream);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    const int threads = 256;

    // Temporary buffer for hidden layer activations
    CudaDeviceBuffer<__half> d_hidden(n_points * D_hidden);

    // --- Layer 1: C[n_points, D_hidden] = A[n_points, D_in] * B[D_in, D_hidden] ---
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                D_hidden, n_points, D_in,
                &alpha,
                m_weights1.get(), D_hidden,
                d_inputs, D_in,
                &beta,
                d_hidden.get(), D_hidden);


    // Add bias and apply ReLU
    dim3 grid_b1(((size_t)n_points * D_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b1, threads, 0, stream>>>(
        d_hidden.get(), m_biases1.get(), (size_t)n_points * D_hidden, D_hidden
    );

    // --- Layer 2: C[n_points, D_out] = A[n_points, D_hidden] * B[D_hidden, D_out] ---
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                D_out, n_points, D_hidden,
                &alpha,
                m_weights2.get(), D_out,
                d_hidden.get(), D_hidden,
                &beta,
                d_outputs, D_out);

    dim3 grid_b2(((size_t)n_points * D_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b2, threads, 0, stream>>>(
        d_outputs, m_biases2.get(), (size_t)n_points * D_out, D_out
    );

    // --- Final Activation ---
    dim3 grid_act(((size_t)n_points * D_out + threads - 1) / threads);
    forwardprop_softplus_kernel<<<grid_act, threads, 0, stream>>>(
        (size_t)n_points * D_out, 
        d_outputs
    );
}



__global__ void backprop_softplus_kernel(
    int n_elements,
    const __half* d_incoming_grad, // dL/dy
    const __half* d_raw_logits,    // x (input to softplus)
    float* d_output_grad          // dL/dx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // Derivative of softplus(x) is sigmoid(x).
        // dL/dx = dL/dy * sigmoid(x)
        d_output_grad[idx] = __half2float(__hmul(d_incoming_grad[idx], hsigmoid(d_raw_logits[idx])));
    }
}


// In mlp.cu

void ProposalNetwork::backward(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const __half* dL_d_density,
    const __half* d_inputs,
    const __half* d_hidden_activations,
    const __half* d_output_logits,
    float* d_grad_hash_features
) {
    // --- Setup ---
    cublasSetStream(handle, stream);
    const float alpha_fp32 = 1.0f;
    const float beta_fp32 = 1.0f; // Accumulate gradients
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h_zero = __float2half(0.0f);
    const int threads = 256;

    // --- Temporary buffers ---
    CudaDeviceBuffer<float> d_grad_output_logits(n_points * D_out);
    CudaDeviceBuffer<float> d_grad_hidden(n_points * D_hidden);

    // --- 1. Backpropagate through final Softplus Activation ---
    dim3 grid_act(((size_t)n_points * D_out + threads - 1) / threads);
    backprop_softplus_kernel<<<grid_act, threads, 0, stream>>>(
        (size_t)n_points * D_out, dL_d_density, d_output_logits,
        d_grad_output_logits.get()
    );

    // --- 2. Backpropagate Layer 2 ---
    // C[D_hidden, D_out] = A[n_points, D_hidden]^T * B[n_points, D_out]
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_hidden, D_out, n_points,
        &alpha_fp32,
        d_hidden_activations, CUDA_R_16F, D_hidden,
        d_grad_output_logits.get(), CUDA_R_16F, D_out,
        &beta_fp32,
        m_weights2_grad.get(), CUDA_R_32F, D_out,       // FIX: Correct ldc
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    dim3 grid_b2((D_out + threads - 1) / threads);
    sum_bias_gradients_kernel<<<grid_b2, threads, 0, stream>>>(
        d_grad_output_logits.get(), m_biases2_grad.get(), n_points, D_out
    );

    // dL/d_hidden = dL/d_logits * weights2^T
    // C[n_points, D_hidden] = A[n_points, D_out] * B[D_hidden, D_out]^T
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_points, D_hidden, D_out,
        &alpha_h,
        d_grad_output_logits.get(), CUDA_R_16F, D_out,
        m_weights2.get(), CUDA_R_32F, D_out, // FIX: Correct ldb
        &beta_h_zero,
        d_grad_hidden.get(), CUDA_R_32F, D_hidden,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    // --- 3. Backpropagate through Layer 1 Activation (ReLU) ---
    dim3 grid_relu_bwd(((size_t)n_points * D_hidden + threads - 1) / threads);
    backprop_relu_kernel<<<grid_relu_bwd, threads, 0, stream>>>(
        d_grad_hidden.get(), d_hidden_activations, (size_t)n_points * D_hidden
    );

    // --- 4. Backpropagate Layer 1 ---
    // C[D_in, D_hidden] = A[n_points, D_in]^T * B[n_points, D_hidden]
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_in, D_hidden, n_points,
        &alpha_fp32,
        d_inputs, CUDA_R_16F, D_in,
        d_grad_hidden.get(), CUDA_R_16F, D_hidden,
        &beta_fp32,
        m_weights1_grad.get(), CUDA_R_32F, D_hidden,           // FIX: Correct ldc
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    dim3 grid_b1((D_hidden + threads - 1) / threads);
    sum_bias_gradients_kernel<<<grid_b1, threads, 0, stream>>>(
        d_grad_hidden.get(), m_biases1_grad.get(), n_points, D_hidden
    );
    // dL/d_inputs = dL/d_hidden * weights1^T
    // C[n_points, D_in] = A[n_points, D_hidden] * B[D_in, D_hidden]^T
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_points, D_in, D_hidden,
        &alpha_fp32,
        d_grad_hidden.get(), CUDA_R_16F, D_hidden,
        m_weights1.get(), CUDA_R_32F, D_hidden, // FIX: Correct ldb
        &beta_h_zero,
        d_grad_hash_features, CUDA_R_32F, D_in,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

// ===================================================================
// ================= PROPOSAL NETWORK IMPLEMENTATION =================
// ===================================================================