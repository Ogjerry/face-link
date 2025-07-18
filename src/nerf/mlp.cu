#include "../../include/nerf/mlp.cuh"
#include "../../include/nerf/hashing.cuh"
#include <curand_kernel.h>
#include <random>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_fp16.h>



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
        grad = 0.0f;  // Reset NaN/inf gradients
    } else if (grad > grad_clip_val) {
        grad = grad_clip_val;
    } else if (grad < -grad_clip_val) {
        grad = -grad_clip_val;
    };
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


    float update = lr * m_hat / (sqrtf(v_hat) + epsilon);

    if (isnan(update) || isinf(update)) {
        update = 0.0f;
    } else if (update > 0.1f) {  // Limit update magnitude
        update = 0.1f;
    } else if (update < -0.1f) {
        update = -0.1f;
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
    MLP* d_mlp,         // device mlp pointer
    int layer_ind, 
    unsigned long long seed
    ) {
    int fan_in, fan_out; // weight matrix dims
    __half* weights_ptr;

    if (layer_ind == 0) {
        fan_in = d_mlp->D_in;
        fan_out = d_mlp->D_hidden;
        weights_ptr = const_cast<__half*>(d_mlp->density_weights1());
    } else if (layer_ind == 1) {
        fan_in = d_mlp->D_hidden;
        fan_out = d_mlp->D_density_out;
        weights_ptr = const_cast<__half*>(d_mlp->density_weights2());
    } else if (layer_ind == 2) {
        fan_in = d_mlp->D_color_in;
        fan_out = d_mlp->D_color_hidden;
        weights_ptr = const_cast<__half*>(d_mlp->color_weights1());
    } else if (layer_ind == 3) {
        fan_in = d_mlp->D_color_hidden;
        fan_out = d_mlp->D_color_hidden;
        weights_ptr = const_cast<__half*>(d_mlp->color_weights2());
    } else if (layer_ind == 4) {
        fan_in = d_mlp->D_color_hidden;
        fan_out = d_mlp->D_color_out;
        weights_ptr = const_cast<__half*>(d_mlp->color_weights3());
    } else {
        return;
    }
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= fan_in * fan_out) return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);

    const float std_dev = sqrtf(2.0f / (float) fan_in);
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


MLP::MLP(unsigned long long seed) :
        // Density network params
        m_density_weights1(D_in * D_hidden),
        m_density_biases1(D_hidden),
        m_density_weights2(D_hidden * D_density_out),
        m_density_biases2(D_density_out),

        // Color network params
        m_color_weights1(D_color_in * D_color_hidden),
        m_color_biases1(D_color_hidden),
        m_color_weights2(D_color_hidden * D_color_hidden),
        m_color_biases2(D_color_hidden),
        m_color_weights3(D_color_hidden * D_color_out),
        m_color_biases3(D_color_out),

        // Density network grads
        m_density_weights1_grad(D_in * D_hidden),
        m_density_biases1_grad(D_hidden),
        m_density_weights2_grad(D_hidden * D_density_out),
        m_density_biases2_grad(D_density_out),

        // Color network grads
        m_color_weights1_grad(D_color_in * D_color_hidden),
        m_color_biases1_grad(D_color_hidden),
        m_color_weights2_grad(D_color_hidden * D_color_hidden),
        m_color_biases2_grad(D_color_hidden),
        m_color_weights3_grad(D_color_hidden * D_color_out),
        m_color_biases3_grad(D_color_out),

        // Adam states for density network
        m_dw1_m(D_in * D_hidden),
        m_dw1_v(D_in * D_hidden),
        m_db1_m(D_hidden),
        m_db1_v(D_hidden),
        m_dw2_m(D_hidden * D_density_out),
        m_dw2_v(D_hidden * D_density_out),
        m_db2_m(D_density_out),
        m_db2_v(D_density_out),

        // Adams for color network
        m_cw1_m(D_color_in * D_color_hidden),
        m_cw1_v(D_color_in * D_color_hidden),
        m_cb1_m(D_color_hidden),
        m_cb1_v(D_color_hidden),
        m_cw2_m(D_color_hidden * D_color_hidden), 
        m_cw2_v(D_color_hidden * D_color_hidden),
        m_cb2_m(D_color_hidden),
        m_cb2_v(D_color_hidden),
        m_cw3_m(D_color_hidden * D_color_out),
        m_cw3_v(D_color_hidden * D_color_out),
        m_cb3_m(D_color_out),
        m_cb3_v(D_color_out)

    {
        printf("View-dependent MLP constructed.\n");

        // Lambda expression to zero out buffers for weights and grads
        auto zero_out_buffer = [](CudaManagedBuffer<float>& buf){
            if (buf.size() == 0) return;
            const int block_size = 256;
            const int grid_size = (block_size + buf.size() - 1) / block_size;
            zero_float_buffer_kernel<<< grid_size, block_size>>>(buf.get(),buf.size() );
        };

        // Zero out all gradient and optimizer state buffers upon creation.
        // This only needs to happen once.
        zero_out_buffer(m_density_weights1_grad); zero_out_buffer(m_density_biases1_grad);
        zero_out_buffer(m_density_weights2_grad); zero_out_buffer(m_density_biases2_grad);
        zero_out_buffer(m_color_weights1_grad); zero_out_buffer(m_color_biases1_grad);
        zero_out_buffer(m_color_weights2_grad); zero_out_buffer(m_color_biases2_grad);
        zero_out_buffer(m_color_weights3_grad); zero_out_buffer(m_color_biases3_grad);

        zero_out_buffer(m_dw1_m); zero_out_buffer(m_dw1_v);
        zero_out_buffer(m_db1_m); zero_out_buffer(m_db1_v);
        zero_out_buffer(m_dw2_m); zero_out_buffer(m_dw2_v);
        zero_out_buffer(m_db2_m); zero_out_buffer(m_db2_v);

        zero_out_buffer(m_cw1_m); zero_out_buffer(m_cw1_v);
        zero_out_buffer(m_cb1_m); zero_out_buffer(m_cb1_v);
        zero_out_buffer(m_cw2_m); zero_out_buffer(m_cw2_v);
        zero_out_buffer(m_cb2_m); zero_out_buffer(m_cb2_v);
        zero_out_buffer(m_cw3_m); zero_out_buffer(m_cw3_v);
        zero_out_buffer(m_cb3_m); zero_out_buffer(m_cb3_v);

        CHECK_CUDA_THROW(cudaDeviceSynchronize());
        printf("Initializing MLP weights and biases..\n");

        const int n_threads = 256;
        const float BIAS_INIT_VAL = 0.0f;

        // --- Density Network Initialization ---
    init_mlp_weights_kernel<<< (D_in * D_hidden + n_threads - 1) / n_threads, n_threads >>>(this, 0, seed);
    init_mlp_biases_kernel<<< (D_hidden + n_threads - 1) / n_threads, n_threads >>>(m_density_biases1.get(), m_density_biases1.size(), BIAS_INIT_VAL);
    init_mlp_weights_kernel<<< (D_hidden * D_density_out + n_threads - 1) / n_threads, n_threads >>>(this, 1, seed + 1);
    init_mlp_biases_kernel<<< (D_density_out + n_threads - 1) / n_threads, n_threads >>>(m_density_biases2.get(), m_density_biases2.size(), BIAS_INIT_VAL);

    // --- REVISED: Color Network Initialization ---
    // Layer 1
    init_mlp_weights_kernel<<< (D_color_in * D_color_hidden + n_threads - 1) / n_threads, n_threads >>>(this, 2, seed + 2);
    init_mlp_biases_kernel<<< (D_color_hidden + n_threads - 1) / n_threads, n_threads >>>(m_color_biases1.get(), m_color_biases1.size(), BIAS_INIT_VAL);
    // Layer 2
    init_mlp_weights_kernel<<< (D_color_hidden * D_color_hidden + n_threads - 1) / n_threads, n_threads >>>(this, 3, seed + 3);
    init_mlp_biases_kernel<<< (D_color_hidden + n_threads - 1) / n_threads, n_threads >>>(m_color_biases2.get(), m_color_biases2.size(), BIAS_INIT_VAL);
    // Output Layer
    init_mlp_weights_kernel<<< (D_color_hidden * D_color_out + n_threads - 1) / n_threads, n_threads >>>(this, 4, seed + 4);
    init_mlp_biases_kernel<<< (D_color_out + n_threads - 1) / n_threads, n_threads >>>(m_color_biases3.get(), m_color_biases3.size(), BIAS_INIT_VAL);


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


// In mlp.cu

void MLP::forward(
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
        m_density_weights1.get(), D_hidden,    // B^T with ldb = D_hidden
        d_input_hash_features, D_in,           // A^T with lda = D_in
        &beta,
        d_hidden1_density, D_hidden                    // C^T with ldc = D_hidden
    );

    dim3 grid_b1(((size_t)n_points * D_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b1, threads, 0, stream>>>(
        d_hidden1_density, m_density_biases1.get(), (size_t)n_points * D_hidden, D_hidden
    );

    // --- 2. Density Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_density_out, n_points, D_hidden,
        &alpha,
        m_density_weights2.get(), D_density_out,
        d_hidden1_density, D_hidden,
        &beta,
        d_density_out_full, D_density_out
    );

    dim3 grid_b2(((size_t)n_points * D_density_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b2, threads, 0, stream>>>(
        d_density_out_full, m_density_biases2.get(), (size_t)n_points * D_density_out, D_density_out
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
        m_color_weights1.get(), D_color_hidden,
        d_color_net_input, D_color_in,
        &beta,
        d_hidden1_color, D_color_hidden
    );

    dim3 grid_b3(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b3, threads, 0, stream>>>(
        d_hidden1_color, m_color_biases1.get(), (size_t)n_points * D_color_hidden, D_color_hidden
    );

    // --- 5. Color Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_color_hidden, n_points, D_color_hidden,
        &alpha,
        m_color_weights2.get(), D_color_hidden,
        d_hidden1_color, D_color_hidden,
        &beta,
        d_hidden2_color, D_color_hidden
    );

    dim3 grid_b4(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    add_bias_kernel<<<grid_b4, threads, 0, stream>>>(
        d_hidden2_color, m_color_biases2.get(), (size_t)n_points * D_color_hidden, D_color_hidden
    );


    // --- 6. Color Net Layer 3 (Output)
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_color_out, n_points, D_color_hidden,
        &alpha,
        m_color_weights3.get(), D_color_out,
        d_hidden2_color, D_color_hidden,
        &beta,
        d_rgb_out, D_color_out
    );
    dim3 grid_b5(((size_t)n_points * D_color_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b5, threads, 0, stream>>>(
        d_rgb_out, m_color_biases3.get(), (size_t)n_points * D_color_out, D_color_out
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
        m_density_weights1.get(), D_hidden,
        d_input_hash_features, D_in,
        &beta,
        d_hidden1.get(), D_hidden);

    dim3 grid_b1(((size_t)n_points * D_hidden + threads - 1) / threads);
    add_bias_and_relu_kernel<<<grid_b1, threads, 0, stream>>>(
        d_hidden1.get(), m_density_biases1.get(), (size_t)n_points * D_hidden, D_hidden);

    // --- 2. Density Net Layer 2 ---
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D_density_out, n_points, D_hidden,
        &alpha,
        m_density_weights2.get(), D_density_out,
        d_hidden1.get(), D_hidden,
        &beta,
        d_density_out_full.get(), D_density_out);

    dim3 grid_b2(((size_t)n_points * D_density_out + threads - 1) / threads);
    add_bias_kernel<<<grid_b2, threads, 0, stream>>>(
        d_density_out_full.get(), m_density_biases2.get(), (size_t)n_points * D_density_out, D_density_out);

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
    __half* grad_in_out, 
    const __half* activations, 
    size_t n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (__hle(activations[idx], __float2half(0.0f))) {
            grad_in_out[idx] = __float2half(0.0f);
        }
    }
};


__global__ void sum_bias_gradients_kernel(
    const __half* d_output_grads, // Incoming gradients [n_points, n_cols]
    float* d_bias_grads,          // Output bias gradients [n_cols]
    int n_points, 
    int n_cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n_cols) {
        float sum = 0.0f;
        for (int i = 0; i < n_points; i++) {
            sum += __half2float(d_output_grads[i * n_cols + col]);
        }
        atomicAdd(&d_bias_grads[col], sum);
    }
};

__global__ void backprop_output_activations_kernel(
    int n_points,
    const __half* dL_d_density,       // from loss function
    const __half* dL_d_color,         // from loss function
    const __half* d_raw_density_out,  // raw output from density net (before softplus)
    const __half* d_raw_rgb_out,      // raw output from color net (before sigmoid)
    __half* d_grad_density,           // output grad for density net
    __half* d_grad_color              // output grad for color net
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    float logit_d = __half2float(d_raw_density_out[idx * MLP::D_density_out]);
    float grad_d = __half2float(dL_d_density[idx]);
    d_grad_density[idx * MLP::D_density_out] = __float2half(grad_d * hsigmoid_derivative(logit_d));

    for (int i = 1; i < MLP::D_density_out; ++i) {
        d_grad_density[idx * MLP::D_density_out + i] = __float2half(0.0f);
    }

    for (int i = 0; i < 3; i++) {
        int grad_idx = idx * 3 + i;
        float logit_c = __half2float(d_raw_rgb_out[grad_idx]);
        float grad_c = __half2float(dL_d_color[grad_idx]);
        float sig_c = hsigmoid(logit_c);
        d_grad_color[grad_idx] = __float2half(grad_c * sig_c * (1.0f - sig_c));
    }
};

__global__ void split_and_add_grads_kernel(
    int n_points,
    const __half* d_grad_color_input, // Gradient coming back from the color network
    __half* d_grad_density_output,    // Gradient for the density network's output (additive)
    __half* d_grad_sh_features        // Gradient for the SH features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    for (int i = 0; i < MLP::D_geo_feat; i++) {
        int density_idx = idx * MLP::D_density_out + (i + 1);
        int color_idx = idx * MLP::D_color_in + i;
        d_grad_density_output[density_idx] = __hadd(d_grad_density_output[density_idx], d_grad_color_input[color_idx]);
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




void MLP::backward(
    cublasHandle_t handle,
    cudaStream_t stream,
    int n_points,
    const __half* dL_d_density,
    const __half* dL_d_color,
    const __half* d_input_hash_features,
    const __half* d_input_sh_features,
    const __half* d_hidden1_density,
    const __half* d_density_out_full,
    const __half* d_color_net_input,
    const __half* d_hidden1_color,
    const __half* d_hidden2_color,
    const __half* d_rgb_out,
    __half* d_grad_hash_features,
    __half* d_grad_sh_features
) {
    cublasSetStream(handle, stream);
    const float alpha_fp32 = 1.0f;
    const float beta_fp32 = 1.0f;
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h_zero = __float2half(0.0f);
    const int threads = 256;

    // Temporary buffers
    CudaDeviceBuffer<__half> d_grad_rgb_out(n_points * D_color_out);
    CudaDeviceBuffer<__half> d_grad_hidden2(n_points * D_color_hidden);
    CudaDeviceBuffer<__half> d_grad_hidden1(n_points * D_color_hidden);
    CudaDeviceBuffer<__half> d_grad_color_input(n_points * D_color_in);
    CudaDeviceBuffer<__half> d_grad_density_out(n_points * D_density_out);
    CudaDeviceBuffer<__half> d_grad_density_hidden(n_points * D_hidden);

    // --- 1. Backprop through final activations
    dim3 grid_out(((size_t)n_points + threads - 1) / threads);
    backprop_output_activations_kernel<<<grid_out, threads, 0, stream>>>(
        n_points, dL_d_density, dL_d_color, d_density_out_full, d_rgb_out,
        d_grad_density_out.get(), d_grad_rgb_out.get()
    );

    // --- 2. Backprop Color Layer 3 (Output Layer) ---
    // Weight gradient: dL/dW_c2 = d_hidden2^T * dL/d_rgb_out
    // Row-major: [D_color_hidden, D_color_out] = [D_color_hidden, n_points] * [n_points, D_color_out]
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D_color_out, D_color_hidden, n_points,
        &alpha_fp32,
        d_grad_rgb_out.get(), CUDA_R_16F, D_color_out,
        d_hidden2_color, CUDA_R_16F, D_color_hidden,
        &beta_fp32,
        m_color_weights3_grad.get(), CUDA_R_32F, D_color_out,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // Input gradient: dL/d_hidden2 = dL/d_rgb_out * weights2^T
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_color_hidden, n_points, D_color_out,
        &alpha_h,
        m_color_weights3.get(), D_color_out,
        d_grad_rgb_out.get(), D_color_out,
        &beta_h_zero,
        d_grad_hidden2.get(), D_color_hidden);

    // --- 3. Backprop Color Layer 2 ---
    dim3 grid_h2(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    backprop_relu_kernel<<<grid_h2, threads, 0, stream>>>(
        d_grad_hidden2.get(), d_hidden2_color, (size_t)n_points * D_color_hidden
    );

    // Weight gradient
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D_color_hidden, D_color_hidden, n_points,
        &alpha_fp32,
        d_grad_hidden2.get(), CUDA_R_16F, D_color_hidden,
        d_hidden1_color, CUDA_R_16F, D_color_hidden,
        &beta_fp32,
        m_color_weights2_grad.get(), CUDA_R_32F, D_color_hidden,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // 
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_color_hidden, n_points, D_color_hidden,
        &alpha_h,
        m_color_weights2.get(), D_color_hidden,
        d_grad_hidden2.get(), D_color_hidden,
        &beta_h_zero,
        d_grad_hidden1.get(), D_color_hidden);


    // --- 4. Backprop Color Layer 1 ---
    dim3 grid_h1_color(((size_t)n_points * D_color_hidden + threads - 1) / threads);
    backprop_relu_kernel<<<grid_h1_color, threads, 0, stream>>>(
        d_grad_hidden1.get(), d_hidden1_color, 
        (size_t)n_points * D_color_hidden
    );

    // Weight gradient
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D_color_hidden, D_color_in, n_points,
        &alpha_fp32,
        d_grad_hidden1.get(), CUDA_R_16F, D_color_hidden,
        d_color_net_input, CUDA_R_16F, D_color_in,
        &beta_fp32,
        m_color_weights1_grad.get(), CUDA_R_32F, D_color_hidden,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // 
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_color_in, n_points, D_color_hidden,
        &alpha_h,
        m_color_weights1.get(), D_color_hidden,
        d_grad_hidden1.get(), D_color_hidden,
        &beta_h_zero,
        d_grad_hidden1.get(), D_color_in); // hidden

    
    // --- 5. Backprop through Concatenation ---
    split_and_add_grads_kernel<<<grid_out, threads, 0, stream>>>(
        n_points, d_grad_color_input.get(), d_grad_density_out.get(), 
        d_grad_sh_features
    );

    // --- 6. Backprop Density Layer 2 ---
    // Weight gradient
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D_density_out, D_hidden, n_points,
        &alpha_fp32,
        d_grad_density_out.get(), CUDA_R_16F, D_density_out,
        d_hidden1_density, CUDA_R_16F, D_hidden,
        &beta_fp32,
        m_density_weights2_grad.get(), CUDA_R_32F, D_density_out,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Input gradient
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_hidden, n_points, D_density_out,
        &alpha_h,
        m_density_weights2.get(), D_density_out,
        d_grad_density_out.get(), D_density_out,
        &beta_h_zero,
        d_grad_density_hidden.get(), D_hidden);

    // --- 7. Backprop Density Layer 1 ---
    dim3 grid_h1_density(((size_t)n_points * D_hidden + threads - 1) / threads);
    backprop_relu_kernel<<<grid_h1_density, threads, 0, stream>>>(
        d_grad_density_hidden.get(), d_hidden1_density, (size_t)n_points * D_hidden);

    // Weight gradient
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D_hidden, D_in, n_points,
        &alpha_fp32,
        d_grad_density_hidden.get(), CUDA_R_16F, D_hidden,
        d_input_hash_features, CUDA_R_16F, D_in,
        &beta_fp32,
        m_density_weights1_grad.get(), CUDA_R_32F, D_hidden,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // Input gradient
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D_in, n_points, D_hidden,
        &alpha_h,
        m_density_weights1.get(), D_hidden,
        d_grad_density_hidden.get(), D_hidden,
        &beta_h_zero,
        d_grad_hash_features, D_in);
}


void MLP::adam_update(
    float lr, 
    float beta1, 
    float beta2, 
    float epsilon, 
    int step,
    float l2_reg_weight,
    int grad_accumulation_steps,
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
            1.0f * grad_accumulation_steps
        );
        //CHECK_CUDA_THROW(cudaDeviceSynchronize());
    };

    update_buf(m_density_weights1, m_density_weights1_grad, m_dw1_m ,m_dw1_v);
    update_buf(m_density_biases1, m_density_biases1_grad, m_db1_m, m_db1_v);
    update_buf(m_density_weights2, m_density_weights2_grad, m_dw2_m, m_dw2_v);
    update_buf(m_density_biases2, m_density_biases2_grad, m_db2_m, m_db2_v);
    update_buf(m_color_weights1, m_color_weights1_grad, m_cw1_m, m_cw1_v);
    update_buf(m_color_biases1, m_color_biases1_grad, m_cb1_m, m_cb1_v);
    update_buf(m_color_weights2, m_color_weights2_grad, m_cw2_m, m_cw2_v);
    update_buf(m_color_biases2, m_color_biases2_grad, m_cb2_m, m_cb2_v);
    update_buf(m_color_weights3, m_color_weights3_grad, m_cw3_m, m_cw3_v);
    update_buf(m_color_biases3, m_color_biases3_grad, m_cb3_m, m_cb3_v);
}



void MLP::zero_grad(cudaStream_t stream) {
    auto zero_out_buffer = [&](CudaManagedBuffer<float>& buf){
        if (buf.size() > 0) {
            const int block_size = 256;
            const int grid_size = (buf.size() + block_size - 1) / block_size;
            zero_float_buffer_kernel<<<grid_size, block_size, 0, stream>>>(buf.get(), buf.size());
        }
    };
    
    zero_out_buffer(m_density_weights1_grad);
    zero_out_buffer(m_density_biases1_grad);
    zero_out_buffer(m_density_weights2_grad);
    zero_out_buffer(m_density_biases2_grad);
    zero_out_buffer(m_color_weights1_grad);
    zero_out_buffer(m_color_biases1_grad);
    zero_out_buffer(m_color_weights2_grad);
    zero_out_buffer(m_color_biases2_grad);
    zero_out_buffer(m_color_weights3_grad);
    zero_out_buffer(m_color_biases3_grad);
}


void MLP::release_optimizer_states() {
    printf("Releasing MLP optimizer states and gradient buffers...\n");

    // --- Release Adam Optimizer States (m) ---
    m_dw1_m.free();
    m_db1_m.free();
    m_dw2_m.free();
    m_db2_m.free();
    m_cw1_m.free();
    m_cb1_m.free();
    m_cw2_m.free();
    m_cb2_m.free();
    m_cw3_m.free();
    m_cb3_m.free();

    // --- Release Adam Optimizer States (v) ---
    m_dw1_v.free();
    m_db1_v.free();
    m_dw2_v.free();
    m_db2_v.free();
    m_cw1_v.free();
    m_cb1_v.free();
    m_cw2_v.free();
    m_cb2_v.free();
    m_cw3_v.free();
    m_cb3_v.free();

    // --- Release Gradient Buffers ---
    m_density_weights1_grad.free();
    m_density_biases1_grad.free();
    m_density_weights2_grad.free();
    m_density_biases2_grad.free();
    m_color_weights1_grad.free();
    m_color_biases1_grad.free();
    m_color_weights2_grad.free();
    m_color_biases2_grad.free();
    m_color_weights3_grad.free();
    m_color_biases3_grad.free();
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
    __half* d_output_grad          // dL/dx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // Derivative of softplus(x) is sigmoid(x).
        // dL/dx = dL/dy * sigmoid(x)
        d_output_grad[idx] = __hmul(d_incoming_grad[idx], hsigmoid(d_raw_logits[idx]));
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
    __half* d_grad_hash_features
) {
    // --- Setup ---
    cublasSetStream(handle, stream);
    const float alpha_fp32 = 1.0f;
    const float beta_fp32 = 1.0f; // Accumulate gradients
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h_zero = __float2half(0.0f);
    const int threads = 256;

    // --- Temporary buffers ---
    CudaDeviceBuffer<__half> d_grad_output_logits(n_points * D_out);
    CudaDeviceBuffer<__half> d_grad_hidden(n_points * D_hidden);

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
    cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_points, D_hidden, D_out,
        &alpha_h,
        d_grad_output_logits.get(), D_out,
        m_weights2.get(), D_out, // FIX: Correct ldb
        &beta_h_zero,
        d_grad_hidden.get(), D_hidden
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
    cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_points, D_in, D_hidden,
        &alpha_h,
        d_grad_hidden.get(), D_hidden,
        m_weights1.get(), D_hidden, // FIX: Correct ldb
        &beta_h_zero,
        d_grad_hash_features, D_in
    );
}

// ===================================================================
// ================= PROPOSAL NETWORK IMPLEMENTATION =================
// ===================================================================