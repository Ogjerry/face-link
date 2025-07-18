#pragma once

#define MLP_h
#ifdef  MLP_h

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "nerf_config.cuh"

#include <string>
#include "../common/cuda_wrappers.h"
//#include "hashing.cuh"
#include "occupancy_grid.cuh"
#include <curand_kernel.h>
#include "../common/math_utils.h"

/////////////////////////////////////////////////////////////////////////
// ------------- Multiple Layer Perceptron ------------- //
/////////////////////////////////////////////////////////////////////////

class MLP;
struct DeviceHashTableAccessor;

__global__ void init_mlp_biases_kernel(
    float* biases,
    size_t n_elements,
    float val
);

__global__ void init_mlp_weights_kernel(
    MLP* d_mlp,         // device mlp pointer
    int layer_ind, 
    unsigned long long seed
);


// --- Spherical Harmonics for View-Direction Encoding ---
const int SH_DEGREE = 1;
const int SH_COEFS = (SH_DEGREE + 1) * (SH_DEGREE + 1);


__device__ inline void sh_encode(const float3& dir, __half* coefs) {
    // dir shoud be normalized before calling
    const float x = dir.x, y = dir.y, z = dir.z;
    // C0, C1, C2, C3 are constants for the SH basis functions
    const float C0 = 0.28209479177387814f;
    const float C1 = 0.4886025119029199f;
    coefs[0] = __float2half(C0);
    coefs[1] = __float2half(-C1 * y);
    coefs[2] = __float2half(C1 * z);
    coefs[3] = __float2half(-C1 * x);
};


// ---------------------------
// ------- CUBLAS HLPER ------
// ---------------------------

__device__ __forceinline__ __half hrelu(const __half& x) {
    // Return the maximum of x and 0. This is the ReLU function.
    return __hmax(x, __float2half(0.0f));
};

__global__ void add_bias_and_relu_kernel(
    __half* matrix, 
    const __half* bias, 
    int m,
    int n
);

__global__ void add_bias_kernel(
    __half* matrix, 
    const __half* bias, 
    int num_elements,
    int n_cols
);
// ---------------------------
// ------- CUBLAS HLPER ------
// ---------------------------


// ---------------------------
// ------- BACKWARD HLP ------
// ---------------------------





__global__ void backprop_relu_kernel(
    __half* grad_in_out, 
    const __half* activations, 
    size_t n_elements
);


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
    float grad_clip_val = 1.0f
);


__global__ void sum_bias_gradients_kernel(
    const __half* d_output_grads, // Incoming gradients [n_points, n_cols]
    float* d_bias_grads,          // Output bias gradients [n_cols]
    int n_points, 
    int n_cols
);

__global__ void backprop_output_activations_kernel(
    int n_points,
    const __half* dL_d_density,       // from loss function
    const __half* dL_d_color,         // from loss function
    const __half* d_raw_density_out,  // raw output from density net (before softplus)
    const __half* d_raw_rgb_out,      // raw output from color net (before sigmoid)
    __half* d_grad_density,           // output grad for density net
    __half* d_grad_color              // output grad for color net
);

__global__ void split_and_add_grads_kernel(
    int n_points,
    const __half* d_grad_color_input, // Gradient coming back from the color network
    __half* d_grad_density_output,    // Gradient for the density network's output (additive)
    __half* d_grad_sh_features        // Gradient for the SH features
);


// ---------------------------
// ------- BACKWARD HLP ------
// ---------------------------


class MLP {
public:
        // --- Network Dimensions ---
    static constexpr int D_in = N_LEVELS * F_val;
    static constexpr int D_hidden = 64;
    static constexpr int D_geo_feat = 15; // Geometric features to pass to color network
    static constexpr int D_density_out = 1 + D_geo_feat; // 1 for density, 15 for features
    static constexpr int D_color_in = D_geo_feat + SH_COEFS;
    static constexpr int D_color_hidden = 64;
    static constexpr int D_color_out = 3; // RGB

private:
     // --- PARAMETER BUFFERS (FP16) ---
    CudaManagedBuffer<__half> m_density_weights1, m_density_biases1;
    CudaManagedBuffer<__half> m_density_weights2, m_density_biases2;
    CudaManagedBuffer<__half> m_color_weights1, m_color_biases1;
    CudaManagedBuffer<__half> m_color_weights2, m_color_biases2;
    CudaManagedBuffer<__half> m_color_weights3, m_color_biases3;

    // --- GRADIENT BUFFERS (FP32) ---
    CudaManagedBuffer<float> m_density_weights1_grad, m_density_biases1_grad;
    CudaManagedBuffer<float> m_density_weights2_grad, m_density_biases2_grad;
    CudaManagedBuffer<float> m_color_weights1_grad, m_color_biases1_grad;
    CudaManagedBuffer<float> m_color_weights2_grad, m_color_biases2_grad;
    CudaManagedBuffer<float> m_color_weights3_grad, m_color_biases3_grad;


    // --- ADAM OPTIMIZER STATE BUFFERS (FP32) ---
    // 1st moment (m)
    CudaManagedBuffer<float> m_dw1_m, m_db1_m;
    CudaManagedBuffer<float> m_dw2_m, m_db2_m;
    CudaManagedBuffer<float> m_cw1_m, m_cb1_m;
    CudaManagedBuffer<float> m_cw2_m, m_cb2_m;
    // NEW
    CudaManagedBuffer<float> m_cw3_m, m_cb3_m;
    // 2nd moment (v)
    CudaManagedBuffer<float> m_dw1_v, m_db1_v;
    CudaManagedBuffer<float> m_dw2_v, m_db2_v;
    CudaManagedBuffer<float> m_cw1_v, m_cb1_v;
    CudaManagedBuffer<float> m_cw2_v, m_cb2_v;
    // NEW
    CudaManagedBuffer<float> m_cw3_v, m_cb3_v;

    // Helper for backprop
    __device__ float dot_product(int N, const float* a, const float* b, int stride_b=1) const {
        float accum = 0;
        for(int i=0; i<N; ++i) accum += a[i] * (b[i*stride_b]);
        return accum;
    }
     __device__ float dot_product(int N, const float* a, float b_val) const {
        float accum = 0;
        for(int i=0; i<N; ++i) accum += a[i] * b_val;
        return accum;
    }

public:
    // --- Public Methods ---
    MLP(unsigned long long seed);

    // --- MULTI-GPU: Add accessors for each GRADIENT buffer ---
    float* density_weights1_grad_ptr() { return m_density_weights1_grad.get(); }
    size_t density_weights1_grad_size() { return m_density_weights1_grad.size(); }
    
    float* density_biases1_grad_ptr() { return m_density_biases1_grad.get(); }
    size_t density_biases1_grad_size() { return m_density_biases1_grad.size(); }
    
    float* density_weights2_grad_ptr() { return m_density_weights2_grad.get(); }
    size_t density_weights2_grad_size() { return m_density_weights2_grad.size(); }

    float* density_biases2_grad_ptr() { return m_density_biases2_grad.get(); }
    size_t density_biases2_grad_size() { return m_density_biases2_grad.size(); }

    float* color_weights1_grad_ptr() { return m_color_weights1_grad.get(); }
    size_t color_weights1_grad_size() { return m_color_weights1_grad.size(); }

    float* color_biases1_grad_ptr() { return m_color_biases1_grad.get(); }
    size_t color_biases1_grad_size() { return m_color_biases1_grad.size(); }
    
    float* color_weights2_grad_ptr() { return m_color_weights2_grad.get(); }
    size_t color_weights2_grad_size() { return m_color_weights2_grad.size(); }

    float* color_biases2_grad_ptr() { return m_color_biases2_grad.get(); }
    size_t color_biases2_grad_size() { return m_color_biases2_grad.size(); }
    
    float* color_weights3_grad_ptr() { return m_color_weights3_grad.get(); }
    size_t color_weights3_grad_size() {return m_color_weights3_grad.size(); }

    float* color_biases3_grad_ptr() { return m_color_biases3_grad.get(); }
    size_t color_biases3_grad_size() { return m_color_biases3_grad.size(); }
    // --- End of new accessors ---


    /**
     * The key concept is that density (σ) is a property of the 3D point 
     * in space and should not change based on your viewing angle. 
     * However, color (RGB) should change with the viewing angle to
     * model effects like reflections and shininess.
     * 
     *         [ 3D Position 'x' ]                     [ View Direction 'd' ]
                       |                                         |
                       |                                         |
                       v                                         v
            +---------------------+                   +--------------------------+
            | Multi-Res Hash      |                   | Spherical Harmonics (SH) |
            | Encoder             |                   | Encoder                  |
            +---------------------+                   +--------------------------+
                       |                                         |
                       |                                         |
           [ 32D Hash Features ]                                 |
                       |                                         |
                       v                                         |
      +----------------------------------+                       |
      |      DENSITY NETWORK             |                       |
      |----------------------------------|                       |
      |   Layer 1: (32 -> 64) + ReLU     |                       |
      |              |                   |                       |
      |              v                   |                       |
      |   Layer 2: (64 -> 16)            |                       |
      +----------------------------------+                       |
                       |                                         |
      (The 16 outputs from this network are split into two parts)|
                       |                                         |
           +-----------+-----------+                             |
           |                       |                             |
           v                       v                             |
    [ Final Density σ ]  [ 15D Geometric Features ]     [ 4D SH Features ]
    (Output #1)              |                                 |
                             |                                 |
                             +---------------+-----------------+
                                             |
                                             v
                                 +---------------------+
                                 |    Concatenate      |
                                 +---------------------+
                                             |
                                             v
                                [ 19D Combined Features ]
                                             |
                                             v
                        +------------------------------------+
                        |         COLOR NETWORK              |
                        |------------------------------------|
                        |   Layer 1: (19 -> 64) + ReLU       |
                        |                 |                  |
                        |                 v                  |
                        |   Layer 2: (64 -> 3) + Sigmoid     |
                        +------------------------------------+
                                          |
                                          v
                                   [ Final RGB Color ]
                                      (Output #2)
     *
    */

    void forward_density(
        cublasHandle_t handle,
        cudaStream_t stream,
        int n_points,
        const __half* d_input_hash_features,
        __half* d_out_density
    ) const;



    // --- FORWARD PASS (Corrected and Flexible Version) ---
    void forward(
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
    ) const;



    __device__ inline void safe_atomic_add(float* address, float val) {

        if (!isnan(val) && !isinf(val)) {
            atomicAdd(address, val);
        }

    }

    // ... (rest of the file, class MLP, etc.)
    // --- BACKWARD PASS (Numerically Stable Version) ---
    void backward(
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
    );



    void adam_update(
        float lr, 
        float beta1, 
        float beta2, 
        float epsilon, 
        int step, 
        float l2_reg_weight,
        int grad_accumulation_steps, 
        cudaStream_t stream
    );

    void zero_grad(cudaStream_t stream);


    // --- Device-side Accessors ---
    __host__ __device__ const __half* density_weights1() const { return m_density_weights1.get(); }
    __host__ __device__ const __half* density_biases1()  const { return m_density_biases1.get(); }
    __host__ __device__ const __half* density_weights2() const { return m_density_weights2.get(); }
    __host__ __device__ const __half* density_biases2()  const { return m_density_biases2.get(); }
    __host__ __device__ const __half* color_weights1() const { return m_color_weights1.get(); }
    __host__ __device__ const __half* color_biases1()  const { return m_color_biases1.get(); }
    __host__ __device__ const __half* color_weights2() const { return m_color_weights2.get(); }
    __host__ __device__ const __half* color_biases2()  const { return m_color_biases2.get(); }
    __host__ __device__ const __half* color_weights3() const { return m_color_weights3.get(); };
    __host__ __device__ const __half* color_biases3() const { return m_color_biases3.get(); };

    void release_optimizer_states();
};





/////////////////////////////////////////////////////////////////////////
// ------------- Multiple Layer Perceptron ------------- //
/////////////////////////////////////////////////////////////////////////



// ===================================================================
// =================== PROPOSAL NETWORK DEFINITION ===================
// ===================================================================

__global__ void forwardprop_softplus_kernel(
    int n_elements, 
    __half* d_in_out
);

__global__ void backprop_softplus_kernel(
    int n_elements,
    const __half* d_incming_grad,
    const __half* d_raw_logits,
    __half* d_output_grad
);

// It has no color head and fewer layers.
class ProposalNetwork {
public:
    // --- Network Dimensions ---
    // Note: D_in is the same as the main MLP, as it also uses the hash grid.
    static constexpr int D_in = 16 * F_val;
    static constexpr int D_hidden = 64;
    static constexpr int D_out = 1; // Only one output: density

private:
    // --- PARAMETER BUFFERS ---
    CudaManagedBuffer<__half> m_weights1, m_biases1;
    CudaManagedBuffer<__half> m_weights2, m_biases2;

    // --- GRADIENT BUFFERS ---
    CudaManagedBuffer<float> m_weights1_grad, m_biases1_grad;
    CudaManagedBuffer<float> m_weights2_grad, m_biases2_grad;


    // --- ADAM OPTIMIZER STATE BUFFERS ---
    CudaManagedBuffer<float> m_w1_m, m_w1_v;
    CudaManagedBuffer<float> m_b1_m, m_b1_v;
    CudaManagedBuffer<float> m_w2_m, m_w2_v;
    CudaManagedBuffer<float> m_b2_m, m_b2_v;

public:
    // --- Public Methods ---
    ProposalNetwork(unsigned long long seed);
    void adam_update_and_zero_grad(float lr, float beta1, float beta2, float epsilon, int step, float l2_reg_weight, float grad_unscaler, cudaStream_t stream);


    // --- MULTI-GPU: Add accessors for each gradient buffer ---
    float* weights1_grad_ptr() { return m_weights1_grad.get(); }
    size_t weights1_grad_size() { return m_weights1_grad.size(); }

    float* biases1_grad_ptr() { return m_biases1_grad.get(); }
    size_t biases1_grad_size() { return m_biases1_grad.size(); }

    float* weights2_grad_ptr() { return m_weights2_grad.get(); }
    size_t weights2_grad_size() { return m_weights2_grad.size(); }

    float* biases2_grad_ptr() { return m_biases2_grad.get(); }
    size_t biases2_grad_size() { return m_biases2_grad.size(); }
    // --- End of new accessors ---


    void forward(
         cublasHandle_t handle,
        cudaStream_t stream,
        int n_points,
        const __half* d_inputs,
        __half* d_outputs
    ) const;

     void backward(
        cublasHandle_t handle,
        cudaStream_t stream,
        int n_points,
        const __half* dL_d_density,       // Incoming gradient from the loss
        const __half* d_inputs,           // The original hash features from the forward pass
        const __half* d_hidden_activations, // The hidden layer activations from the forward pass
        const __half* d_output_logits,      // The raw output from layer 2 (before softplus)
        __half* d_grad_hash_features      // Output: the gradient w.r.t. the hash features
    );
};




__global__ void init_proposal_weights_kernel(
    __half* weights, int fan_in, 
    int fan_out, unsigned long long seed
);


__global__ void activate_density_kernel(
    int n_points, 
    const __half* d_density_out_full, 
    __half* d_out_density
);




// ===================================================================
// =================== PROPOSAL NETWORK DEFINITION ===================
// ===================================================================
#endif