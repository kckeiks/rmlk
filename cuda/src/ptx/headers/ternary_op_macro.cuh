#include "cuda_fp16.h"

#define LONG_TERNARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *output_shape, \
    const size_t *x_strides, \
    const size_t *y_strides, \
    const size_t *z_strides, \
    const TYPENAME *x_data, \
    const TYPENAME *y_data, \
    const TYPENAME *z_data, \
    TYPENAME *out \
) { \
    const size_t *dims = output_shape; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int x_i = 0; \
        unsigned int y_i = 0; \
        unsigned int z_i = 0; \
        for (int d = num_dims - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            x_i += i_dim * x_strides[d]; \
            y_i += i_dim * y_strides[d]; \
            z_i += i_dim * z_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = x_data ? x_data[x_i] : out[i]; \
        TYPENAME y = y_data ? y_data[y_i] : out[i]; \
        TYPENAME z = z_data ? z_data[z_i] : out[i]; \
        TYPENAME fx; \
        FUNC\
        out[i] = fx; \
    } \
} \

#define TERNARY_OP(TYPENAME, FORWARD, FUNC) \
    LONG_TERNARY_OP(TYPENAME, FORWARD, fx = (FUNC);)