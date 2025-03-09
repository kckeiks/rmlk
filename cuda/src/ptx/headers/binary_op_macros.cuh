/*
 * This file is derived from the `dfdx` project:
 * Original Repository: https://github.com/coreylowman/dfdx
 * Original File Path: dfdx-core/src/tensor_ops/utilities/binary_op_macros.cuh
 * Original Author: Corey Lowman
 * Original License: MIT License
 *
 * Modifications by: Michael Meier
 * Date of Modification: 2025
 *
 * This modified file is distributed under the MIT License,
 * in accordance with the original license terms.
 */
#define LONG_BINARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,   /* The number of elements in the output.       */\
    const size_t rank,        /* The rank of input tensor (must be > 0).     */\
    const size_t *info,       /* The shape and stride of a and b.            */\
    const TYPENAME *a,        /* The input tensor data.                      */\
    const TYPENAME *b,        /* The input tensor data.                      */\
    TYPENAME *out             /* The output data.                            */\
) { \
    const size_t *dims = info; \
    const size_t *a_strides = info + rank; \
    const size_t *b_strides = info + 2 * rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int a_i = 0; \
        unsigned int b_i = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            a_i += i_dim * a_strides[d]; \
            b_i += i_dim * b_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = a[a_i]; \
        TYPENAME y = b[b_i]; \
        TYPENAME fx; \
        FUNC\
        out[i] = fx; \
    } \
} \

#define LONG_BINARY_OP_ALPHA_BETA_INPLACE(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const TYPENAME alpha,        /* The alpha value.                            */\
    const TYPENAME beta,         /* The beta value.                             */\
    const size_t num_elems,      /* The number of axes to reduce (must be > 0). */\
    const size_t rank,           /* The rank of input tensor (must be > 0).     */\
    const size_t *info,          /* The shape and stride of a and b.            */\
    const TYPENAME *a,           /* The input tensor data.                      */\
    TYPENAME *b                  /* The output data.                            */\
) { \
    const size_t *dims = info; \
    const size_t *a_strides = info + rank; \
    const size_t *b_strides = info + 2 * rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int a_i = 0; \
        unsigned int b_i = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            a_i += i_dim * a_strides[d]; \
            b_i += i_dim * b_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = a[a_i]; \
        TYPENAME y = b[b_i]; \
        TYPENAME fx; \
        FUNC\
        b[b_i] = fx; \
    } \
} \

#define BINARY_OP(TYPENAME, FORWARD, FUNC) \
    LONG_BINARY_OP(TYPENAME, FORWARD, fx = (FUNC);)

#define BINARY_OP_ALPHA_BETA_INPLACE(TYPENAME, FORWARD, FUNC) \
    LONG_BINARY_OP_ALPHA_BETA_INPLACE(TYPENAME, FORWARD, fx = (FUNC);)
