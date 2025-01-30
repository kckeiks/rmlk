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
#include "cuda_fp16.h"

#define LONG_BINARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2 * num_dims; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int lhs_i = 0; \
        unsigned int rhs_i = 0; \
        for (int d = num_dims - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            lhs_i += i_dim * lhs_strides[d]; \
            rhs_i += i_dim * rhs_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = lhs ? lhs[lhs_i] : out[i]; \
        TYPENAME y = rhs ? rhs[rhs_i] : out[i]; \
        TYPENAME fx; \
        FUNC\
        out[i] = fx; \
    } \
} \

#define BINARY_OP(TYPENAME, FORWARD, FUNC) \
    LONG_BINARY_OP(TYPENAME, FORWARD, fx = (FUNC);)
