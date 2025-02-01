/*
 * This file is derived from the `dfdx` project:
 * Original Repository: https://github.com/coreylowman/dfdx
 * Original File Path: dfdx-core/src/tensor_ops/mul/binary_mul.cu
 * Original Author: Corey Lowman
 * Original License: MIT License
 *
 * Modifications by: Michael Meier
 * Date of Modification: 2025
 *
 * This modified file is distributed under the MIT License,
 * in accordance with the original license terms.
 */
#include "binary_op_macros.cuh"

BINARY_OP(__half, mul_fwd_f16, x * y)

BINARY_OP(float, mul_fwd_f32, x * y)

BINARY_OP(double, mul_fwd_f64, x * y)
