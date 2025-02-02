/*
 * This file is derived from the `dfdx` project:
 * Original Repository: https://github.com/coreylowman/dfdx
 * Original File Path: dfdx-core/src/tensor_ops/add/binary_add.cu
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

BINARY_OP(__half, add_fwd_f16, x + y)

BINARY_OP(float, add_fwd_f32, x + y)

BINARY_OP(double, add_fwd_f64, x + y)

BINARY_OP_ALPHA_BETA_INPLACE(__half, add_alpha_beta_inplace_fwd_f16, alpha * x + beta * y)

BINARY_OP_ALPHA_BETA_INPLACE(float, add_alpha_beta_inplace_fwd_f32, alpha * x + beta * y)

BINARY_OP_ALPHA_BETA_INPLACE(double, add_alpha_beta_inplace_fwd_f64, alpha * x + beta * y)
