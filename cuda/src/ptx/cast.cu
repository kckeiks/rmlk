#include <cstdint>
#include "cast_op_macro.cuh"
#include "cuda_fp16.h"

// Floating point to floating point and vice versa.
CAST_OP(__half, float, cast_fwd_f16_to_f32, __half2float(x))
CAST_OP(float, __half, cast_fwd_f32_to_f16, __float2half(x))
CAST_OP(__half, double, cast_fwd_f16_to_f64, static_cast<double>(__half2float(x)))
CAST_OP(double, __half, cast_fwd_f64_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(float, double, cast_fwd_f32_to_f64, static_cast<double>(x))
CAST_OP(double, float, cast_fwd_f64_to_f32, static_cast<float>(x))

// Floating point to fixed point and vice versa.
CAST_OP(int8_t, __half, cast_fwd_i8_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, int8_t, cast_fwd_f16_to_i8, static_cast<int8_t>(__half2float(x)))
CAST_OP(int8_t, float, cast_fwd_i8_to_f32, static_cast<float>(x))
CAST_OP(float, int8_t, cast_fwd_f32_to_i8, static_cast<int8_t>(x))
CAST_OP(int8_t, double, cast_fwd_i8_to_f64, static_cast<double>(x))
CAST_OP(double, int8_t, cast_fwd_f64_to_i8, static_cast<int8_t>(x))

CAST_OP(uint8_t, __half, cast_fwd_u8_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, uint8_t, cast_fwd_f16_to_u8, static_cast<uint8_t>(__half2float(x)))
CAST_OP(uint8_t, float, cast_fwd_u8_to_f32, static_cast<float>(x))
CAST_OP(float, uint8_t, cast_fwd_f32_to_u8, static_cast<uint8_t>(x))
CAST_OP(uint8_t, double, cast_fwd_u8_to_f64, static_cast<double>(x))
CAST_OP(double, uint8_t, cast_fwd_f64_to_u8, static_cast<uint8_t>(x))

CAST_OP(int16_t, __half, cast_fwd_i16_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, int16_t, cast_fwd_f16_to_i16, static_cast<int16_t>(__half2float(x)))
CAST_OP(int16_t, float, cast_fwd_i16_to_f32, static_cast<float>(x))
CAST_OP(float, int16_t, cast_fwd_f32_to_i16, static_cast<int16_t>(x))
CAST_OP(int16_t, double, cast_fwd_i16_to_f64, static_cast<double>(x))
CAST_OP(double, int16_t, cast_fwd_f64_to_i16, static_cast<int16_t>(x))

CAST_OP(uint16_t, __half, cast_fwd_u16_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, uint16_t, cast_fwd_f16_to_u16, static_cast<uint16_t>(__half2float(x)))
CAST_OP(uint16_t, float, cast_fwd_u16_to_f32, static_cast<float>(x))
CAST_OP(float, uint16_t, cast_fwd_f32_to_u16, static_cast<uint16_t>(x))
CAST_OP(uint16_t, double, cast_fwd_u16_to_f64, static_cast<double>(x))
CAST_OP(double, uint16_t, cast_fwd_f64_to_u16, static_cast<uint16_t>(x))

CAST_OP(int32_t, __half, cast_fwd_i32_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, int32_t, cast_fwd_f16_to_i32, static_cast<int32_t>(__half2float(x)))
CAST_OP(int32_t, float, cast_fwd_i32_to_f32, static_cast<float>(x))
CAST_OP(float, int32_t, cast_fwd_f32_to_i32, static_cast<int32_t>(x))
CAST_OP(int32_t, double, cast_fwd_i32_to_f64, static_cast<double>(x))
CAST_OP(double, int32_t, cast_fwd_f64_to_i32, static_cast<int32_t>(x))

CAST_OP(uint32_t, __half, cast_fwd_u32_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, uint32_t, cast_fwd_f16_to_u32, static_cast<uint32_t>(__half2float(x)))
CAST_OP(uint32_t, float, cast_fwd_u32_to_f32, static_cast<float>(x))
CAST_OP(float, uint32_t, cast_fwd_f32_to_u32, static_cast<uint32_t>(x))
CAST_OP(uint32_t, double, cast_fwd_u32_to_f64, static_cast<double>(x))
CAST_OP(double, uint32_t, cast_fwd_f64_to_u32, static_cast<uint32_t>(x))

CAST_OP(int64_t, __half, cast_fwd_i64_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, int64_t, cast_fwd_f16_to_i64, static_cast<int64_t>(__half2float(x)))
CAST_OP(int64_t, float, cast_fwd_i64_to_f32, static_cast<float>(x))
CAST_OP(float, int64_t, cast_fwd_f32_to_i64, static_cast<int64_t>(x))
CAST_OP(int64_t, double, cast_fwd_i64_to_f64, static_cast<double>(x))
CAST_OP(double, int64_t, cast_fwd_f64_to_i64, static_cast<int64_t>(x))

CAST_OP(uint64_t, __half, cast_fwd_u64_to_f16, __float2half(static_cast<float>(x)))
CAST_OP(__half, uint64_t, cast_fwd_f16_to_u64, static_cast<uint64_t>(__half2float(x)))
CAST_OP(uint64_t, float, cast_fwd_u64_to_f32, static_cast<float>(x))
CAST_OP(float, uint64_t, cast_fwd_f32_to_u64, static_cast<uint64_t>(x))
CAST_OP(uint64_t, double, cast_fwd_u64_to_f64, static_cast<double>(x))
CAST_OP(double, uint64_t, cast_fwd_f64_to_u64, static_cast<uint64_t>(x))

// Floating point to fixed point and vice versa.
CAST_OP(int8_t, int16_t, cast_fwd_i8_to_i16, static_cast<int16_t>(x))
CAST_OP(int16_t, int8_t, cast_fwd_i16_to_i8, static_cast<int8_t>(x))
CAST_OP(int8_t, int32_t, cast_fwd_i8_to_i32, static_cast<int32_t>(x))
CAST_OP(int32_t, int8_t, cast_fwd_i32_to_i8, static_cast<int8_t>(x))
CAST_OP(int8_t, int64_t, cast_fwd_i8_to_i64, static_cast<int64_t>(x))
CAST_OP(int64_t, int8_t, cast_fwd_i64_to_i8, static_cast<int8_t>(x))

CAST_OP(int16_t, int32_t, cast_fwd_i16_to_i32, static_cast<int32_t>(x))
CAST_OP(int32_t, int16_t, cast_fwd_i32_to_i16, static_cast<int16_t>(x))
CAST_OP(int16_t, int64_t, cast_fwd_i16_to_i64, static_cast<int64_t>(x))
CAST_OP(int64_t, int16_t, cast_fwd_i64_to_i16, static_cast<int16_t>(x))

CAST_OP(int32_t, int64_t, cast_fwd_i32_to_i64, static_cast<int64_t>(x))
CAST_OP(int64_t, int32_t, cast_fwd_i64_to_i32, static_cast<int32_t>(x))

CAST_OP(uint8_t, uint16_t, cast_fwd_u8_to_u16, static_cast<uint16_t>(x))
CAST_OP(uint16_t, uint8_t, cast_fwd_u16_to_u8, static_cast<uint8_t>(x))
CAST_OP(uint8_t, uint32_t, cast_fwd_u8_to_u32, static_cast<uint32_t>(x))
CAST_OP(uint32_t, uint8_t, cast_fwd_u32_to_u8, static_cast<uint8_t>(x))
CAST_OP(uint8_t, uint64_t, cast_fwd_u8_to_u64, static_cast<uint64_t>(x))
CAST_OP(uint64_t, uint8_t, cast_fwd_u64_to_u8, static_cast<uint8_t>(x))

CAST_OP(uint16_t, uint32_t, cast_fwd_u16_to_u32, static_cast<uint32_t>(x))
CAST_OP(uint32_t, uint16_t, cast_fwd_u32_to_u16, static_cast<uint16_t>(x))
CAST_OP(uint16_t, uint64_t, cast_fwd_u16_to_u64, static_cast<uint64_t>(x))
CAST_OP(uint64_t, uint16_t, cast_fwd_u64_to_u16, static_cast<uint16_t>(x))

CAST_OP(uint32_t, uint64_t, cast_fwd_u32_to_u64, static_cast<uint64_t>(x))
CAST_OP(uint64_t, uint32_t, cast_fwd_u64_to_u32, static_cast<uint32_t>(x))

// Boolean casting.
CAST_OP(__half, bool, cast_fwd_f16_to_bool, __half2float(x) == 0.0f ? false : true)
CAST_OP(float, bool, cast_fwd_f32_to_bool, x == 0.0f ? false : true)
CAST_OP(double, bool, cast_fwd_f64_to_bool, x == 0.0 ? false : true)
CAST_OP(int8_t, bool, cast_fwd_i8_to_bool, x == 0 ? false : true)
CAST_OP(uint8_t, bool, cast_fwd_u8_to_bool, x == 0 ? false : true)
CAST_OP(int16_t, bool, cast_fwd_i16_to_bool, x == 0 ? false : true)
CAST_OP(uint16_t, bool, cast_fwd_u16_to_bool, x == 0 ? false : true)
CAST_OP(int32_t, bool, cast_fwd_i32_to_bool, x == 0 ? false : true)
CAST_OP(uint32_t, bool, cast_fwd_u32_to_bool, x == 0 ? false : true)
CAST_OP(int64_t, bool, cast_fwd_i64_to_bool, x == 0 ? false : true)
CAST_OP(uint64_t, bool, cast_fwd_u64_to_bool, x == 0 ? false : true)

CAST_OP(bool, __half, cast_fwd_bool_to_f16, x ? __float2half(1.0f) : __float2half(0.0f))
CAST_OP(bool, float, cast_fwd_bool_to_f32, x ? 1.0f : 0.0f)
CAST_OP(bool, double, cast_fwd_bool_to_f64, x ? 1.0 : 0.0)
CAST_OP(bool, int8_t, cast_fwd_bool_to_i8, x ? 1 : 0)
CAST_OP(bool, uint8_t, cast_fwd_bool_to_u8, x ? 1 : 0)
CAST_OP(bool, int16_t, cast_fwd_bool_to_i16, x ? 1 : 0)
CAST_OP(bool, uint16_t, cast_fwd_bool_to_u16, x ? 1 : 0)
CAST_OP(bool, int32_t, cast_fwd_bool_to_i32, x ? 1 : 0)
CAST_OP(bool, uint32_t, cast_fwd_bool_to_u32, x ? 1 : 0)
CAST_OP(bool, int64_t, cast_fwd_bool_to_i64, x ? 1 : 0)
CAST_OP(bool, uint64_t, cast_fwd_bool_to_u64, x ? 1 : 0)