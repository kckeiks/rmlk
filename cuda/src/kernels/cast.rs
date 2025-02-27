use crate::ptx::CAST;

pub const MODULE_NAME: &str = "cast";
pub const FWD_FN_NAMES: &[&str] = &[
    "cast_fwd_f16_to_f32",
    "cast_fwd_f32_to_f16",
    "cast_fwd_f16_to_f64",
    "cast_fwd_f64_to_f16",
    "cast_fwd_f32_to_f64",
    "cast_fwd_f64_to_f32",
    "cast_fwd_i8_to_f16",
    "cast_fwd_f16_to_i8",
    "cast_fwd_i8_to_f32",
    "cast_fwd_f32_to_i8",
    "cast_fwd_i8_to_f64",
    "cast_fwd_f64_to_i8",
    "cast_fwd_i16_to_f16",
    "cast_fwd_f16_to_i16",
    "cast_fwd_i16_to_f32",
    "cast_fwd_f32_to_i16",
    "cast_fwd_i16_to_f64",
    "cast_fwd_f64_to_i16",
    "cast_fwd_i32_to_f16",
    "cast_fwd_f16_to_i32",
    "cast_fwd_i32_to_f32",
    "cast_fwd_f32_to_i32",
    "cast_fwd_i32_to_f64",
    "cast_fwd_f64_to_i32",
    "cast_fwd_i64_to_f16",
    "cast_fwd_f16_to_i64",
    "cast_fwd_i64_to_f32",
    "cast_fwd_f32_to_i64",
    "cast_fwd_i64_to_f64",
    "cast_fwd_f64_to_i64",
    "cast_fwd_u8_to_f16",
    "cast_fwd_f16_to_u8",
    "cast_fwd_u8_to_f32",
    "cast_fwd_f32_to_u8",
    "cast_fwd_u8_to_f64",
    "cast_fwd_f64_to_u8",
    "cast_fwd_u16_to_f16",
    "cast_fwd_f16_to_u16",
    "cast_fwd_u16_to_f32",
    "cast_fwd_f32_to_u16",
    "cast_fwd_u16_to_f64",
    "cast_fwd_f64_to_u16",
    "cast_fwd_u32_to_f16",
    "cast_fwd_f16_to_u32",
    "cast_fwd_u32_to_f32",
    "cast_fwd_f32_to_u32",
    "cast_fwd_u32_to_f64",
    "cast_fwd_f64_to_u32",
    "cast_fwd_u64_to_f16",
    "cast_fwd_f16_to_u64",
    "cast_fwd_u64_to_f32",
    "cast_fwd_f32_to_u64",
    "cast_fwd_u64_to_f64",
    "cast_fwd_f64_to_u64",
    "cast_fwd_i8_to_i16",
    "cast_fwd_i16_to_i8",
    "cast_fwd_i8_to_i32",
    "cast_fwd_i32_to_i8",
    "cast_fwd_i8_to_i64",
    "cast_fwd_i64_to_i8",
    "cast_fwd_i16_to_i32",
    "cast_fwd_i32_to_i16",
    "cast_fwd_i16_to_i64",
    "cast_fwd_i64_to_i16",
    "cast_fwd_i32_to_i64",
    "cast_fwd_i64_to_i32",
    "cast_fwd_u8_to_u16",
    "cast_fwd_u16_to_u8",
    "cast_fwd_u8_to_u32",
    "cast_fwd_u32_to_u8",
    "cast_fwd_u8_to_u64",
    "cast_fwd_u64_to_u8",
    "cast_fwd_u16_to_u32",
    "cast_fwd_u32_to_u16",
    "cast_fwd_u16_to_u64",
    "cast_fwd_u64_to_u16",
    "cast_fwd_u32_to_u64",
    "cast_fwd_u64_to_u32",
    "cast_fwd_f16_to_bool",
    "cast_fwd_f32_to_bool",
    "cast_fwd_f64_to_bool",
    "cast_fwd_i8_to_bool",
    "cast_fwd_u8_to_bool",
    "cast_fwd_i16_to_bool",
    "cast_fwd_u16_to_bool",
    "cast_fwd_i32_to_bool",
    "cast_fwd_u32_to_bool",
    "cast_fwd_i64_to_bool",
    "cast_fwd_u64_to_bool",
    "cast_fwd_bool_to_f16",
    "cast_fwd_bool_to_f32",
    "cast_fwd_bool_to_f64",
    "cast_fwd_bool_to_i8",
    "cast_fwd_bool_to_u8",
    "cast_fwd_bool_to_i16",
    "cast_fwd_bool_to_u16",
    "cast_fwd_bool_to_i32",
    "cast_fwd_bool_to_u32",
    "cast_fwd_bool_to_i64",
    "cast_fwd_bool_to_u64",
];
pub const PTX_SRC: &str = CAST;

#[derive(Debug, Copy, Clone)]
pub enum CastKernel {
    F16ToF32,
    F32ToF16,
    F16ToF64,
    F64ToF16,
    F32ToF64,
    F64ToF32,
    I8ToF16,
    F16ToI8,
    I8ToF32,
    F32ToI8,
    I8ToF64,
    F64ToI8,
    I16ToF16,
    F16ToI16,
    I16ToF32,
    F32ToI16,
    I16ToF64,
    F64ToI16,
    I32ToF16,
    F16ToI32,
    I32ToF32,
    F32ToI32,
    I32ToF64,
    F64ToI32,
    I64ToF16,
    F16ToI64,
    I64ToF32,
    F32ToI64,
    I64ToF64,
    F64ToI64,
    U8ToF16,
    F16ToU8,
    U8ToF32,
    F32ToU8,
    U8ToF64,
    F64ToU8,
    U16ToF16,
    F16ToU16,
    U16ToF32,
    F32ToU16,
    U16ToF64,
    F64ToU16,
    U32ToF16,
    F16ToU32,
    U32ToF32,
    F32ToU32,
    U32ToF64,
    F64ToU32,
    U64ToF16,
    F16ToU64,
    U64ToF32,
    F32ToU64,
    U64ToF64,
    F64ToU64,
    I8ToI16,
    I16ToI8,
    I8ToI32,
    I32ToI8,
    I8ToI64,
    I64ToI8,
    I16ToI32,
    I32ToI16,
    I16ToI64,
    I64ToI16,
    I32ToI64,
    I64ToI32,
    U8ToU16,
    U16ToU8,
    U8ToU32,
    U32ToU8,
    U8ToU64,
    U64ToU8,
    U16ToU32,
    U32ToU16,
    U16ToU64,
    U64ToU16,
    U32ToU64,
    U64ToU32,
    F16ToBool,
    F32ToBool,
    F64ToBool,
    I8ToBool,
    U8ToBool,
    I16ToBool,
    U16ToBool,
    I32ToBool,
    U32ToBool,
    I64ToBool,
    U64ToBool,
    BoolToF16,
    BoolToF32,
    BoolToF64,
    BoolToI8,
    BoolToU8,
    BoolToI16,
    BoolToU16,
    BoolToI32,
    BoolToU32,
    BoolToI64,
    BoolToU64,
}

impl CastKernel {
    pub fn as_str(&self) -> &'static str {
        match self {
            CastKernel::F16ToF32 => "cast_fwd_f16_to_f32",
            CastKernel::F32ToF16 => "cast_fwd_f32_to_f16",
            CastKernel::F16ToF64 => "cast_fwd_f16_to_f64",
            CastKernel::F64ToF16 => "cast_fwd_f64_to_f16",
            CastKernel::F32ToF64 => "cast_fwd_f32_to_f64",
            CastKernel::F64ToF32 => "cast_fwd_f64_to_f32",
            CastKernel::I8ToF16 => "cast_fwd_i8_to_f16",
            CastKernel::F16ToI8 => "cast_fwd_f16_to_i8",
            CastKernel::I8ToF32 => "cast_fwd_i8_to_f32",
            CastKernel::F32ToI8 => "cast_fwd_f32_to_i8",
            CastKernel::I8ToF64 => "cast_fwd_i8_to_f64",
            CastKernel::F64ToI8 => "cast_fwd_f64_to_i8",
            CastKernel::I16ToF16 => "cast_fwd_i16_to_f16",
            CastKernel::F16ToI16 => "cast_fwd_f16_to_i16",
            CastKernel::I16ToF32 => "cast_fwd_i16_to_f32",
            CastKernel::F32ToI16 => "cast_fwd_f32_to_i16",
            CastKernel::I16ToF64 => "cast_fwd_i16_to_f64",
            CastKernel::F64ToI16 => "cast_fwd_f64_to_i16",
            CastKernel::I32ToF16 => "cast_fwd_i32_to_f16",
            CastKernel::F16ToI32 => "cast_fwd_f16_to_i32",
            CastKernel::I32ToF32 => "cast_fwd_i32_to_f32",
            CastKernel::F32ToI32 => "cast_fwd_f32_to_i32",
            CastKernel::I32ToF64 => "cast_fwd_i32_to_f64",
            CastKernel::F64ToI32 => "cast_fwd_f64_to_i32",
            CastKernel::I64ToF16 => "cast_fwd_i64_to_f16",
            CastKernel::F16ToI64 => "cast_fwd_f16_to_i64",
            CastKernel::I64ToF32 => "cast_fwd_i64_to_f32",
            CastKernel::F32ToI64 => "cast_fwd_f32_to_i64",
            CastKernel::I64ToF64 => "cast_fwd_i64_to_f64",
            CastKernel::F64ToI64 => "cast_fwd_f64_to_i64",
            CastKernel::U8ToF16 => "cast_fwd_u8_to_f16",
            CastKernel::F16ToU8 => "cast_fwd_f16_to_u8",
            CastKernel::U8ToF32 => "cast_fwd_u8_to_f32",
            CastKernel::F32ToU8 => "cast_fwd_f32_to_u8",
            CastKernel::U8ToF64 => "cast_fwd_u8_to_f64",
            CastKernel::F64ToU8 => "cast_fwd_f64_to_u8",
            CastKernel::U16ToF16 => "cast_fwd_u16_to_f16",
            CastKernel::F16ToU16 => "cast_fwd_f16_to_u16",
            CastKernel::U16ToF32 => "cast_fwd_u16_to_f32",
            CastKernel::F32ToU16 => "cast_fwd_f32_to_u16",
            CastKernel::U16ToF64 => "cast_fwd_u16_to_f64",
            CastKernel::F64ToU16 => "cast_fwd_f64_to_u16",
            CastKernel::U32ToF16 => "cast_fwd_u32_to_f16",
            CastKernel::F16ToU32 => "cast_fwd_f16_to_u32",
            CastKernel::U32ToF32 => "cast_fwd_u32_to_f32",
            CastKernel::F32ToU32 => "cast_fwd_f32_to_u32",
            CastKernel::U32ToF64 => "cast_fwd_u32_to_f64",
            CastKernel::F64ToU32 => "cast_fwd_f64_to_u32",
            CastKernel::U64ToF16 => "cast_fwd_u64_to_f16",
            CastKernel::F16ToU64 => "cast_fwd_f16_to_u64",
            CastKernel::U64ToF32 => "cast_fwd_u64_to_f32",
            CastKernel::F32ToU64 => "cast_fwd_f32_to_u64",
            CastKernel::U64ToF64 => "cast_fwd_u64_to_f64",
            CastKernel::F64ToU64 => "cast_fwd_f64_to_u64",
            CastKernel::I8ToI16 => "cast_fwd_i8_to_i16",
            CastKernel::I16ToI8 => "cast_fwd_i16_to_i8",
            CastKernel::I8ToI32 => "cast_fwd_i8_to_i32",
            CastKernel::I32ToI8 => "cast_fwd_i32_to_i8",
            CastKernel::I8ToI64 => "cast_fwd_i8_to_i64",
            CastKernel::I64ToI8 => "cast_fwd_i64_to_i8",
            CastKernel::I16ToI32 => "cast_fwd_i16_to_i32",
            CastKernel::I32ToI16 => "cast_fwd_i32_to_i16",
            CastKernel::I16ToI64 => "cast_fwd_i16_to_i64",
            CastKernel::I64ToI16 => "cast_fwd_i64_to_i16",
            CastKernel::I32ToI64 => "cast_fwd_i32_to_i64",
            CastKernel::I64ToI32 => "cast_fwd_i64_to_i32",
            CastKernel::U8ToU16 => "cast_fwd_u8_to_u16",
            CastKernel::U16ToU8 => "cast_fwd_u16_to_u8",
            CastKernel::U8ToU32 => "cast_fwd_u8_to_u32",
            CastKernel::U32ToU8 => "cast_fwd_u32_to_u8",
            CastKernel::U8ToU64 => "cast_fwd_u8_to_u64",
            CastKernel::U64ToU8 => "cast_fwd_u64_to_u8",
            CastKernel::U16ToU32 => "cast_fwd_u16_to_u32",
            CastKernel::U32ToU16 => "cast_fwd_u32_to_u16",
            CastKernel::U16ToU64 => "cast_fwd_u16_to_u64",
            CastKernel::U64ToU16 => "cast_fwd_u64_to_u16",
            CastKernel::U32ToU64 => "cast_fwd_u32_to_u64",
            CastKernel::U64ToU32 => "cast_fwd_u64_to_u32",
            CastKernel::F16ToBool => "cast_fwd_f16_to_bool",
            CastKernel::F32ToBool => "cast_fwd_f32_to_bool",
            CastKernel::F64ToBool => "cast_fwd_f64_to_bool",
            CastKernel::I8ToBool => "cast_fwd_i8_to_bool",
            CastKernel::U8ToBool => "cast_fwd_u8_to_bool",
            CastKernel::I16ToBool => "cast_fwd_i16_to_bool",
            CastKernel::U16ToBool => "cast_fwd_u16_to_bool",
            CastKernel::I32ToBool => "cast_fwd_i32_to_bool",
            CastKernel::U32ToBool => "cast_fwd_u32_to_bool",
            CastKernel::I64ToBool => "cast_fwd_i64_to_bool",
            CastKernel::U64ToBool => "cast_fwd_u64_to_bool",
            CastKernel::BoolToF16 => "cast_fwd_bool_to_f16",
            CastKernel::BoolToF32 => "cast_fwd_bool_to_f32",
            CastKernel::BoolToF64 => "cast_fwd_bool_to_f64",
            CastKernel::BoolToI8 => "cast_fwd_bool_to_i8",
            CastKernel::BoolToU8 => "cast_fwd_bool_to_u8",
            CastKernel::BoolToI16 => "cast_fwd_bool_to_i16",
            CastKernel::BoolToU16 => "cast_fwd_bool_to_u16",
            CastKernel::BoolToI32 => "cast_fwd_bool_to_i32",
            CastKernel::BoolToU32 => "cast_fwd_bool_to_u32",
            CastKernel::BoolToI64 => "cast_fwd_bool_to_i64",
            CastKernel::BoolToU64 => "cast_fwd_bool_to_u64",
        }
    }
}

#[cfg(test)]
mod test {
    use crate::kernels::cast::CastKernel;
    use crate::kernels::unary;
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use half::f16;

    macro_rules! generate_cast_test {
            (
                $test_name:ident,
                $variant:expr,
                $input_ty:ty,
                $output_ty:ty,
                [$($input:expr),*],
                [$($expected:expr),*]
            ) => {
                #[test]
                fn $test_name() {
                    let device = CudaDevice::new(0).unwrap();

                    let x_shape = vec![2, 2];
                    let x_data = device.htod_copy::<$input_ty>(vec![$($input),*]).unwrap();

                    let f = utils::load_cast_kernel(&device, $variant).unwrap();

                    let output_shape = x_shape.clone();
                    let mut out_data = device.alloc_zeros::<$output_ty>(output_shape.iter().product()).unwrap();

                    unsafe {
                        unary::explicit_io_types_compute(f, &x_data, &mut out_data).unwrap();
                    }

                    let result: Vec<$output_ty> = device.dtoh_sync_copy(&out_data).unwrap();

                    assert_eq!(result, vec![$($expected),*]);
                }
            };
        }

    generate_cast_test!(
        test_cast_f16_to_f32,
        CastKernel::F16ToF32,
        f16,
        f32,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_f16,
        CastKernel::F32ToF16,
        f32,
        f16,
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_f64,
        CastKernel::F16ToF64,
        f16,
        f64,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_f16,
        CastKernel::F64ToF16,
        f64,
        f16,
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f32_to_f64,
        CastKernel::F32ToF64,
        f32,
        f64,
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_f32,
        CastKernel::F64ToF32,
        f64,
        f32,
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_i8_to_f16,
        CastKernel::I8ToF16,
        i8,
        f16,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_i8,
        CastKernel::F16ToI8,
        f16,
        i8,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(-3.9)
        ],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i8_to_f32,
        CastKernel::I8ToF32,
        i8,
        f32,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_i8,
        CastKernel::F32ToI8,
        f32,
        i8,
        [0.0_f32, 1.0_f32, 2.1_f32, -3.9_f32],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i8_to_f64,
        CastKernel::I8ToF64,
        i8,
        f64,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_i8,
        CastKernel::F64ToI8,
        f64,
        i8,
        [0.0_f64, 1.0_f64, 2.1_f64, -3.9_f64],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i16_to_f16,
        CastKernel::I16ToF16,
        i16,
        f16,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_i16,
        CastKernel::F16ToI16,
        f16,
        i16,
        [
            f16::from_f32(32767.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(-3.9)
        ],
        // The value 32767.0 is rounded up because f16 has limited precision.
        // The rounded up value's higher bits are discarded according
        // to two’s complement arithmetic.
        [-32768, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i16_to_f32,
        CastKernel::I16ToF32,
        i16,
        f32,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_i16,
        CastKernel::F32ToI16,
        f32,
        i16,
        [0.0_f32, 1.0_f32, 2.1_f32, -3.9_f32],
        [0_i16, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i16_to_f64,
        CastKernel::I16ToF64,
        i16,
        f64,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_i16,
        CastKernel::F64ToI16,
        f64,
        i16,
        [0.0_f64, 1.0_f64, 2.1_f64, -3.9_f64],
        [0_i16, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i32_to_f16,
        CastKernel::I32ToF16,
        i32,
        f16,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_i32,
        CastKernel::F16ToI32,
        f16,
        i32,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(-3.9)
        ],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_i32_to_f32,
        CastKernel::I32ToF32,
        i32,
        f32,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_i32,
        CastKernel::F32ToI32,
        f32,
        i32,
        [0.0_f32, 1.0_f32, 2.1_f32, -3.9_f32],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_i32_to_f64,
        CastKernel::I32ToF64,
        i32,
        f64,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_i32,
        CastKernel::F64ToI32,
        f64,
        i32,
        [0.0_f64, 1.0_f64, 2.1_f64, -3.9_f64],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_i64_to_f16,
        CastKernel::I64ToF16,
        i64,
        f16,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_i64,
        CastKernel::F16ToI64,
        f16,
        i64,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(-3.9)
        ],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_i64_to_f32,
        CastKernel::I64ToF32,
        i64,
        f32,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_i64,
        CastKernel::F32ToI64,
        f32,
        i64,
        [0.0_f32, 1.0_f32, 2.1_f32, -3.9_f32],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_i64_to_f64,
        CastKernel::I64ToF64,
        i64,
        f64,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_i64,
        CastKernel::F64ToI64,
        f64,
        i64,
        [0.0_f64, 1.0_f64, 2.1_f64, -3.9_f64],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_u8_to_f16,
        CastKernel::U8ToF16,
        u8,
        f16,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_u8,
        CastKernel::F16ToU8,
        f16,
        u8,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(3.9)
        ],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u8_to_f32,
        CastKernel::U8ToF32,
        u8,
        f32,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_u8,
        CastKernel::F32ToU8,
        f32,
        u8,
        [0.0_f32, 1.0_f32, 2.1_f32, 3.9_f32],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u8_to_f64,
        CastKernel::U8ToF64,
        u8,
        f64,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_u8,
        CastKernel::F64ToU8,
        f64,
        u8,
        [0.0_f64, 1.0_f64, 2.1_f64, 3.9_f64],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u16_to_f16,
        CastKernel::U16ToF16,
        u16,
        f16,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_u16,
        CastKernel::F16ToU16,
        f16,
        u16,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(3.9)
        ],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u16_to_f32,
        CastKernel::U16ToF32,
        u16,
        f32,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_u16,
        CastKernel::F32ToU16,
        f32,
        u16,
        [0.0_f32, 1.0_f32, 2.1_f32, 3.9_f32],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u16_to_f64,
        CastKernel::U16ToF64,
        u16,
        f64,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_u16,
        CastKernel::F64ToU16,
        f64,
        u16,
        [0.0_f64, 1.0_f64, 2.1_f64, 3.9_f64],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u32_to_f16,
        CastKernel::U32ToF16,
        u32,
        f16,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_u32,
        CastKernel::F16ToU32,
        f16,
        u32,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(3.9)
        ],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_u32_to_f32,
        CastKernel::U32ToF32,
        u32,
        f32,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_u32,
        CastKernel::F32ToU32,
        f32,
        u32,
        [0.0_f32, 1.0_f32, 2.1_f32, 3.9_f32],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_u32_to_f64,
        CastKernel::U32ToF64,
        u32,
        f64,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_u32,
        CastKernel::F64ToU32,
        f64,
        u32,
        [0.0_f64, 1.0_f64, 2.1_f64, 3.9_f64],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_u64_to_f16,
        CastKernel::U64ToF16,
        u64,
        f16,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0)
        ]
    );

    generate_cast_test!(
        test_cast_f16_to_u64,
        CastKernel::F16ToU64,
        f16,
        u64,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.1),
            f16::from_f32(3.9)
        ],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_u64_to_f32,
        CastKernel::U64ToF32,
        u64,
        f32,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32]
    );

    generate_cast_test!(
        test_cast_f32_to_u64,
        CastKernel::F32ToU64,
        f32,
        u64,
        [0.0_f32, 1.0_f32, 2.1_f32, 3.9_f32],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_u64_to_f64,
        CastKernel::U64ToF64,
        u64,
        f64,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]
    );

    generate_cast_test!(
        test_cast_f64_to_u64,
        CastKernel::F64ToU64,
        f64,
        u64,
        [0.0_f64, 1.0_f64, 2.1_f64, 3.9_f64],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_i8_to_i16,
        CastKernel::I8ToI16,
        i8,
        i16,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [0_i16, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i16_to_i8,
        CastKernel::I16ToI8,
        i16,
        i8,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i8_to_i32,
        CastKernel::I8ToI32,
        i8,
        i32,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_i32_to_i8,
        CastKernel::I32ToI8,
        i32,
        i8,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i8_to_i64,
        CastKernel::I8ToI64,
        i8,
        i64,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_i64_to_i8,
        CastKernel::I64ToI8,
        i64,
        i8,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [0_i8, 1_i8, 2_i8, -3_i8]
    );

    generate_cast_test!(
        test_cast_i16_to_i32,
        CastKernel::I16ToI32,
        i16,
        i32,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_i32_to_i16,
        CastKernel::I32ToI16,
        i32,
        i16,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [0_i16, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i16_to_i64,
        CastKernel::I16ToI64,
        i16,
        i64,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_i64_to_i16,
        CastKernel::I64ToI16,
        i64,
        i16,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [0_i16, 1_i16, 2_i16, -3_i16]
    );

    generate_cast_test!(
        test_cast_i32_to_i64,
        CastKernel::I32ToI64,
        i32,
        i64,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [0_i64, 1_i64, 2_i64, -3_i64]
    );

    generate_cast_test!(
        test_cast_i64_to_i32,
        CastKernel::I64ToI32,
        i64,
        i32,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [0_i32, 1_i32, 2_i32, -3_i32]
    );

    generate_cast_test!(
        test_cast_u8_to_u16,
        CastKernel::U8ToU16,
        u8,
        u16,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u16_to_u8,
        CastKernel::U16ToU8,
        u16,
        u8,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u8_to_u32,
        CastKernel::U8ToU32,
        u8,
        u32,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_u32_to_u8,
        CastKernel::U32ToU8,
        u32,
        u8,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u8_to_u64,
        CastKernel::U8ToU64,
        u8,
        u64,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_u64_to_u8,
        CastKernel::U64ToU8,
        u64,
        u8,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [0_u8, 1_u8, 2_u8, 3_u8]
    );

    generate_cast_test!(
        test_cast_u16_to_u32,
        CastKernel::U16ToU32,
        u16,
        u32,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_u32_to_u16,
        CastKernel::U32ToU16,
        u32,
        u16,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u16_to_u64,
        CastKernel::U16ToU64,
        u16,
        u64,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_u64_to_u16,
        CastKernel::U64ToU16,
        u64,
        u16,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [0_u16, 1_u16, 2_u16, 3_u16]
    );

    generate_cast_test!(
        test_cast_u32_to_u64,
        CastKernel::U32ToU64,
        u32,
        u64,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [0_u64, 1_u64, 2_u64, 3_u64]
    );

    generate_cast_test!(
        test_cast_u64_to_u32,
        CastKernel::U64ToU32,
        u64,
        u32,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [0_u32, 1_u32, 2_u32, 3_u32]
    );

    generate_cast_test!(
        test_cast_f16_to_bool,
        CastKernel::F16ToBool,
        f16,
        bool,
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0)
        ],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_f32_to_bool,
        CastKernel::F32ToBool,
        f32,
        bool,
        [0.0_f32, 1.0_f32, 2.0_f32, -3.0_f32],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_f64_to_bool,
        CastKernel::F64ToBool,
        f64,
        bool,
        [0.0_f64, 1.0_f64, 2.0_f64, -3.0_f64],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_i8_to_bool,
        CastKernel::I8ToBool,
        i8,
        bool,
        [0_i8, 1_i8, 2_i8, -3_i8],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_u8_to_bool,
        CastKernel::U8ToBool,
        u8,
        bool,
        [0_u8, 1_u8, 2_u8, 3_u8],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_i16_to_bool,
        CastKernel::I16ToBool,
        i16,
        bool,
        [0_i16, 1_i16, 2_i16, -3_i16],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_u16_to_bool,
        CastKernel::U16ToBool,
        u16,
        bool,
        [0_u16, 1_u16, 2_u16, 3_u16],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_i32_to_bool,
        CastKernel::I32ToBool,
        i32,
        bool,
        [0_i32, 1_i32, 2_i32, -3_i32],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_u32_to_bool,
        CastKernel::U32ToBool,
        u32,
        bool,
        [0_u32, 1_u32, 2_u32, 3_u32],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_i64_to_bool,
        CastKernel::I64ToBool,
        i64,
        bool,
        [0_i64, 1_i64, 2_i64, -3_i64],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_u64_to_bool,
        CastKernel::U64ToBool,
        u64,
        bool,
        [0_u64, 1_u64, 2_u64, 3_u64],
        [false, true, true, true]
    );

    generate_cast_test!(
        test_cast_bool_to_f16,
        CastKernel::BoolToF16,
        bool,
        f16,
        [false, true, false, true],
        [
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0)
        ]
    );

    generate_cast_test!(
        test_cast_bool_to_f32,
        CastKernel::BoolToF32,
        bool,
        f32,
        [false, true, false, true],
        [0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32]
    );

    generate_cast_test!(
        test_cast_bool_to_f64,
        CastKernel::BoolToF64,
        bool,
        f64,
        [false, true, false, true],
        [0.0_f64, 1.0_f64, 0.0_f64, 1.0_f64]
    );

    generate_cast_test!(
        test_cast_bool_to_i8,
        CastKernel::BoolToI8,
        bool,
        i8,
        [false, true, false, true],
        [0_i8, 1_i8, 0_i8, 1_i8]
    );

    generate_cast_test!(
        test_cast_bool_to_u8,
        CastKernel::BoolToU8,
        bool,
        u8,
        [false, true, false, true],
        [0_u8, 1_u8, 0_u8, 1_u8]
    );

    generate_cast_test!(
        test_cast_bool_to_i16,
        CastKernel::BoolToI16,
        bool,
        i16,
        [false, true, false, true],
        [0_i16, 1_i16, 0_i16, 1_i16]
    );

    generate_cast_test!(
        test_cast_bool_to_u16,
        CastKernel::BoolToU16,
        bool,
        u16,
        [false, true, false, true],
        [0_u16, 1_u16, 0_u16, 1_u16]
    );

    generate_cast_test!(
        test_cast_bool_to_i32,
        CastKernel::BoolToI32,
        bool,
        i32,
        [false, true, false, true],
        [0_i32, 1_i32, 0_i32, 1_i32]
    );

    generate_cast_test!(
        test_cast_bool_to_u32,
        CastKernel::BoolToU32,
        bool,
        u32,
        [false, true, false, true],
        [0_u32, 1_u32, 0_u32, 1_u32]
    );

    generate_cast_test!(
        test_cast_bool_to_i64,
        CastKernel::BoolToI64,
        bool,
        i64,
        [false, true, false, true],
        [0_i64, 1_i64, 0_i64, 1_i64]
    );

    generate_cast_test!(
        test_cast_bool_to_u64,
        CastKernel::BoolToU64,
        bool,
        u64,
        [false, true, false, true],
        [0_u64, 1_u64, 0_u64, 1_u64]
    );

    #[test]
    fn test_cast_f64_to_f32_inf() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![1, 5];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![1e40, -1e40, f64::NAN, -f64::NAN, 1.0])
            .unwrap();

        let f = utils::load_cast_kernel(&device, CastKernel::F64ToF32).unwrap();

        let output_shape = x_shape.clone();
        let mut out_data = device
            .alloc_zeros::<f32>(output_shape.iter().product())
            .unwrap();

        unsafe {
            unary::explicit_io_types_compute(f, &x_data, &mut out_data).unwrap();
        }

        let result: Vec<f32> = device.dtoh_sync_copy(&out_data).unwrap();
        assert_eq!(result[0], f32::INFINITY);
        assert_eq!(result[1], f32::NEG_INFINITY);
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        assert_eq!(result[4], 1.0);
    }
}
