use crate::kernels::BINARY_MUL;

pub const MODULE_NAME: &str = "binary_mul";
pub const FWD_FN_NAMES: [&'static str; 3] = ["bmul_fwd_f16", "bmul_fwd_f32", "bmul_fwd_f64"];
pub const PTX_SRC: &str = BINARY_MUL;
