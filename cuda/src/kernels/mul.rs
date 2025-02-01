use crate::ptx::MUL;

pub const MODULE_NAME: &str = "mul";
pub const FWD_FN_NAMES: [&'static str; 3] = ["mul_fwd_f16", "mul_fwd_f32", "mul_fwd_f64"];
pub const PTX_SRC: &str = MUL;
