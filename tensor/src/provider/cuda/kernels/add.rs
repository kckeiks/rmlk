use rmlk_cuda::BINARY_ADD;

pub const MODULE_NAME: &str = "binary_add";
pub const FWD_FN_NAMES: [&'static str; 3] = ["badd_fwd_f16", "badd_fwd_f32", "badd_fwd_f64"];
pub const PTX_SRC: &str = BINARY_ADD;
