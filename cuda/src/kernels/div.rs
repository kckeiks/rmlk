use crate::ptx::DIV;

pub const MODULE_NAME: &str = "div";
pub const FWD_FN_NAMES: [&'static str; 3] = ["div_fwd_f16", "div_fwd_f32", "div_fwd_f64"];
pub const PTX_SRC: &str = DIV;

