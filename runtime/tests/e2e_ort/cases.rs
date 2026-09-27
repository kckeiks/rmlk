//! Registered full-model e2e cases.
//!
//! ResNet (item 60) and Llama (item 61) add entries here. The shared runner
//! (item 59) iterates [`all`].

/// One tensor feed: artifact role → ONNX graph input name, with optional shape
/// override when the sidecar omits `shape`.
#[derive(Debug, Clone)]
pub struct InputBinding {
    /// Role in the artifact sidecar (`input`, …).
    pub role: &'static str,
    /// Graph input name expected by the ONNX model.
    pub name: &'static str,
}

/// Outputs compared between rmlk and ORT.
#[derive(Debug, Clone)]
pub struct OutputBinding {
    pub name: &'static str,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub id: &'static str,
    pub artifact_id: &'static str,
    pub atol: f32,
    pub rtol: f32,
    pub inputs: &'static [InputBinding],
    pub outputs: &'static [OutputBinding],
}

/// All registered full-model cases. Empty until items 60 / 61 land.
pub fn all() -> &'static [Case] {
    &[]
}
