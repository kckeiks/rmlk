//! Registered full-model e2e cases.

/// How to turn a resolved artifact role into a model feed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // RawF32 reserved for non-image feed cases.
pub enum FeedKind {
    /// Little-endian f32 blob; sidecar must include `shape` (+ `dtype` f32).
    RawF32,
    /// JPEG/PNG asset; torchvision-style ImageNet ResNet preprocess → NCHW f32.
    ImageNetResNet,
}

/// One tensor feed: artifact role → ONNX graph input name.
#[derive(Debug, Clone)]
pub struct InputBinding {
    /// Role in the artifact sidecar (`image`, `input`, …).
    pub role: &'static str,
    /// Graph input name expected by the ONNX model.
    pub name: &'static str,
    pub kind: FeedKind,
}

/// Outputs compared between rmlk and ORT.
#[derive(Debug, Clone)]
pub struct OutputBinding {
    pub name: &'static str,
}

/// How the shared runner executes a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseKind {
    /// Single forward: resolve input roles, compare named float outputs.
    Feed,
    /// Greedy decode loop (Llama); compare next-token ids + last-row logits.
    LlamaGreedy { max_new_tokens: usize },
}

#[derive(Debug, Clone)]
pub struct Case {
    pub id: &'static str,
    pub artifact_id: &'static str,
    pub kind: CaseKind,
    pub atol: f32,
    pub rtol: f32,
    pub inputs: &'static [InputBinding],
    pub outputs: &'static [OutputBinding],
    /// If set, rmlk (and ORT) top-1 class id on the first output must match.
    pub expect_top1: Option<usize>,
}

const RESNET34_INPUTS: &[InputBinding] = &[InputBinding {
    role: "image",
    name: "input",
    kind: FeedKind::ImageNetResNet,
}];

const RESNET34_OUTPUTS: &[OutputBinding] = &[OutputBinding { name: "output" }];

/// ResNet34 ImageNet-1k, fixed dog.jpeg fixture (top-1 class 207 golden_retriever).
pub const RESNET34: Case = Case {
    id: "resnet34",
    artifact_id: "2026-09-27",
    kind: CaseKind::Feed,
    // Full-model GPU vs GPU: looser than node-suite. Peak abs ~2e-3 observed;
    // small logits are limited by atol (rtol barely helps near zero).
    atol: 5e-3,
    rtol: 1e-2,
    inputs: RESNET34_INPUTS,
    outputs: RESNET34_OUTPUTS,
    // This photo's ORT top-1 under the documented preprocess (not 208 Labrador).
    expect_top1: Some(207),
};

/// Llama-3.2-3B-Instruct greedy decode (short prompt, few tokens).
pub const LLAMA32: Case = Case {
    id: "llama3.2",
    artifact_id: "2026-09-27",
    kind: CaseKind::LlamaGreedy { max_new_tokens: 8 },
    // LLM decode vs ORT: start a bit looser than ResNet; tighten if stable.
    atol: 1e-2,
    rtol: 1e-2,
    inputs: &[],
    outputs: &[],
    expect_top1: None,
};

/// All registered full-model cases.
pub fn all() -> &'static [Case] {
    &[RESNET34, LLAMA32]
}
