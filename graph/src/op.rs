use std::str::FromStr;

#[derive(Clone, Copy)]
pub enum Op {
    NoOp,
    Add,
    Cast,
    Concat,
    Conv,
    Const,
    ConstantOfShape,
    Div,
    Expand,
    Equal,
    Flatten,
    Gather,
    Gemm,
    GlobalAveragePool,
    MaxPool,
    Mul,
    MatMul,
    Pow,
    Range,
    ReduceMean,
    Relu,
    Reshape,
    ScatterND,
    Shape,
    Sigmoid,
    Slice,
    Softmax,
    Sqrt,
    Sub,
    Transpose,
    Unsqueeze,
    Where,
}

impl FromStr for Op {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let op = match s {
            "Add" => Self::Add,  // Resnet.
            "Cast" => Self::Cast,
            "Concat" => Self::Concat,
            "Conv" => Self::Conv,  // Resnet.
            "Constant" => Self::Const,
            "ConstantOfShape" => Self::ConstantOfShape,
            "Div" => Self::Div,
            "Equal" => Self::Equal,
            "Expand" => Self::Expand,
            "Flatten" => Self::Flatten,  // Resnet.
            "Gather" => Self::Gather,
            "Gemm" => Self::Gemm,  // Resnet.
            "GlobalAveragePool" => Self::GlobalAveragePool,  // Resnet.
            "MaxPool" => Self::MaxPool,  // Resnet.
            "MatMul" => Self::MatMul, // Do not need for resnet.
            "Mul" => Self::Mul, // Do not need for resnet.
            "Pow" => Self::Pow,
            "Range" => Self::Range,
            "Relu" => Self::Relu,  // Resnet.
            "ReduceMean" => Self::ReduceMean,
            "Reshape" => Self::Reshape,
            "ScatterND" => Self::ScatterND,
            "Shape" => Self::Shape,
            "Sigmoid" => Self::Sigmoid,
            "Slice" => Self::Slice,
            "Softmax" => Self::Softmax,
            "Sqrt" => Self::Sqrt,
            "Sub" => Self::Sub, // Do not need for resnet.
            "Transpose" => Self::Transpose,
            "Unsqueeze" => Self::Unsqueeze,
            "Where" => Self::Where,
            op => panic!("We do not support operation {op}"),
        };

        Ok(op)
    }
}
