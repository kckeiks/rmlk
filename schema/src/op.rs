use crate::attributes;
use crate::error::Error;
use crate::model::StringStringEntryProto;
use crate::node::Node;
use crate::onnx::OperatorSetIdProto;
use crate::value::ValueInfo;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum Op {
    NoOp = 0,
    Add = 1,
    Cast = 2,
    Concat = 3,
    Conv = 4,
    Const = 5,
    ConstantOfShape = 6,
    Div = 7,
    Expand = 8,
    Equal = 9,
    Flatten = 10,
    Gather = 11,
    Gemm = 12,
    GlobalAveragePool = 13,
    MaxPool = 14,
    Mul = 15,
    MatMul = 16,
    Pow = 17,
    Range = 18,
    Relu = 19,
    ReduceMean = 20,
    Reshape = 21,
    ScatterND = 22,
    Shape = 23,
    Sigmoid = 24,
    Slice = 25,
    Softmax = 26,
    Sqrt = 27,
    Sub = 28,
    Transpose = 29,
    Trilu = 30,
    Unsqueeze = 31,
    Where = 32,
    // Todo: we have two constants variants. Fix it.
    Constant = 33,
}

impl TryFrom<u32> for Op {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let op = match value {
            0 => Self::NoOp,
            1 => Self::Add, // Done.
            2 => Self::Cast,
            3 => Self::Concat,
            4 => Self::Conv,  // Done.
            5 => Self::Const, // Done.
            6 => Self::ConstantOfShape,
            7 => Self::Div,
            8 => Self::Expand,
            9 => Self::Equal,
            10 => Self::Flatten, // Done.
            11 => Self::Gather,
            12 => Self::Gemm,              // Done.
            13 => Self::GlobalAveragePool, // Done.
            14 => Self::MaxPool,           // Done.
            15 => Self::Mul,               // Done.
            16 => Self::MatMul,
            17 => Self::Pow,
            18 => Self::Range,
            19 => Self::Relu, // Done.
            20 => Self::ReduceMean,
            21 => Self::Reshape,
            22 => Self::ScatterND,
            23 => Self::Shape,
            24 => Self::Sigmoid,
            25 => Self::Slice,
            26 => Self::Softmax,
            27 => Self::Sqrt,
            28 => Self::Sub,
            29 => Self::Transpose,
            30 => Self::Unsqueeze,
            31 => Self::Where,
            op => panic!("Unknown operation {op}"),
        };

        Ok(op)
    }
}

impl FromStr for Op {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let op = match s {
            "Add" | "add" => Self::Add, // Done.
            "Cast" | "cast" => Self::Cast,
            "Concat" | "concat" => Self::Concat,
            "Conv" | "conv" => Self::Conv,             // Done.
            "Constant" | "constant" => Self::Constant, // Done.
            "ConstantOfShape" | "constantofshape" => Self::ConstantOfShape,
            "Div" | "div" => Self::Div,
            "Equal" | "equal" => Self::Equal,
            "Expand" | "expand" => Self::Expand,
            "Flatten" | "flatten" => Self::Flatten, // Done.
            "Gather" | "gather" => Self::Gather,
            "Gemm" | "gemm" => Self::Gemm, // Done.
            "GlobalAveragePool" | "globalaveragepool" => Self::GlobalAveragePool, // Done.
            "MaxPool" | "maxpool" => Self::MaxPool, // Done.
            "MatMul" | "matmul" => Self::MatMul,
            "Mul" | "mul" => Self::Mul, // Done.
            "Pow" | "pow" => Self::Pow,
            "Range" | "range" => Self::Range,
            "Relu" | "relu" => Self::Relu, // Done.
            "ReduceMean" | "reducemean" => Self::ReduceMean,
            "Reshape" | "reshape" => Self::Reshape,
            "ScatterND" | "scatternd" => Self::ScatterND,
            "Shape" | "shape" => Self::Shape,
            "Sigmoid" | "sigmoid" => Self::Sigmoid,
            "Slice" | "slice" => Self::Slice,
            "Softmax" | "softmax" => Self::Softmax,
            "Sqrt" | "sqrt" => Self::Sqrt,
            "Sub" | "sub" => Self::Sub,
            "Transpose" | "transpose" => Self::Transpose,
            "Unsqueeze" | "unsqueeze" => Self::Unsqueeze,
            "Where" | "where" => Self::Where,
            op => panic!("We do not support operation {op}"),
        };

        Ok(op)
    }
}

#[derive(Deserialize, Serialize)]
pub struct OperatorSetId {
    domain: Option<String>,
    version: i64,
}

impl TryFrom<OperatorSetIdProto<'_>> for OperatorSetId {
    type Error = Error;

    fn try_from(value: OperatorSetIdProto) -> Result<Self, Self::Error> {
        Ok(Self {
            domain: value.domain.map(|cow| cow.to_string()),
            version: value.version.ok_or(Error::MissingField {
                name: "OperatorSetId::version".to_string(),
            })?,
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct Function {
    name: Option<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attribute: Attribute,
    node: Vec<Node>,
    doc_string: Option<String>,
    opset_import: Vec<OperatorSetId>,
    domain: Option<String>,
    overload: Option<String>,
    value_info: Vec<ValueInfo>,
    metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Deserialize, Serialize)]
enum Attribute {
    String { value: Vec<String> },
    Object { value: Vec<attributes::Attribute> },
}
