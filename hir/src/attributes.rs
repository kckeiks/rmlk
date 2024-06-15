use crate::graph::Graph;
use crate::model::StringStringEntryProto;
use crate::tensor::{SparseTensor, Tensor};
use crate::types::Type;

pub struct Attribute {
    name: String,
    ref_attr_name: Option<String>,
    doc_string: Option<String>,
    ty: AttributeType,
}

enum AttributeType {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Tensor),
    Graph(Graph),
    SparseTensor(SparseTensor),
    Type(Type),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    Graphs(Vec<Graph>),
    SparseTensors(Vec<SparseTensor>),
    Types(Vec<Type>),
}

pub struct ValueInfo {
    pub name: String,
    pub ty: Type,
    pub doc_string: Option<String>,
    pub metadata_props: Vec<StringStringEntryProto>,
}
