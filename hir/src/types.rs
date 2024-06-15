use crate::tensor::TensorShape;
use std::collections::HashMap;

pub struct Type {
    value: Option<InnerType>,
    denotation: Option<String>,
}

enum InnerType {
    Map { map: HashMap<i32, Type> },
    Tensor { elem_type: i32, shape: TensorShape },
    Sequence { elem_type: Vec<Type> },
    SparseTensor { elem_type: i32, shape: TensorShape },
}
