mod pb;

pub use pb::onnx::{NodeProto, ValueInfoProto, TypeProto, Version, OperatorStatus, OperatorSetIdProto, StringStringEntryProto, TensorProto, TensorShapeProto, GraphProto, ModelProto, SparseTensorProto, FunctionProto, AttributeProto, TensorAnnotation};

pub mod tensor_proto {
    pub use super::pb::onnx::mod_TensorProto::*;
}

pub mod ty_proto {
    pub use super::pb::onnx::mod_TypeProto::*;
}

pub mod tensor_shape_proto {
    pub use super::pb::onnx::mod_TensorShapeProto::*;
}

pub mod dimension_proto {
    pub use super::pb::onnx::mod_TensorShapeProto::mod_Dimension::*;
}

pub mod attributte_proto {
    pub use super::pb::onnx::mod_AttributeProto::*;
}