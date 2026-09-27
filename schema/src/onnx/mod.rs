mod node;
mod pb;

pub use pb::onnx::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    SparseTensorProto, StringStringEntryProto, TensorAnnotation, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto,
};

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

pub mod attribute_proto {
    pub use super::pb::onnx::mod_AttributeProto::*;
}

pub use node::{Category, NodeWithMetadata, NodeWithValue, ValueInfoV2};
