use crate::traverse::{OnnxGraphTraverser, Result, TraversalError};
use rmlk_schema::onnx::{NodeProto, TensorProto, ValueInfoProto};
use std::collections::HashSet;

pub struct ListOps {
    pub ops: HashSet<String>,
}

impl<'a> OnnxGraphTraverser<'a> for ListOps {
    fn check_input(&mut self, _: ValueInfoProto<'a>) -> Result<bool> {
        Ok(false)
    }

    fn check_output(&mut self, _: ValueInfoProto<'a>) -> Result<bool> {
        Ok(false)
    }

    fn check_initializer(&mut self, _: TensorProto<'a>) -> Result<bool> {
        Ok(false)
    }

    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool> {
        match node.op_type {
            Some(ty) => {
                self.ops.insert(ty.to_string());
                Ok(false)
            }
            None => {
                log::error!(
                    "inner node does not have an operation type: {:?}",
                    node.name
                );
                Err(TraversalError::InvalidInnerNode)
            }
        }
    }
}
