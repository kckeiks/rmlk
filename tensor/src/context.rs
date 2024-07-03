use crate::provider::Provider;
use crate::tensor::Tensor;
use crate::Result;
use rmlk_ir::DataType;

/// The execution context.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionContext<P> {
    /// The provider for this context.
    provider: P,
    /// All the values for the entire graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    /// For each entry, the order is `inputs` then `outputs`.
    values: Vec<Tensor<P>>,
    /// Offset from where the node's inputs & outputs region begins in `values`.
    offsets: Vec<usize>,
    /// Minimum index value for all the nodes in the graph for this context.
    min_value: usize,
}

impl<P: Provider> ExecutionContext<P> {
    pub fn provider(&self) -> &P {
        todo!()
    }

    pub fn allocate(&self, dtype: DataType, shape: Vec<usize>) -> Result<&mut Tensor<P::Data>> {
        todo!()
    }

    pub fn get_input(&self, index: usize) -> Result<&Tensor<P::Data>> {
        todo!()
    }

    pub fn get_output_mut(&mut self, index: usize) -> Result<&mut Tensor<P::Data>> {
        todo!()
    }
}
