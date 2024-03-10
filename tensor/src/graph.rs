use crate::alloc::TensorAllocator;
use crate::compute;
use crate::tensor::op::Op;
use crate::tensor::raw::RawTensorPtr;
use std::alloc::Allocator;
use std::collections::{HashSet, VecDeque};

pub type Result<T> = std::result::Result<T, GraphError>;

pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
}

pub enum Order {
    EvalOrderLeftToRight,
    EvalOrderRightToLeft,
}

pub struct GraphBuilder<A> {
    alloc: A,
    order: Order,
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    pub(crate) fn new_in(alloc: A) -> Result<Self> {
        Ok(Self {
            alloc,
            order: Order::EvalOrderLeftToRight,
        })
    }

    /*                          R
                   F1                      F2
           A                 B                 c (leaf)
        a0 a1 a2       b0  b1  b2  b3
       *leafs*       bb0
                  *leafs*

       <Left to Right> (read src in rev)
       nodes = R, F2, F1, B, A, b3, b2, b1, b0, a2, a1, a0, bb0
       leaf  = c, *b leaves*, *a leaves*
       buf   =  (push on front, pop on back)


      <Right to Left> (read src in order)
       nodes = R, F1, F2, A, B, a0, a1, a2,
       leaf  =  c, *a leaves*
       buf   = b2, b3, b1, b0,  (push on front, pop on back)

    */
    pub fn build_graph<T: TensorAllocator>(
        mut self,
        root: RawTensorPtr<T, T::MetadataAlloc>,
    ) -> Result<Graph<A, T>> {
        // Todo: use what's left from scratch buffer and reuse for these.
        let mut nodes = Vec::new_in(self.alloc.clone());
        let mut leaves = Vec::new_in(self.alloc.clone());
        let mut buf = VecDeque::new_in(self.alloc.clone());
        buf.push_front(root);

        while let Some(next) = buf.pop_back() {
            let raw_t = next.try_borrow().map_err(|_| GraphError::InvalidTensor)?;
            if matches!(self.order, Order::EvalOrderLeftToRight) {
                for ptr in raw_t.src.iter().rev() {
                    if let Some(t) = ptr {
                        buf.push_front(t.clone());
                    }
                }
            } else {
                for ptr in raw_t.src.iter() {
                    if let Some(t) = ptr {
                        buf.push_front(t.clone());
                    }
                }
            }

            if matches!(raw_t.op, Op::NoOp) {
                leaves.push(next.clone());
            } else {
                nodes.push(next.clone());
            }
        }

        Ok(Graph {
            nodes,
            leaves,
            order: self.order,
        })
    }

    // Uses recursion.
    pub fn build_graph_rec<T: TensorAllocator>(
        mut self,
        root: RawTensorPtr<T, T::MetadataAlloc>,
    ) -> Result<Graph<A, T>> {
        let mut nodes = Vec::new_in(self.alloc.clone());
        let mut leaves = Vec::new_in(self.alloc.clone());

        self.visit_parents(&mut nodes, &mut leaves, root.clone())?;

        Ok(Graph {
            nodes,
            leaves,
            order: self.order,
        })
    }

    fn visit_parents<T: TensorAllocator>(
        &self,
        nodes: &mut Vec<RawTensorPtr<T, T::MetadataAlloc>, A>,
        leaves: &mut Vec<RawTensorPtr<T, T::MetadataAlloc>, A>,
        node: RawTensorPtr<T, T::MetadataAlloc>,
    ) -> Result<()> {
        let next = node.try_borrow().map_err(|_| GraphError::InvalidTensor)?;
        if matches!(self.order, Order::EvalOrderLeftToRight) {
            for ptr in next.src.iter().rev() {
                if let Some(t) = ptr {
                    self.visit_parents(nodes, leaves, t.clone());
                }
            }
        } else {
            for ptr in next.src.iter() {
                if let Some(t) = ptr {
                    self.visit_parents(nodes, leaves, t.clone());
                }
            }
        }

        if matches!(next.op, Op::NoOp) {
            leaves.push(node.clone());
        } else {
            nodes.push(node.clone());
        }

        Ok(())
    }
}

pub struct Graph<A: Allocator, T: TensorAllocator> {
    nodes: Vec<RawTensorPtr<T, T::MetadataAlloc>, A>,
    leaves: Vec<RawTensorPtr<T, T::MetadataAlloc>, A>,
    order: Order,
}

impl<A, T> Graph<A, T>
where
    A: Allocator,
    T: TensorAllocator,
{
    pub fn compute(&self) -> Result<()> {
        for node in &self.nodes {
            // Todo: remove unwrap.
            let dst = node.try_borrow().unwrap();
            match dst.op {
                Op::Add => {
                    unimplemented!()
                }
                _ => unimplemented!(),
            }
        }

        Ok(())
    }
}
