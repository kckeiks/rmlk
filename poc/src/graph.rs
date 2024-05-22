use crate::alloc::TensorAllocator;
use crate::compute;
use crate::compute::{ComputeError, Job, Plan, Worker};
use crate::tensor::op::Op;
use crate::tensor::raw::RawTensorPtr;
use std::alloc::{Allocator, Global};
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
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

impl GraphBuilder<Global> {
    pub(crate) fn new() -> Self {
        Self {
            alloc: Global,
            order: Order::EvalOrderLeftToRight,
        }
    }
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

        let mut pool = Vec::new_in(self.alloc.clone());
        pool.push(Worker::new());

        Ok(Graph {
            nodes,
            leaves,
            order: self.order,
            pool,
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

        let mut pool = Vec::new_in(self.alloc.clone());
        pool.push(Worker::new());

        Ok(Graph {
            nodes,
            leaves,
            order: self.order,
            pool,
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
    pool: Vec<Worker, A>,
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
                    let (tx, rx) = mpsc::sync_channel(1);
                    let worker = self.pool.first().unwrap();
                    let a = dst.src[0]
                        .as_ref()
                        .expect("At least two operands to exist")
                        .try_borrow()
                        .map_err(|_| GraphError::TensorNotFound)?;
                    let b = dst.src[1]
                        .as_ref()
                        .expect("At least two operands to exist")
                        .try_borrow()
                        .map_err(|_| GraphError::TensorNotFound)?;
                    let job = Job {
                        plan: Plan::Add {
                            a: unsafe { a.light_tensor() },
                            b: unsafe { b.light_tensor() },
                            dst: unsafe { dst.light_tensor() },
                            index: 0,
                            total: 1,
                        },
                        finished: tx,
                    };
                    worker.execute(job).unwrap();
                    let r = rx.recv().unwrap();
                }
                _ => unimplemented!(),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::graph::GraphBuilder;
    use crate::tensor::{Builder, Tensor};
    use log::debug;

    #[test]
    fn test_graph_compute_1d() {
        let mut builder = Builder::default();
        let shape = [10];
        let mut a = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            a.set(&[i], i as f32).unwrap();
        }
        let mut b = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            b.set(&[i], i as f32).unwrap();
        }
        let mut c = a.add(b).unwrap();
        // Todo: `add` should initialize?
        for i in 0..shape[0] {
            c.set(&[i], 0.0f32).unwrap();
        }
        let mut graph_builder = GraphBuilder::new();
        let graph = graph_builder.build_graph_rec(c.ptr()).unwrap();
        graph.compute().unwrap();

        let mut result = Vec::new();
        for i in 0..shape[0] {
            let val = c.get(&[i]).unwrap();
            result.push(val);
        }

        let expected = vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];

        debug!("result={result:?}");
        debug!("expected={expected:?}");

        assert_eq!(result, expected);
    }

    #[test]
    fn test_graph_compute_2d() {
        let mut builder = Builder::default();
        let shape = [10, 10];
        let mut a = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                a.set(&[i, j], j as f32).unwrap();
            }
        }
        let mut b = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                b.set(&[i, j], j as f32).unwrap();
            }
        }
        let mut c = a.add(b).unwrap();
        // Todo: `add` should initialize?
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                c.set(&[i, j], 0.0f32).unwrap();
            }
        }
        let mut graph_builder = GraphBuilder::new();
        let graph = graph_builder.build_graph_rec(c.ptr()).unwrap();
        graph.compute().unwrap();

        let mut result = Vec::new();
        for i in 0..shape[0] {
            let mut row = Vec::new();
            for j in 0..shape[1] {
                let val = c.get(&[i, j]).unwrap();
                row.push(val);
            }
            result.push(row);
        }

        let expected = vec![
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        ];

        debug!("result={result:?}");
        debug!("expected={expected:?}");

        assert_eq!(result, expected);
    }

    #[test]
    fn test_graph_compute_3d() {
        let mut builder = Builder::default();
        let shape = [10, 10, 10];
        let mut a = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    a.set(&[i, j, k], k as f32).unwrap();
                }
            }
        }
        let mut b = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    b.set(&[i, j, k], k as f32).unwrap();
                }
            }
        }
        let mut c = a.add(b).unwrap();
        // Todo: `add` should initialize?
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    c.set(&[i, j, k], 0.0f32).unwrap();
                }
            }
        }
        let mut graph_builder = GraphBuilder::new();
        let graph = graph_builder.build_graph_rec(c.ptr()).unwrap();
        graph.compute().unwrap();

        let mut result = Vec::new();
        for i in 0..shape[0] {
            let mut row = Vec::new();
            for j in 0..shape[1] {
                let mut col = Vec::new();
                for k in 0..shape[2] {
                    let val = c.get(&[i, j, k]).unwrap();
                    col.push(val);
                }
                row.push(col);
            }
            result.push(row);
        }

        let mut expected = Vec::new();
        for i in 0..shape[0] {
            expected.push(vec![
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            ]);
        }

        debug!("result={result:?}");
        debug!("expected={expected:?}");

        assert_eq!(result, expected);
    }

    #[test]
    fn test_graph_compute_4d() {
        let mut builder = Builder::default();
        let shape = [10, 10, 10, 10];
        let mut a = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        a.set(&[i, j, k, l], l as f32).unwrap();
                    }
                }
            }
        }
        let mut b = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        b.set(&[i, j, k, l], l as f32).unwrap();
                    }
                }
            }
        }
        let mut c = a.add(b).unwrap();
        // Todo: `add` should initialize?
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        c.set(&[i, j, k, l], 0.0f32).unwrap();
                    }
                }
            }
        }
        let mut graph_builder = GraphBuilder::new();
        let graph = graph_builder.build_graph_rec(c.ptr()).unwrap();
        graph.compute().unwrap();

        let mut result = Vec::new();
        for i in 0..shape[0] {
            let mut arr2 = Vec::new();
            for j in 0..shape[1] {
                let mut arr3 = Vec::new();
                for k in 0..shape[2] {
                    let mut arr4 = Vec::new();
                    for l in 0..shape[3] {
                        let val = c.get(&[i, j, k, l]).unwrap();
                        arr4.push(val);
                    }
                    arr3.push(arr4);
                }
                arr2.push(arr3);
            }
            result.push(arr2);
        }

        let mut expected = Vec::new();
        for i in 0..shape[0] {
            let mut row = Vec::new();
            for j in 0..shape[1] {
                row.push(vec![
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                    vec![0.0_f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                ]);
            }
            expected.push(row);
        }

        debug!("result={result:?}");
        debug!("expected={expected:?}");

        assert_eq!(result, expected);
    }
}
