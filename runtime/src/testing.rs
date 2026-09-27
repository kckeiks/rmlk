//! Single-op test helpers for CUDA backend unit tests.
//!
//! Builds a one-op graph through [`GraphBuilder`] and [`Builder::from_graph`],
//! so every case exercises the production entry point.

#![allow(dead_code)] // helpers are adopted op-by-op during the section 12 migration

use crate::core::error::InferenceError;
use crate::core::{Builder, Value};
use rmlk_schema::{AsRawBytes, AttributeType, DataType, DataTypeMap, GraphBuilder, Op, Tensor};
use std::collections::HashMap;
use std::fmt::Debug;

/// Fluent builder for a single-op inference test.
///
/// Feeds are wired to the op in declaration order (`input` / `constant`
/// calls). The graph output is always named `y`.
///
/// ```ignore
/// let out = OpTest::new(Op::Add)
///     .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
///     .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
///     .run::<f32>()
///     .unwrap();
/// assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
/// ```
pub struct OpTest {
    op: Op,
    feeds: Vec<Feed>,
    attributes: Vec<(String, AttributeType)>,
    output_dims: Option<Vec<usize>>,
}

enum Feed {
    Input {
        name: String,
        dtype: DataType,
        dims: Vec<usize>,
        value: Value,
    },
    Constant {
        name: String,
        tensor: Tensor,
    },
}

impl OpTest {
    pub fn new(op: Op) -> Self {
        Self {
            op,
            feeds: Vec::new(),
            attributes: Vec::new(),
            output_dims: None,
        }
    }

    /// Add a graph input. Auto-named `in0`, `in1`, …
    pub fn input<T>(mut self, dims: impl AsRef<[usize]>, data: Vec<T>) -> Self
    where
        T: AsRawBytes + DataTypeMap,
        Value: From<(Vec<T>, Vec<usize>)>,
    {
        let dims = dims.as_ref().to_vec();
        let name = format!("in{}", self.feeds.len());
        let value = Value::from((data, dims.clone()));
        self.feeds.push(Feed::Input {
            name,
            dtype: T::data_type(),
            dims,
            value,
        });
        self
    }

    /// Add an initializer feed (not a graph input). Auto-named `const0`, …
    pub fn constant<T>(mut self, dims: impl AsRef<[usize]>, data: Vec<T>) -> Self
    where
        T: AsRawBytes,
    {
        let dims = dims.as_ref().to_vec();
        let name = format!(
            "const{}",
            self.feeds
                .iter()
                .filter(|f| matches!(f, Feed::Constant { .. }))
                .count()
        );
        let tensor = Tensor::from_vec(dims, data).expect("constant tensor");
        self.feeds.push(Feed::Constant { name, tensor });
        self
    }

    /// Attach an attribute to the op.
    pub fn attr(mut self, name: impl Into<String>, value: AttributeType) -> Self {
        self.attributes.push((name.into(), value));
        self
    }

    /// Set the graph output shape.
    ///
    /// Defaults to the first feed's shape when omitted. Required when the op
    /// has no feeds (e.g. `Constant`) or when the output shape differs from
    /// the first feed (broadcast).
    pub fn output(mut self, dims: impl AsRef<[usize]>) -> Self {
        self.output_dims = Some(dims.as_ref().to_vec());
        self
    }

    /// Run the op and return the output tensor as `Vec<T>`.
    pub fn run<T>(self) -> Result<Vec<T>, InferenceError>
    where
        T: DataTypeMap,
        Vec<T>: TryFrom<Value, Error = InferenceError>,
    {
        let mut outputs = self.execute(T::data_type())?;
        let value = outputs
            .remove("y")
            .expect("single-op graph always has output `y`");
        Vec::<T>::try_from(value)
    }

    /// Run the op and return the error. Panics if the run succeeds.
    pub fn run_err(self) -> InferenceError {
        // Output dtype is only used to declare the value node; Bool is a fine
        // placeholder when we expect failure before producing output.
        match self.execute(DataType::Bool) {
            Ok(_) => panic!("expected op to fail, but it succeeded"),
            Err(e) => e,
        }
    }

    fn execute(self, output_dtype: DataType) -> Result<HashMap<String, Value>, InferenceError> {
        let Self {
            op,
            feeds,
            attributes,
            output_dims,
        } = self;

        let output_dims = output_dims.unwrap_or_else(|| {
            feeds
                .iter()
                .find_map(|f| match f {
                    Feed::Input { dims, .. } => Some(dims.clone()),
                    Feed::Constant { tensor, .. } => Some(tensor.dims.clone()),
                })
                .expect("OpTest needs .output([...]) when there are no feeds")
        });

        let mut g = GraphBuilder::new();
        let mut op_inputs = Vec::with_capacity(feeds.len());
        let mut run_inputs = HashMap::new();

        for feed in feeds {
            match feed {
                Feed::Input {
                    name,
                    dtype,
                    dims,
                    value,
                } => {
                    g.input(&name, dtype, &dims)
                        .unwrap_or_else(|e| panic!("GraphBuilder::input({name}): {e}"));
                    op_inputs.push(name.clone());
                    run_inputs.insert(name, value);
                }
                Feed::Constant { name, tensor } => {
                    g.constant(&name, tensor)
                        .unwrap_or_else(|e| panic!("GraphBuilder::constant({name}): {e}"));
                    op_inputs.push(name);
                }
            }
        }

        g.value("y", output_dtype, &output_dims)
            .unwrap_or_else(|e| panic!("GraphBuilder::value(y): {e}"));

        let input_refs: Vec<&str> = op_inputs.iter().map(String::as_str).collect();
        let mut op_builder = g
            .op(op, &input_refs, &["y"])
            .unwrap_or_else(|e| panic!("GraphBuilder::op: {e}"));
        for (name, value) in attributes {
            op_builder = op_builder.attr(name, value);
        }

        g.output("y")
            .unwrap_or_else(|e| panic!("GraphBuilder::output(y): {e}"));
        let graph = g
            .build()
            .unwrap_or_else(|e| panic!("GraphBuilder::build: {e}"));

        let mut instance = Builder::from_graph(graph)
            .and_then(|b| b.build())
            .map_err(|e| InferenceError(anyhow::anyhow!(e)))?;
        instance.run(run_inputs)
    }
}

/// Assert two slices are element-wise close using [`approx`] defaults.
pub fn assert_close<T>(actual: &[T], expected: &[T])
where
    T: approx::AbsDiffEq<Epsilon = T> + Debug + Copy,
{
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.abs_diff_eq(e, T::default_epsilon()),
            "mismatch at index {i}: actual={a:?} expected={e:?}"
        );
    }
}

/// Assert two slices are element-wise close within `epsilon`.
pub fn assert_close_eps<T>(actual: &[T], expected: &[T], epsilon: T)
where
    T: approx::AbsDiffEq<Epsilon = T> + Debug + Copy,
{
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.abs_diff_eq(e, epsilon),
            "mismatch at index {i}: actual={a:?} expected={e:?} (eps={epsilon:?})"
        );
    }
}
