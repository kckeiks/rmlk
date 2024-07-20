use rmlk_ir::DataType;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<T> {
    data: Option<T>,
    dtype: DataType,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(dtype: DataType) -> Self {
        Self {
            data: None,
            dtype,
            shape: Vec::new(),
            stride: Vec::new(),
        }
    }

    pub fn new_with_shape(dtype: DataType, shape: Vec<usize>) -> Self {
        let dims = shape.len();
        let mut stride = vec![0usize; dims];
        stride[dims - 1] = 1;
        for i in (0..dims - 1).rev() {
            stride[i] += stride[i + 1] * shape[i + 1];
        }

        Self {
            data: None,
            dtype,
            shape,
            stride,
        }
    }

    pub fn new_init(data: T, dtype: DataType, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        Self {
            data: Some(data),
            dtype,
            shape,
            stride,
        }
    }

    pub fn init(&mut self, data: T) {
        self.data = Some(data);
    }

    pub fn is_init(&self) -> bool {
        self.data.is_some()
    }

    pub fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> Option<&mut T> {
        self.data.as_mut()
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn reshape(&mut self, shape: Vec<usize>) {
        self.shape = shape;
        let dims = self.shape.len();
        let mut stride = vec![0usize; dims];
        calculate_stride(self.shape.as_slice(), &mut stride.as_mut_slice());
    }

    pub fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
}

fn calculate_stride(shape: &[usize], stride: &mut [usize]) {
    let dims = shape.len();

    debug_assert_eq!(dims, stride.len());

    stride[dims - 1] = 1;
    for i in (0..dims - 1).rev() {
        stride[i] += stride[i + 1] * shape[i + 1];
    }
}
