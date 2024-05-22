use criterion::{black_box, criterion_group, Criterion};
use cudarc::driver::CudaDevice;
use rmlk_tensor::device::cuda::cuda::Cuda;
use rmlk_tensor::device::Device;
use rmlk_tensor::dtype::DType;
use rmlk_tensor::op::Op;
use rmlk_tensor::tensor::Tensor;
use std::alloc::Global;
use std::sync::Arc;
use std::time::Duration;

fn add(shape: Vec<usize>) -> Vec<f32> {
    let device = CudaDevice::new(0).unwrap();
    let cuda = Cuda::new(device, Global);

    let elem_num = shape.iter().product::<usize>();
    let dims = shape.len();
    let mut lhs_strides = vec![0usize; dims];
    lhs_strides[0] = 1;
    for i in 1..dims {
        lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
    }
    let mut rhs_strides = vec![0usize; dims];
    rhs_strides[0] = 1;
    for i in 1..dims {
        rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
    }

    let mut lhs_tensor = Tensor::new(DType::F32, shape.to_vec(), lhs_strides.to_vec());
    let mut rhs_tensor = Tensor::new(DType::F32, shape.to_vec(), rhs_strides.to_vec());
    let mut out_tensor = Tensor::new(DType::F32, shape.to_vec(), lhs_strides.to_vec());

    let lhs_data = cuda.htod_f32(vec![1.0; elem_num]).unwrap();
    let rhs_data = cuda.htod_f32(vec![1.0; elem_num]).unwrap();

    lhs_tensor.set_data(Arc::new(lhs_data));
    rhs_tensor.set_data(Arc::new(rhs_data));

    cuda.forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
        .unwrap();
    cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add-cuda-3x3x3", |b| {
        b.iter(|| add(black_box(vec![3, 3, 3])))
    });
    c.bench_function("add-cuda-3x3x3x3", |b| {
        b.iter(|| add(black_box(vec![3, 3, 3, 3])))
    });
    c.bench_function("add-cuda-100x500x900", |b| {
        b.iter(|| add(black_box(vec![100, 500, 900])))
    });
    c.bench_function("add-cuda-100x500x900x1", |b| {
        b.iter(|| add(black_box(vec![100, 500, 900, 1])))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(7));
    targets = criterion_benchmark
);
