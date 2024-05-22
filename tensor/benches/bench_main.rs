#![feature(allocator_api)]

mod cuda;

use criterion::criterion_main;
criterion_main!(cuda::add::benches,);
