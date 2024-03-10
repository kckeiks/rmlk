use crate::alloc::TensorAllocator;
use crate::tensor::op::Op;
use crate::tensor::raw::{LightTensor, RawTensorPtr, MAX_DIMS};
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{RecvError, SyncSender};

pub type Result<T> = std::result::Result<T, ComputeError>;

pub enum ComputeError {
    WorkerBusy,
}

pub struct Worker {
    message_tx: SyncSender<Message>,
    busy: AtomicBool,
}

impl Worker {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        std::thread::spawn(move || {
            loop {
                match rx.recv() {
                    Ok(m) => match m {
                        Message::Work { job } => {
                            // Todo:
                        }
                        Message::Shutdown => {
                            break;
                        }
                    },
                    Err(e) => {
                        // Client dropped. There is nothing else to do.
                        break;
                    }
                }
            }
        });

        Self {
            message_tx: tx,
            busy: AtomicBool::new(false),
        }
    }

    pub fn execute(&self, jon: Job) -> Result<()> {
        todo!()
    }

    pub fn shutdown(self) {}
}

pub enum Message {
    Work { job: Job },
    Shutdown,
}

pub struct Job {
    plan: Plan,
    finished: SyncSender<usize>,
}

pub enum Plan {
    Add {
        a: LightTensor,
        b: LightTensor,
        dst: LightTensor,
        row_count: usize,
        index: usize,
        total: usize,
    },
}

// Graph and tensors can only exist in a single-thread environment
// so this is ok.
unsafe impl Send for Plan {}

impl Job {
    fn compute(self) -> Result<()> {
        match self.plan {
            Plan::Add {
                a,
                b,
                dst,
                row_count,
                index,
                total,
            } => {
                compute_forward_add(a, b, dst, index, total)?;
            }
            _ => unimplemented!(),
        };

        Ok(())
    }
}

pub fn compute_forward_add(
    mut a: LightTensor,
    mut b: LightTensor,
    mut dst: LightTensor,
    index: usize,
    total: usize,
) -> Result<()> {
    // Todo: panic on overflow.
    let row_count = a.shape[1] * a.shape[2] * a.shape[3];
    let rows_per_thread = (row_count + total - 1) / total;

    let start = rows_per_thread * index;
    let end = core::cmp::min(start + rows_per_thread, row_count);
    for i in start..end {
        let i03 = i / (a.shape[2] * a.shape[1]);
        let i02 = (i - i03 * a.shape[2] * a.shape[1]) / a.shape[1];
        let i01 = (i - i03 * a.shape[2] * a.shape[1]) - i02 * a.shape[1];

        let i13 = i03 % b.shape[3];
        let i12 = i02 % b.shape[2];
        let i11 = i01 % b.shape[1];
        let nr0 = a.shape[0] / b.shape[0];

        let dst_ptr = unsafe {
            let ptr = dst.data.as_mut().as_mut_ptr().cast() as *mut f32;
            core::slice::from_raw_parts_mut(
                // ptr.offset(
                //     (i03 * dst.stride[3] + i02 * dst.stride[2] + i01 * dst.stride[1]) as isize,
                // ),
                ptr.offset((i03 + i02 + i01) as isize),
                nr0,
            )
        };
        let src_a_ptr = unsafe {
            let ptr = a.data.as_mut().as_mut_ptr() as *mut f32;
            core::slice::from_raw_parts_mut(
                // ptr.offset((i03 * a.stride[3] + i02 * a.stride[2] + i01 * a.stride[1]) as isize),
                ptr.offset((i03 + i02 + i01) as isize),
                nr0,
            )
        };
        let src_b_ptr = unsafe {
            let ptr = b.data.as_mut().as_mut_ptr() as *mut f32;
            core::slice::from_raw_parts_mut(
                // ptr.offset((i03 * a.stride[3] + i02 * a.stride[2] + i01 * a.stride[1]) as isize),
                ptr.offset((i03 + i02 + i01) as isize),
                nr0,
            )
        };

        for i in 0..nr0 {
            for j in i..b.shape[0] {
                dst_ptr[j] = src_a_ptr[j] + src_b_ptr[j];
            }
        }

        //  { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
    }
    //
    // for (int ir = ir0; ir < ir1; ++ir) {
    //     // src1 is broadcastable across src0 and dst in i1, i2, i3
    //     const int64_t i03 = ir/(ne02*ne01);
    //     const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
    //     const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);
    //
    //     const int64_t i13 = i03 % ne13;
    //     const int64_t i12 = i02 % ne12;
    //     const int64_t i11 = i01 % ne11;
    //     const int64_t nr0 = ne00 / ne10;
    //
    //     float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
    //     float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
    //     float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
    //
    //     for (int64_t r = 0; r < nr0; ++r) {
    //         ggml_vec_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
    //     }
    // }
    Ok(())
}
