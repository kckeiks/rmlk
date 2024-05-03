use crate::alloc::TensorAllocator;
use crate::tensor::op::Op;
use crate::tensor::raw::{LightTensor, RawTensorPtr, MAX_DIMS};
use log::debug;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{RecvError, SyncSender};

pub type Result<T> = std::result::Result<T, ComputeError>;

#[derive(Debug)]
pub enum ComputeError {
    WorkerBusy,
    Unknown,
}

pub struct Worker {
    message_tx: SyncSender<Message>,
    busy: AtomicBool,
}

impl Worker {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        std::thread::spawn(move || {
            loop {
                match rx.recv() {
                    Ok(m) => match m {
                        Message::Execute { job } => {
                            if let Err(e) = job.compute() {
                                println!("there was an error while computing: {e:?}");
                            }
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

    pub fn execute(&self, job: Job) -> Result<()> {
        self.message_tx
            .send(Message::Execute { job })
            .map_err(|_| ComputeError::Unknown)
    }

    pub fn shutdown(self) {
        let _ = self.message_tx.send(Message::Shutdown);
    }
}

pub enum Message {
    Execute { job: Job },
    Shutdown,
}

pub struct Job {
    pub plan: Plan,
    pub(crate) finished: SyncSender<usize>,
}

pub enum Plan {
    Add {
        a: LightTensor,
        b: LightTensor,
        dst: LightTensor,
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
                index,
                total,
            } => {
                let result = compute_forward_add(a, b, dst, index, total);
                let _ = self.finished.send(index);
                result?;
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

    debug!("row_count={row_count};rows_per_thread={rows_per_thread};start={start};end={end}");

    for i in start..end {
        let i03 = i / (a.shape[2] * a.shape[1]);
        let i02 = (i - i03 * a.shape[2] * a.shape[1]) / a.shape[1];
        let i01 = (i - i03 * a.shape[2] * a.shape[1]) - i02 * a.shape[1];

        let i13 = i03 % b.shape[3];
        let i12 = i02 % b.shape[2];
        let i11 = i01 % b.shape[1];
        let nr0 = a.shape[0] / b.shape[0];

        debug!("i03={i03};i02={i02};i01={i01};");
        debug!("i13={i13};i12={i12};i11={i11};");

        let dst_ptr = unsafe {
            // Todo: be careful with casting here.
            let offset = (i03 * dst.stride[3] + i02 * dst.stride[2] + i01 * dst.stride[1]);
            let ptr = (&mut dst.data.as_mut()[offset..]).as_mut_ptr() as *mut f32;
            core::slice::from_raw_parts_mut(ptr, b.shape[0])
        };
        let src_a_ptr = unsafe {
            let offset = (i03 * a.stride[3] + i02 * a.stride[2] + i01 * a.stride[1]);
            let ptr = a.data.as_mut()[offset..].as_mut_ptr() as *mut f32;
            core::slice::from_raw_parts_mut(ptr, b.shape[0])
        };
        let src_b_ptr = unsafe {
            let offset = (i03 * a.stride[3] + i02 * a.stride[2] + i01 * a.stride[1]);
            let ptr = b.data.as_mut()[offset..].as_mut_ptr() as *mut f32;
            core::slice::from_raw_parts_mut(ptr, b.shape[0])
        };

        debug!(
            "src_a_ptr={};src_b_ptr={};dst_ptr={};",
            src_a_ptr.len(),
            src_b_ptr.len(),
            dst_ptr.len()
        );

        for i in 0..nr0 {
            for j in i..b.shape[0] {
                dst_ptr[j] = src_a_ptr[j] + src_b_ptr[j];
            }
        }
    }

    Ok(())
}
