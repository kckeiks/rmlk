use crate::Args;
use anyhow::Result;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    process::Command,
    sync::oneshot,
    task::JoinHandle,
    time::{sleep, timeout},
};

#[derive(Clone, Default, Serialize)]
pub struct Aggregate {
    pub samples: usize,
    pub mean_utilization_pct: Option<f64>,
    pub max_utilization_pct: Option<f64>,
    pub max_memory_used_mib: Option<f64>,
    pub mean_power_w: Option<f64>,
    pub max_power_w: Option<f64>,
    #[serde(skip)]
    utilization_sum: f64,
    #[serde(skip)]
    power_sum: f64,
    #[serde(skip)]
    power_samples: usize,
}
impl Aggregate {
    fn add(&mut self, util: f64, memory: f64, power: Option<f64>) {
        self.samples += 1;
        self.utilization_sum += util;
        self.mean_utilization_pct = Some(self.utilization_sum / self.samples as f64);
        self.max_utilization_pct = Some(self.max_utilization_pct.unwrap_or(util).max(util));
        self.max_memory_used_mib = Some(self.max_memory_used_mib.unwrap_or(memory).max(memory));
        if let Some(power) = power {
            self.power_samples += 1;
            self.power_sum += power;
            self.mean_power_w = Some(self.power_sum / self.power_samples as f64);
            self.max_power_w = Some(self.max_power_w.unwrap_or(power).max(power));
        }
    }
}
#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub enabled: bool,
    pub device_name: Option<String>,
    pub overall: Aggregate,
    pub by_phase_and_streams: BTreeMap<String, Aggregate>,
    pub failed_samples: usize,
    pub errors: Vec<String>,
}
pub struct Sampler {
    phase: Arc<Mutex<(String, usize)>>,
    data: Arc<Mutex<Snapshot>>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}
impl Drop for Sampler {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}
impl Sampler {
    pub fn start(args: &Args, output: &Path) -> Result<Self> {
        let phase = Arc::new(Mutex::new(("startup".to_owned(), 0)));
        let data = Arc::new(Mutex::new(Snapshot {
            enabled: !args.no_gpu_sampling,
            ..Snapshot::default()
        }));
        let mut csv = File::create(output.join("gpu.csv"))?;
        writeln!(csv, "unix_ms,phase,streams,gpu_name,utilization_pct,memory_used_mib,memory_total_mib,power_w")?;
        let mut sampler = Self {
            phase,
            data,
            stop: None,
            task: None,
        };
        if args.no_gpu_sampling {
            return Ok(sampler);
        }
        let (tx, mut stop) = oneshot::channel();
        let phase = sampler.phase.clone();
        let data = sampler.data.clone();
        let index = args.gpu_index.to_string();
        let interval = Duration::from_secs_f64(args.gpu_sample_interval);
        sampler.stop = Some(tx);
        sampler.task = Some(tokio::spawn(async move {
            loop {
                let tag = phase.lock().unwrap().clone();
                let query = async {
                    let output = timeout(Duration::from_secs(5), Command::new("nvidia-smi")
                        .args(["--id", &index, "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"])
                        .kill_on_drop(true).output()).await.map_err(|_| "nvidia-smi timeout".to_owned())?
                        .map_err(|e| e.to_string())?;
                    if !output.status.success() {
                        return Err(format!(
                            "nvidia-smi: {}",
                            String::from_utf8_lossy(&output.stderr).trim()
                        ));
                    }
                    parse(&String::from_utf8_lossy(&output.stdout))
                };
                let result = tokio::select! { _ = &mut stop => break, result = query => result };
                match result {
                    Ok(sample) => {
                        let unix = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis();
                        let name = sample.name.replace('"', "\"\"");
                        let power = sample.power.map(|p| p.to_string()).unwrap_or_default();
                        let written = writeln!(
                            csv,
                            "{unix},{},{},\"{name}\",{},{},{},{power}",
                            tag.0, tag.1, sample.util, sample.memory, sample.total
                        )
                        .and_then(|_| csv.flush());
                        let mut data = data.lock().unwrap();
                        if let Err(err) = written {
                            data.errors.push(format!("GPU CSV: {err}"));
                            break;
                        }
                        data.device_name = Some(sample.name);
                        data.overall.add(sample.util, sample.memory, sample.power);
                        data.by_phase_and_streams
                            .entry(format!("{}:{}", tag.0, tag.1))
                            .or_default()
                            .add(sample.util, sample.memory, sample.power);
                    }
                    Err(err) => {
                        let mut data = data.lock().unwrap();
                        data.failed_samples += 1;
                        if !data.errors.contains(&err) {
                            data.errors.push(err);
                        }
                    }
                }
                tokio::select! { _ = &mut stop => break, _ = sleep(interval) => {} }
            }
        }));
        Ok(sampler)
    }
    pub fn set_phase(&self, name: &str, n: usize) {
        *self.phase.lock().unwrap() = (name.into(), n);
    }
    pub async fn finish(mut self) -> Snapshot {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.task.take() {
            if let Err(err) = task.await {
                self.data
                    .lock()
                    .unwrap()
                    .errors
                    .push(format!("GPU sampler task: {err}"));
            }
        }
        self.data.lock().unwrap().clone()
    }
}
struct Sample {
    name: String,
    util: f64,
    memory: f64,
    total: f64,
    power: Option<f64>,
}
fn parse(raw: &str) -> std::result::Result<Sample, String> {
    let fields: Vec<_> = raw.trim().split(',').map(str::trim).collect();
    if fields.len() != 5 {
        return Err(format!("unexpected nvidia-smi CSV: {raw}"));
    }
    let number = |i: usize| fields[i].parse::<f64>().ok().filter(|n| n.is_finite());
    Ok(Sample {
        name: fields[0].into(),
        util: number(1).ok_or("GPU utilization unavailable")?,
        memory: number(2).ok_or("GPU memory unavailable")?,
        total: number(3).ok_or("GPU total memory unavailable")?,
        power: number(4),
    })
}
#[cfg(test)]
mod tests {
    #[test]
    fn unavailable_telemetry_is_not_reported_as_zero() {
        assert!(super::parse("GPU, [N/A], 1, 24000, 50").is_err());
        assert!(super::parse("GPU, 0, 1, 24000, [N/A]")
            .unwrap()
            .power
            .is_none());
    }
}
