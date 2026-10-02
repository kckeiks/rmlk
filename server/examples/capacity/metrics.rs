use crate::Args;
use serde::Serialize;

#[derive(Default, Serialize, Clone)]
pub struct Stream {
    pub clip: String,
    pub frames_sent: usize,
    pub late_frames: usize,
    pub max_lateness_ms: f64,
    pub chunks_queued: usize,
    pub chunks_acked: usize,
    pub peak_pending_chunks: usize,
    pub tail_samples: usize,
    pub partials: usize,
    pub sender_realtime_ratio: f64,
    pub realtime_ratio: f64,
    pub errors: Vec<String>,
    #[serde(skip)]
    pub latencies: Vec<f64>,
}

#[derive(Serialize)]
pub struct Latency {
    pub count: usize,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub mean_ms: Option<f64>,
    pub max_ms: Option<f64>,
}
impl Latency {
    pub fn new(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
        Self {
            count: values.len(),
            p50_ms: percentile(&values, 0.5),
            p95_ms: percentile(&values, 0.95),
            p99_ms: percentile(&values, 0.99),
            mean_ms: mean,
            max_ms: values.last().copied(),
        }
    }
}
pub fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = (sorted.len() - 1) as f64 * p;
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo as f64))
}

#[derive(Serialize)]
pub struct Level {
    pub phase: String,
    pub streams: usize,
    pub requested_seconds: f64,
    pub wall_seconds: f64,
    pub frames_sent: usize,
    pub late_frames: usize,
    pub late_frame_ratio: f64,
    pub realtime_ratio: f64,
    pub client_e2e: Latency,
    pub unacked_chunks: usize,
    pub dropped_streams: usize,
    pub safe: bool,
    pub unsafe_reasons: Vec<String>,
    pub per_stream: Vec<Stream>,
}
impl Level {
    pub fn new(
        args: &Args,
        phase: &str,
        n: usize,
        seconds: f64,
        wall: f64,
        streams: Vec<Stream>,
    ) -> Self {
        let frames = streams.iter().map(|s| s.frames_sent).sum::<usize>();
        let late = streams.iter().map(|s| s.late_frames).sum::<usize>();
        let active: Vec<_> = streams.iter().filter(|s| s.frames_sent > 0).collect();
        let rt = if active.is_empty() {
            0.0
        } else {
            active.iter().map(|s| s.realtime_ratio).sum::<f64>() / active.len() as f64
        };
        let client_e2e = Latency::new(
            streams
                .iter()
                .flat_map(|s| s.latencies.iter().copied())
                .collect(),
        );
        let late_ratio = if frames == 0 {
            1.0
        } else {
            late as f64 / frames as f64
        };
        let unacked = streams
            .iter()
            .map(|s| s.chunks_queued.saturating_sub(s.chunks_acked))
            .sum();
        let dropped = streams.iter().filter(|s| !s.errors.is_empty()).count();
        let mut reasons = Vec::new();
        if rt < args.min_realtime_ratio {
            reasons.push(format!(
                "realtime_ratio {rt:.4} < {}",
                args.min_realtime_ratio
            ));
        }
        if late_ratio > args.max_late_frame_ratio {
            reasons.push(format!(
                "late_frame_ratio {late_ratio:.4} > {}",
                args.max_late_frame_ratio
            ));
        }
        if client_e2e
            .p99_ms
            .is_none_or(|p| p > args.max_p99_client_e2e_ms)
        {
            reasons.push("client E2E p99 missing or above gate".into());
        }
        if dropped > 0 {
            reasons.push(format!("{dropped} dropped/failed streams"));
        }
        if unacked > 0 {
            reasons.push(format!("{unacked} queued chunks unacknowledged"));
        }
        if streams.len() != n || streams.iter().any(|s| s.chunks_acked == 0) {
            reasons.push("incomplete stream coverage".into());
        }
        Self {
            phase: phase.into(),
            streams: n,
            requested_seconds: seconds,
            wall_seconds: wall,
            frames_sent: frames,
            late_frames: late,
            late_frame_ratio: late_ratio,
            realtime_ratio: rt,
            client_e2e,
            unacked_chunks: unacked,
            dropped_streams: dropped,
            safe: reasons.is_empty(),
            unsafe_reasons: reasons,
            per_stream: streams,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn percentile_matches_dirigo_linear_interpolation() {
        assert_eq!(percentile(&[], 0.99), None);
        assert_eq!(percentile(&[10., 20.], 0.99), Some(19.9));
    }
    #[test]
    fn failed_or_unacknowledged_streams_cannot_look_safe() {
        let args = Args::parse_from(["test"]);
        let good = Stream {
            frames_sent: 50,
            chunks_queued: 1,
            chunks_acked: 1,
            realtime_ratio: 1.,
            latencies: vec![10.],
            ..Stream::default()
        };
        assert!(Level::new(&args, "ramp", 1, 1., 1., vec![good.clone()]).safe);
        let bad = Stream {
            chunks_queued: 2,
            ..good.clone()
        };
        assert!(!Level::new(&args, "ramp", 1, 1., 1., vec![bad]).safe);
        let failed = Stream {
            errors: vec!["BUSY".into()],
            ..Stream::default()
        };
        assert!(!Level::new(&args, "ramp", 2, 1., 1., vec![good, failed]).safe);
    }
}
