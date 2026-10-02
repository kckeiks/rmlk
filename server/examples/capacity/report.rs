use crate::Summary;
use std::fmt::Write;

pub fn markdown(summary: &Summary<'_>) -> String {
    let mut out = format!("# rmlk continuous capacity report\n\nStatus: **{}**. Ramp safe N: **{}**. Sustained safe N: **{}**.\n\n",
        summary.status, summary.ramp_safe_n, summary.safe_n.map(|n| n.to_string()).unwrap_or_else(|| "unverified".into()));
    if summary.reached_ceiling {
        out.push_str("The ramp reached the configured ceiling; this is a lower bound, not a measured saturation point.\n\n");
    }
    if summary.safe_n.is_none() {
        out.push_str(
            "No passing soak was recorded. Do not infer a safe lower level by subtracting one.\n\n",
        );
    }
    let _ = writeln!(out, "Audio: mono PCM16LE, 16 kHz, {} ms frames, {} ms inference chunks. Hold: {} s/level; soak: {} s.\n",
        summary.config.frame_ms, summary.config.server_chunk_ms, summary.config.hold_seconds, summary.config.soak_seconds);
    let _ = writeln!(out, "Gates: realtime ratio >= {}, E2E p99 <= {} ms, late-frame ratio <= {}. Errors and missing acknowledgments fail.\n",
        summary.config.min_realtime_ratio, summary.config.max_p99_client_e2e_ms, summary.config.max_late_frame_ratio);
    let _ = writeln!(out, "Operator notes: {}\n", summary.config.notes);
    out.push_str("| Phase | N | RT ratio | Late frames | E2E p50 / p95 / p99 (ms) | Unacked | Safe |\n|---|---:|---:|---:|---|---:|---|\n");
    for level in summary.levels {
        let value = |v: Option<f64>| {
            v.map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "unavailable".into())
        };
        let _ = writeln!(
            out,
            "| {} | {} | {:.4} | {:.3}% | {} / {} / {} | {} | {} |",
            level.phase,
            level.streams,
            level.realtime_ratio,
            level.late_frame_ratio * 100.,
            value(level.client_e2e.p50_ms),
            value(level.client_e2e.p95_ms),
            value(level.client_e2e.p99_ms),
            level.unacked_chunks,
            level.safe
        );
    }
    for level in summary.levels.iter().filter(|l| !l.safe) {
        let _ = writeln!(
            out,
            "\n{} N={}: {}\n",
            level.phase,
            level.streams,
            level.unsafe_reasons.join("; ")
        );
    }
    out.push_str("\nGPU telemetry is device-wide, including desktop and other processes. The mean is over periodic samples, not a kernel trace.\n\n");
    let _ = writeln!(
        out,
        "GPU enabled: {}; valid samples: {}; failed samples: {}.\n",
        summary.gpu.enabled, summary.gpu.overall.samples, summary.gpu.failed_samples
    );
    if summary.gpu.overall.samples > 0 {
        let g = &summary.gpu.overall;
        let _ = writeln!(
            out,
            "Overall GPU utilization: mean {:.2}%, peak {:.2}%; peak memory {:.0} MiB.\n",
            g.mean_utilization_pct.unwrap(),
            g.max_utilization_pct.unwrap(),
            g.max_memory_used_mib.unwrap()
        );
        out.push_str("| Phase:N | Samples | Mean GPU % | Peak GPU % | Peak memory MiB |\n|---|---:|---:|---:|---:|\n");
        for (tag, g) in &summary.gpu.by_phase_and_streams {
            let _ = writeln!(
                out,
                "| {tag} | {} | {:.2} | {:.2} | {:.0} |",
                g.samples,
                g.mean_utilization_pct.unwrap(),
                g.max_utilization_pct.unwrap(),
                g.max_memory_used_mib.unwrap()
            );
        }
    } else {
        out.push_str("GPU utilization unavailable.\n");
    }
    for error in &summary.gpu.errors {
        let _ = writeln!(out, "\nGPU telemetry warning: {error}\n");
    }
    out.push_str("\nE2E spans full-chunk enqueue through AudioProcessed and includes client backlog, transport, server queueing, and available inference work. It is not audio-to-word latency; lookahead can retain audio. Server queue/inference timings are unavailable through the rmlk protocol.\n\nClips loop continuously without zero padding. Partial counts do not establish transcription accuracy. Synthetic capacity runs must be followed by a matched real-speech workload. Model weights, quantization, context, native batching, and server settings must be recorded and aligned where possible.\n\nRaw files: `config.json`, `levels.jsonl`, `summary.json`, `gpu.csv`.\n");
    out
}
