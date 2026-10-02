use crate::Args;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, sync::Arc};

#[derive(Clone, Serialize)]
pub struct ClipInfo {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: usize,
    pub samples: usize,
    pub sample_rate_hz: u32,
    pub pcm_conversion: &'static str,
}

pub struct Clip {
    pub info: ClipInfo,
    pub pcm: Arc<Vec<i16>>,
}

#[derive(Deserialize)]
struct Manifest {
    clips: Vec<Pin>,
}

#[derive(Deserialize)]
struct Pin {
    wav: String,
    sha256: String,
    bytes: usize,
}

pub fn load(args: &Args) -> Result<Vec<Clip>> {
    let dir = args
        .data_dir
        .clone()
        .or_else(|| std::env::var_os("RMLK_TESTDATA_CACHE").map(PathBuf::from))
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.cache/rmlk/testdata")
        });
    let manifest: Manifest =
        serde_json::from_str(include_str!("../../tests/manifests/capacity.json"))?;
    let mut paths: Vec<_> = if args.data_dir.is_some() {
        fs::read_dir(&dir)
            .with_context(|| format!("read audio directory {}", dir.display()))?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|p| {
                p.is_file() && matches!(p.extension().and_then(|s| s.to_str()), Some("wav" | "pcm"))
            })
            .collect()
    } else {
        manifest.clips.iter().map(|p| dir.join(&p.wav)).collect()
    };
    // Dirigo sorts by filename before assigning audios[i % len(audios)].
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    if paths.is_empty() {
        bail!("no .wav or .pcm clips in {}", dir.display());
    }
    paths
        .into_iter()
        .map(|path| {
            let data = fs::read(&path).with_context(|| {
                format!(
                    "read {}; generate the pack with server/scripts/gen_capacity_load.py",
                    path.display()
                )
            })?;
            let hash = format!("{:x}", Sha256::digest(&data));
            let name = path
                .file_name()
                .context("audio has no filename")?
                .to_string_lossy()
                .into_owned();
            if args.data_dir.is_none() {
                let pin = manifest
                    .clips
                    .iter()
                    .find(|p| p.wav == name)
                    .context("missing manifest pin")?;
                if hash != pin.sha256 || data.len() != pin.bytes {
                    bail!("capacity checksum/size mismatch: {}", path.display());
                }
            }
            let pcm = if path.extension().and_then(|s| s.to_str()) == Some("pcm") {
                if data.len() % 2 != 0 {
                    bail!("PCM16LE file has an odd byte length: {}", path.display());
                }
                data.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect::<Vec<_>>()
            } else {
                let mut reader = hound::WavReader::new(std::io::Cursor::new(&data))?;
                let spec = reader.spec();
                if spec.sample_rate != 16000
                    || spec.channels != 1
                    || spec.bits_per_sample != 16
                    || spec.sample_format != hound::SampleFormat::Int
                {
                    bail!("require mono PCM16 WAV at 16000 Hz: {}", path.display());
                }
                reader
                    .samples::<i16>()
                    .collect::<std::result::Result<Vec<_>, _>>()?
            };
            if pcm.is_empty() {
                bail!("empty audio: {}", path.display());
            }
            // Match Dirigo's WAV/PCM -> float32 / 32768 -> wire int16 * 32767 conversion.
            let pcm = pcm
                .into_iter()
                .map(|s| ((f32::from(s) / 32768.0) * 32767.0) as i16)
                .collect::<Vec<_>>();
            Ok(Clip {
                info: ClipInfo {
                    name,
                    path,
                    sha256: hash,
                    bytes: data.len(),
                    samples: pcm.len(),
                    sample_rate_hz: 16000,
                    pcm_conversion: "int16 -> float32 / 32768 -> int16 * 32767 (truncation)",
                },
                pcm: Arc::new(pcm),
            })
        })
        .collect()
}

pub fn append_loop(pcm: &[i16], cursor: &mut usize, count: usize, out: &mut Vec<i16>) {
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(pcm.len() - *cursor);
        out.extend_from_slice(&pcm[*cursor..*cursor + n]);
        *cursor = (*cursor + n) % pcm.len();
        remaining -= n;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn wraps_without_padding_or_resetting_at_chunk_boundaries() {
        let mut cursor = 0;
        let mut out = Vec::new();
        super::append_loop(&[1, 2, 3], &mut cursor, 5, &mut out);
        super::append_loop(&[1, 2, 3], &mut cursor, 5, &mut out);
        assert_eq!(out, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]);
    }
}
