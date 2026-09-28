//! Wire protocol frames between clients and this server.
//!
//! Binary layout is documented in `server/docs/protocol.md`.

use thiserror::Error;

const MAX_PAYLOAD_LEN: u32 = 16 << 20; // 16 MiB

const CLIENT_OPEN: u8 = 1;
const CLIENT_AUDIO: u8 = 2;
const SERVER_OPEN_ACK: u8 = 1;

/// Frames sent by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientFrame {
    Open,
    /// Mono PCM16 little-endian samples at 16 kHz (no WAV header).
    Audio { pcm16: Vec<i16> },
    Cancel,
    Finalize,
}

/// Frames sent by the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerFrame {
    OpenAck { session_id: u64 },
    Partial,
    Final,
    Error,
}

/// Codec and framing failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("truncated frame")]
    Truncated,
    #[error("payload longer than {MAX_PAYLOAD_LEN} bytes")]
    PayloadTooLarge,
    #[error("invalid payload for frame type")]
    InvalidPayload,
    #[error("unknown frame type {0}")]
    UnknownType(u8),
    #[error("unsupported frame type")]
    Unsupported,
}

impl ClientFrame {
    /// Encode this frame to the binary wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        match self {
            Self::Open => Ok(encode_message(CLIENT_OPEN, &[])),
            Self::Audio { pcm16 } => {
                let mut payload = Vec::with_capacity(pcm16.len() * 2);
                for sample in pcm16 {
                    payload.extend_from_slice(&sample.to_le_bytes());
                }
                Ok(encode_message(CLIENT_AUDIO, &payload))
            }
            Self::Cancel | Self::Finalize => Err(ProtocolError::Unsupported),
        }
    }

    /// Decode a client frame from the binary wire format.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        let (tag, payload) = decode_message(bytes)?;
        match tag {
            CLIENT_OPEN => {
                if !payload.is_empty() {
                    return Err(ProtocolError::InvalidPayload);
                }
                Ok(Self::Open)
            }
            CLIENT_AUDIO => Ok(Self::Audio {
                pcm16: decode_pcm16_le(payload)?,
            }),
            3 | 4 => Err(ProtocolError::Unsupported),
            other => Err(ProtocolError::UnknownType(other)),
        }
    }
}

impl ServerFrame {
    /// Encode this frame to the binary wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        match self {
            Self::OpenAck { session_id } => {
                Ok(encode_message(SERVER_OPEN_ACK, &session_id.to_le_bytes()))
            }
            Self::Partial | Self::Final | Self::Error => Err(ProtocolError::Unsupported),
        }
    }

    /// Decode a server frame from the binary wire format.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        let (tag, payload) = decode_message(bytes)?;
        match tag {
            SERVER_OPEN_ACK => {
                let session_id = read_u64_le(payload)?;
                Ok(Self::OpenAck { session_id })
            }
            2 | 3 | 4 => Err(ProtocolError::Unsupported),
            other => Err(ProtocolError::UnknownType(other)),
        }
    }
}

fn encode_message(tag: u8, payload: &[u8]) -> Vec<u8> {
    let len = payload.len() as u32;
    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    out.push(tag);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn decode_message(bytes: &[u8]) -> Result<(u8, &[u8]), ProtocolError> {
    if bytes.len() < 5 {
        return Err(ProtocolError::Truncated);
    }
    let tag = bytes[0];
    let payload_len = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    if payload_len > MAX_PAYLOAD_LEN {
        return Err(ProtocolError::PayloadTooLarge);
    }
    let payload_len = payload_len as usize;
    let rest = &bytes[5..];
    if rest.len() < payload_len {
        return Err(ProtocolError::Truncated);
    }
    if rest.len() != payload_len {
        return Err(ProtocolError::InvalidPayload);
    }
    Ok((tag, rest))
}

fn read_u64_le(payload: &[u8]) -> Result<u64, ProtocolError> {
    let bytes: [u8; 8] = payload.try_into().map_err(|_| ProtocolError::InvalidPayload)?;
    Ok(u64::from_le_bytes(bytes))
}

fn decode_pcm16_le(payload: &[u8]) -> Result<Vec<i16>, ProtocolError> {
    if payload.len() % 2 != 0 {
        return Err(ProtocolError::InvalidPayload);
    }
    let mut samples = Vec::with_capacity(payload.len() / 2);
    for chunk in payload.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::{ClientFrame, ProtocolError, ServerFrame};

    #[test]
    fn frame_and_error_types_exist() {
        let _ = ClientFrame::Open;
        let _ = ClientFrame::Audio { pcm16: vec![] };
        let _ = ClientFrame::Cancel;
        let _ = ClientFrame::Finalize;

        let _ = ServerFrame::OpenAck { session_id: 0 };
        let _ = ServerFrame::Partial;
        let _ = ServerFrame::Final;
        let _ = ServerFrame::Error;

        let err = ProtocolError::Truncated;
        assert_eq!(err.to_string(), "truncated frame");
    }

    #[test]
    fn open_round_trip() {
        let frame = ClientFrame::Open;
        let bytes = frame.encode().unwrap();
        assert_eq!(bytes, [1, 0, 0, 0, 0]);
        assert_eq!(ClientFrame::decode(&bytes).unwrap(), ClientFrame::Open);
    }

    #[test]
    fn open_ack_round_trip() {
        let frame = ServerFrame::OpenAck {
            session_id: 0x0123_4567_89ab_cdef,
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap(),
            ServerFrame::OpenAck {
                session_id: 0x0123_4567_89ab_cdef,
            }
        );
    }

    #[test]
    fn open_rejects_non_empty_payload() {
        let bytes = [1, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn open_ack_rejects_wrong_payload_len() {
        let bytes = [1, 4, 0, 0, 0, 1, 2, 3, 4];
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn audio_round_trip() {
        let frame = ClientFrame::Audio {
            pcm16: vec![0, -1, 1, i16::MAX, i16::MIN],
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(ClientFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn audio_empty_payload_round_trip() {
        let frame = ClientFrame::Audio { pcm16: vec![] };
        let bytes = frame.encode().unwrap();
        assert_eq!(bytes, [2, 0, 0, 0, 0]);
        assert_eq!(ClientFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn audio_rejects_odd_byte_length() {
        let bytes = [2, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }
}
