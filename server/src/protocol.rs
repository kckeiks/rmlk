//! Wire protocol frames between clients and this server.
//!
//! Binary layout is documented in `server/docs/protocol.md`.

use thiserror::Error;

const MAX_PAYLOAD_LEN: u32 = 16 << 20; // 16 MiB

const CLIENT_OPEN: u8 = 1;
const CLIENT_AUDIO: u8 = 2;
const CLIENT_CANCEL: u8 = 3;
const CLIENT_FINALIZE: u8 = 4;
const SERVER_OPEN_ACK: u8 = 1;
const SERVER_PARTIAL: u8 = 2;
const SERVER_FINAL: u8 = 3;
const SERVER_ERROR: u8 = 4;

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
    Partial { text: String },
    Final { text: String },
    Error { code: u16, message: String },
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
}

impl ClientFrame {
    /// Encode this frame to the binary wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        match self {
            Self::Open => encode_message(CLIENT_OPEN, &[]),
            Self::Audio { pcm16 } => {
                let mut payload = Vec::with_capacity(pcm16.len() * 2);
                for sample in pcm16 {
                    payload.extend_from_slice(&sample.to_le_bytes());
                }
                encode_message(CLIENT_AUDIO, &payload)
            }
            Self::Cancel => encode_message(CLIENT_CANCEL, &[]),
            Self::Finalize => encode_message(CLIENT_FINALIZE, &[]),
        }
    }

    /// Decode a client frame from the binary wire format.
    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        let (tag, payload) = decode_message(bytes)?;
        match tag {
            CLIENT_OPEN => {
                require_empty_payload(payload)?;
                Ok(Self::Open)
            }
            CLIENT_AUDIO => Ok(Self::Audio {
                pcm16: decode_pcm16_le(payload)?,
            }),
            CLIENT_CANCEL => {
                require_empty_payload(payload)?;
                Ok(Self::Cancel)
            }
            CLIENT_FINALIZE => {
                require_empty_payload(payload)?;
                Ok(Self::Finalize)
            }
            other => Err(ProtocolError::UnknownType(other)),
        }
    }
}

impl ServerFrame {
    /// Encode this frame to the binary wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        match self {
            Self::OpenAck { session_id } => {
                encode_message(SERVER_OPEN_ACK, &session_id.to_le_bytes())
            }
            Self::Partial { text } => encode_message(SERVER_PARTIAL, text.as_bytes()),
            Self::Final { text } => encode_message(SERVER_FINAL, text.as_bytes()),
            Self::Error { code, message } => {
                let mut payload = Vec::with_capacity(2 + message.len());
                payload.extend_from_slice(&code.to_le_bytes());
                payload.extend_from_slice(message.as_bytes());
                encode_message(SERVER_ERROR, &payload)
            }
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
            SERVER_PARTIAL => Ok(Self::Partial {
                text: decode_utf8(payload)?,
            }),
            SERVER_FINAL => Ok(Self::Final {
                text: decode_utf8(payload)?,
            }),
            SERVER_ERROR => {
                let (code, message) = decode_error_payload(payload)?;
                Ok(Self::Error { code, message })
            }
            other => Err(ProtocolError::UnknownType(other)),
        }
    }
}

fn encode_message(tag: u8, payload: &[u8]) -> Result<Vec<u8>, ProtocolError> {
    if payload.len() > MAX_PAYLOAD_LEN as usize {
        return Err(ProtocolError::PayloadTooLarge);
    }
    let len = payload.len() as u32;
    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    out.push(tag);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
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

fn decode_utf8(payload: &[u8]) -> Result<String, ProtocolError> {
    String::from_utf8(payload.to_vec()).map_err(|_| ProtocolError::InvalidPayload)
}

fn require_empty_payload(payload: &[u8]) -> Result<(), ProtocolError> {
    if payload.is_empty() {
        Ok(())
    } else {
        Err(ProtocolError::InvalidPayload)
    }
}

fn decode_error_payload(payload: &[u8]) -> Result<(u16, String), ProtocolError> {
    if payload.len() < 2 {
        return Err(ProtocolError::InvalidPayload);
    }
    let code = u16::from_le_bytes([payload[0], payload[1]]);
    let message = decode_utf8(&payload[2..])?;
    Ok((code, message))
}

#[cfg(test)]
mod tests {
    use super::{ClientFrame, ProtocolError, ServerFrame};
    use proptest::prelude::*;

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

    #[test]
    fn partial_round_trip() {
        let frame = ServerFrame::Partial {
            text: "hello".into(),
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(ServerFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn final_round_trip() {
        let frame = ServerFrame::Final {
            text: "hello world".into(),
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(ServerFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn partial_empty_text_round_trip() {
        let frame = ServerFrame::Partial { text: String::new() };
        let bytes = frame.encode().unwrap();
        assert_eq!(bytes, [2, 0, 0, 0, 0]);
        assert_eq!(ServerFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn partial_rejects_invalid_utf8() {
        let bytes = [2, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn cancel_round_trip() {
        let bytes = ClientFrame::Cancel.encode().unwrap();
        assert_eq!(bytes, [3, 0, 0, 0, 0]);
        assert_eq!(ClientFrame::decode(&bytes).unwrap(), ClientFrame::Cancel);
    }

    #[test]
    fn finalize_round_trip() {
        let bytes = ClientFrame::Finalize.encode().unwrap();
        assert_eq!(bytes, [4, 0, 0, 0, 0]);
        assert_eq!(ClientFrame::decode(&bytes).unwrap(), ClientFrame::Finalize);
    }

    #[test]
    fn cancel_rejects_non_empty_payload() {
        let bytes = [3, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn finalize_rejects_non_empty_payload() {
        let bytes = [4, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn error_round_trip() {
        let frame = ServerFrame::Error {
            code: 1,
            message: "malformed frame".into(),
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(ServerFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn error_empty_message_round_trip() {
        let frame = ServerFrame::Error {
            code: 4,
            message: String::new(),
        };
        let bytes = frame.encode().unwrap();
        assert_eq!(bytes, [4, 2, 0, 0, 0, 4, 0]);
        assert_eq!(ServerFrame::decode(&bytes).unwrap(), frame);
    }

    #[test]
    fn error_rejects_short_payload() {
        let bytes = [4, 1, 0, 0, 0, 0xff];
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn error_rejects_invalid_utf8_message() {
        let bytes = [4, 3, 0, 0, 0, 1, 0, 0xff];
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    #[test]
    fn truncated_header() {
        assert_eq!(
            ClientFrame::decode(&[]).unwrap_err(),
            ProtocolError::Truncated
        );
        assert_eq!(
            ClientFrame::decode(&[1, 0, 0, 0]).unwrap_err(),
            ProtocolError::Truncated
        );
    }

    #[test]
    fn truncated_payload() {
        // Claims 2 payload bytes; only 1 follows.
        let bytes = [1, 2, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::Truncated
        );
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::Truncated
        );
    }

    #[test]
    fn oversized_payload_len() {
        // Length = MAX_PAYLOAD_LEN + 1; no payload bytes needed.
        let bytes = [1, 1, 0, 0, 1];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::PayloadTooLarge
        );
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::PayloadTooLarge
        );
    }

    #[test]
    fn encode_rejects_oversized_payload() {
        let payload = vec![0u8; (super::MAX_PAYLOAD_LEN as usize) + 1];
        assert_eq!(
            super::encode_message(1, &payload).unwrap_err(),
            ProtocolError::PayloadTooLarge
        );
    }

    #[test]
    fn unknown_client_tag_type() {
        let bytes = [0, 0, 0, 0, 0];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::UnknownType(0)
        );
        let bytes = [255, 0, 0, 0, 0];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::UnknownType(255)
        );
    }

    #[test]
    fn unknown_server_tag_type() {
        let bytes = [5, 0, 0, 0, 0];
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap_err(),
            ProtocolError::UnknownType(5)
        );
    }

    #[test]
    fn trailing_bytes_are_invalid_payload() {
        let bytes = [1, 0, 0, 0, 0, 0xff];
        assert_eq!(
            ClientFrame::decode(&bytes).unwrap_err(),
            ProtocolError::InvalidPayload
        );
    }

    fn arb_client_frame() -> impl Strategy<Value = ClientFrame> {
        prop_oneof![
            Just(ClientFrame::Open),
            Just(ClientFrame::Cancel),
            Just(ClientFrame::Finalize),
            prop::collection::vec(any::<i16>(), 0..2048)
                .prop_map(|pcm16| ClientFrame::Audio { pcm16 }),
        ]
    }

    fn arb_utf8(max_chars: usize) -> impl Strategy<Value = String> {
        prop::collection::vec(any::<char>(), 0..=max_chars)
            .prop_map(|chars| chars.into_iter().collect())
    }

    fn arb_server_frame() -> impl Strategy<Value = ServerFrame> {
        prop_oneof![
            any::<u64>().prop_map(|session_id| ServerFrame::OpenAck { session_id }),
            arb_utf8(256).prop_map(|text| ServerFrame::Partial { text }),
            arb_utf8(256).prop_map(|text| ServerFrame::Final { text }),
            (any::<u16>(), arb_utf8(256))
                .prop_map(|(code, message)| ServerFrame::Error { code, message }),
        ]
    }

    proptest! {
        #[test]
        fn client_frame_round_trip_prop(frame in arb_client_frame()) {
            let bytes = frame.encode().expect("encode");
            prop_assert_eq!(ClientFrame::decode(&bytes).expect("decode"), frame);
        }

        #[test]
        fn server_frame_round_trip_prop(frame in arb_server_frame()) {
            let bytes = frame.encode().expect("encode");
            prop_assert_eq!(ServerFrame::decode(&bytes).expect("decode"), frame);
        }
    }
}
