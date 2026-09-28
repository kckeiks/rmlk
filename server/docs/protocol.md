# rmlk-server wire protocol

Transport: WebSocket. One connection carries one streaming session.

The Rust codec (`rmlk_server::protocol`) encodes and decodes these frames as
plain bytes. It does not depend on Axum, tungstenite, or other HTTP/WS types.
The connection layer maps WebSocket messages to those bytes.

## Message kinds

| Kind | WebSocket data | Role |
|------|----------------|------|
| Control / text frames | Binary WebSocket message with a typed header (below) | Session lifecycle and transcripts |
| Audio | Same framing; payload is PCM samples | Streaming audio |

All application messages use one binary frame layout. There is no separate
JSON control channel.

## Binary layout

Every message:

| Field | Size | Notes |
|-------|------|--------|
| `type` | 1 byte | Frame type tag |
| `payload_len` | 4 bytes | Little-endian `u32` byte length of `payload` |
| `payload` | `payload_len` bytes | Type-specific |

Maximum `payload_len`: 16 MiB. Larger values are a protocol error.

### Client → server (`type`)

| Tag | Name | Payload |
|-----|------|---------|
| `1` | `Open` | Empty. Starts a session. Audio is mono PCM16 little-endian at 16 kHz. |
| `2` | `Audio` | Raw PCM16 little-endian mono samples at 16 kHz (no WAV header). Length must be even. |
| `3` | `Cancel` | Empty. Drops the session without requiring a final transcript. |
| `4` | `Finalize` | Empty. End of audio; server should emit `Final` and close the session. |

### Server → client (`type`)

| Tag | Name | Payload |
|-----|------|---------|
| `1` | `OpenAck` | 8 bytes: session id as little-endian `u64`. |
| `2` | `Partial` | UTF-8 transcript text (may be incomplete). |
| `3` | `Final` | UTF-8 transcript text for the completed session (or utterance). |
| `4` | `Error` | 2 bytes little-endian `u16` error code, then UTF-8 message. |

Unknown `type` values are a protocol error.

## Session flow

1. Client connects and sends `Open`.
2. Server replies with `OpenAck`.
3. Client sends zero or more `Audio` frames.
4. Server may send `Partial` frames at any time after `OpenAck`.
5. Client sends `Finalize`, or `Cancel`, or closes the WebSocket.
6. On `Finalize`, server sends `Final` (and may have sent `Partial`s first), then the session ends.
7. On `Cancel` or socket close, the server drops session state; no further frames are required.

Audio before `Open`, or after `Finalize` / `Cancel`, is a protocol error.

## Error codes

| Code | Meaning |
|------|---------|
| `1` | Malformed frame (truncated, bad length, odd PCM byte length, invalid UTF-8). |
| `2` | Unexpected frame for the current session state. |
| `3` | Server busy / backpressure (session or queue full). |
| `4` | Internal server error. |

## Rust types

| Wire name | Rust enum variant |
|-----------|-------------------|
| `Open` | `ClientFrame::Open` |
| `Audio` | `ClientFrame::Audio { pcm16 }` — `Vec<i16>`, wire bytes are LE PCM16 @ 16 kHz mono |
| `Cancel` | `ClientFrame::Cancel` |
| `Finalize` | `ClientFrame::Finalize` |
| `OpenAck` | `ServerFrame::OpenAck { session_id }` |
| `Partial` | `ServerFrame::Partial` |
| `Final` | `ServerFrame::Final` |
| `Error` | `ServerFrame::Error` |
