//! Wire protocol frames between clients and this server.

use thiserror::Error;

/// Frames sent by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientFrame {
    Open,
    Audio,
    Cancel,
    Finalize,
}

/// Frames sent by the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerFrame {
    OpenAck,
    Partial,
    Final,
    Error,
}

/// Codec and framing failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("protocol error")]
    Unspecified,
}

#[cfg(test)]
mod tests {
    use super::{ClientFrame, ProtocolError, ServerFrame};

    #[test]
    fn frame_and_error_types_exist() {
        let _ = ClientFrame::Open;
        let _ = ClientFrame::Audio;
        let _ = ClientFrame::Cancel;
        let _ = ClientFrame::Finalize;

        let _ = ServerFrame::OpenAck;
        let _ = ServerFrame::Partial;
        let _ = ServerFrame::Final;
        let _ = ServerFrame::Error;

        let err = ProtocolError::Unspecified;
        assert_eq!(err.to_string(), "protocol error");
    }
}
