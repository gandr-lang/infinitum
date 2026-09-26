//! The OpenAI-compatible HTTP surface of the infinitum engine.
//!
//! [`serve`] answers `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`
//! and `POST /v1/chat/completions` over any [`infinitum_chat::ChatBackend`].
//! Request bodies are validated and translated into the protocol-neutral
//! chat request as ninfer's server translates them, and responses, streamed
//! or not, carry ninfer's fields, so a client moves between the two servers
//! unchanged.

extern crate alloc;

mod error;
mod oplog;
mod parse;
mod pretty;
mod render;
mod server;

pub use crate::error::ApiError;
pub use crate::error::Code;
pub use crate::error::Param;
pub use crate::oplog::log_capacity;
pub use crate::parse::Defaults;
pub use crate::server::Access;
pub use crate::server::ApiKey;
pub use crate::server::ModelId;
pub use crate::server::RequestBytes;
pub use crate::server::ServeConfig;
pub use crate::server::StatsInterval;
pub use crate::server::serve;
pub use crate::server::warm_up;
