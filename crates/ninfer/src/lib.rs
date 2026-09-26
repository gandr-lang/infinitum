//! infinitum's ninfer backend.
//!
//! [`Ninfer`] plans a round graph: ninfer runs the DFlash2 round as one fused
//! Engine round, so the canonical DFlash2 graph plans at its width and every
//! other graph is refused with the reason. With the `engine` feature,
//! `Session` opens ninfer's C++ Engine through a `cxx` bridge and runs the
//! plan, with infinitum's round preview making each round's output decision,
//! and `ChatEngine` serves chat requests on it as an
//! `infinitum_chat::ChatBackend`.

#[cfg(feature = "engine")]
mod bridge;
#[cfg(feature = "engine")]
mod chat;
mod options;
mod plan;
#[cfg(feature = "engine")]
mod session;
mod text;

#[cfg(feature = "engine")]
pub use chat::ChatEngine;
pub use options::ChatTemplate;
pub use options::Concurrency;
pub use options::ConcurrencyOutOfRange;
pub use options::ContextLimit;
pub use options::CudaGraph;
pub use options::DeviceOrdinal;
pub use options::EngineOptions;
pub use options::KvBelowContext;
pub use options::KvCapacity;
pub use options::KvStorage;
pub use options::MalformedKvCapacity;
pub use options::MalformedPrefillChunk;
pub use options::PendingTimeout;
pub use options::PrefillChunk;
pub use options::UnknownCudaGraph;
pub use options::UnknownKvStorage;
pub use plan::DFlash2Plan;
pub use plan::Ninfer;
#[cfg(feature = "engine")]
pub use session::EngineFailure;
#[cfg(feature = "engine")]
pub use session::Finish;
#[cfg(feature = "engine")]
pub use session::Generation;
#[cfg(feature = "engine")]
pub use session::Operation;
#[cfg(feature = "engine")]
pub use session::RoundTally;
#[cfg(feature = "engine")]
pub use session::Session;
#[cfg(feature = "engine")]
pub use session::Speculation;
#[cfg(feature = "engine")]
pub use session::ThrownKind;
pub use text::RawText;
pub use text::RenderedBytes;
