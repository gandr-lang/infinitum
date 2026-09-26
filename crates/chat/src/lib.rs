//! A chat request and its outcome, as infinitum hands them to a serving
//! backend.
//!
//! An HTTP surface translates its wire format into a [`ChatRequest`]; a
//! [`ChatBackend`] runs it, publishing admission and text deltas to
//! [`ChatEvents`] as they commit, and returns a [`ChatOutcome`] or a classified
//! [`ChatFailure`]. Nothing here names a protocol or a backend.

extern crate alloc;

mod backend;
mod outcome;
mod request;
mod runtime;

pub use backend::CancelToken;
pub use backend::Cancellation;
pub use backend::ChatBackend;
pub use backend::ChatEvents;
pub use backend::ChatFailure;
pub use backend::DeltaText;
pub use backend::FailureKind;
pub use backend::ModelName;
pub use backend::Submission;
pub use backend::Thinking;
pub use outcome::Admission;
pub use outcome::Channel;
pub use outcome::ChatOutcome;
pub use outcome::Finish;
pub use outcome::GeneratedToolCall;
pub use outcome::ReusePath;
pub use outcome::Tally;
pub use outcome::Telemetry;
pub use outcome::ThinkingSpend;
pub use request::Automatic;
pub use request::CacheBoundary;
pub use request::CacheMarker;
pub use request::ChatRequest;
pub use request::Count;
pub use request::Delivery;
pub use request::Effort;
pub use request::Generation;
pub use request::InstructionBytes;
pub use request::Marked;
pub use request::Message;
pub use request::PrefixReuse;
pub use request::Prompt;
pub use request::PromptCache;
pub use request::Role;
pub use request::Sampling;
pub use request::Seed;
pub use request::Setting;
pub use request::SpecialTokens;
pub use request::StopScope;
pub use request::StructuralPrefixes;
pub use request::Switch;
pub use request::ThinkingBudget;
pub use request::ToolCall;
pub use runtime::ByteSize;
pub use runtime::Capacity;
pub use runtime::ContextCache;
pub use runtime::CounterDigest;
pub use runtime::KvSizing;
pub use runtime::RequestGauges;
pub use runtime::RuntimeCounters;
