//! What a backend reports about itself outside any one request: the
//! capacities it resolved at startup and its running counters.

use crate::outcome::Tally;

/// A size in bytes.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ByteSize(pub u64);

/// The capacities a backend resolved when it opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capacity
{
    /// Main KV capacity in tokens.
    pub kv_tokens: Tally,
    /// The backend's name for its KV storage.
    pub kv_storage: String,
    /// How the Main KV capacity was sized.
    pub kv_sizing: KvSizing,
    /// KV page groups in use.
    pub pages: Tally,
    /// KV page groups at most.
    pub max_pages: Tally,
    /// The runtime reservation.
    pub runtime: ByteSize,
    /// Device memory free after startup.
    pub free: ByteSize,
    /// The context cache.
    pub cache: ContextCache,
}

/// How the Main KV capacity was sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvSizing
{
    /// As configured.
    Explicit,
    /// From device memory.
    Automatic,
}

/// The context cache's resolved capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextCache
{
    /// Disabled: only root prefixes.
    RootOnly,
    /// Enabled with these capacities.
    Enabled
    {
        /// Active lanes.
        lanes: Tally,
        /// Device checkpoint slots beyond the lanes.
        device_states: Tally,
        /// Host checkpoint slots.
        host_states: Tally,
        /// Host KV capacity.
        host_kv: ByteSize,
        /// Private continuations.
        private: Tally,
        /// Shared prefixes.
        shared: Tally,
        /// Long anchors per continuation.
        anchors: Tally,
    },
}

/// One snapshot of a backend's counters: cumulative totals, which only grow,
/// and current gauges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RuntimeCounters
{
    /// Prompt tokens prefill evaluated, reused prefixes excluded.
    pub prefill_tokens: Tally,
    /// Tokens decode rounds committed.
    pub decode_tokens: Tally,
    /// Decode batches run.
    pub decode_rounds: Tally,
    /// Rows over every decode batch run.
    pub decode_rows: Tally,
    /// Requests in each state now.
    pub requests: RequestGauges,
    /// Host time spent active, cumulative.
    pub host_active: core::time::Duration,
    /// A digest of every other counter and gauge the backend keeps; equal
    /// digests mean nothing else moved.
    pub digest: CounterDigest,
}

/// Requests in each state, at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RequestGauges
{
    /// Running, prefilling or decoding.
    pub running: Tally,
    /// Running and prefilling.
    pub prefilling: Tally,
    /// Running and ready to decode.
    pub decode_ready: Tally,
    /// Waiting for admission.
    pub waiting: Tally,
    /// Materializing a cached prefix.
    pub materializing: Tally,
    /// Waiting on a capture.
    pub capture_pending: Tally,
    /// Finished, awaiting their terminal record.
    pub terminal_pending: Tally,
}

/// A digest over counters not otherwise reported.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CounterDigest(pub u64);
