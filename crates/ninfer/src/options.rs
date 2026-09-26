//! How an Engine is opened: the artifact, the device, the context ceiling,
//! the KV cache's capacity and storage, the prefill chunk, the concurrency,
//! the context cache's state and host KV capacities, `RoPE` position scaling,
//! the pending timeout, CUDA graph capture, and the chat template.

/// The logical ceiling of one request in tokens, prompt and generation
/// together. The Engine also sizes its KV cache from it, so a small ceiling
/// keeps a one-request run's device footprint small.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLimit(core::num::NonZeroU32);

impl From<ContextLimit> for core::num::NonZeroU32
{
    /// Unwrap the limit.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(limit: ContextLimit) -> Self
    {
        return limit.0;
    }
}

impl core::str::FromStr for ContextLimit
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the limit is the parsed value, at least one.
    /// - provides: the command-line spelling of a context ceiling.
    /// - fails: with the integer parser's error on zero, a negative value,
    ///   anything above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary, the one decision the wrapper adds
    ///   to `u32` parsing.
    /// - witness: `tests::a_zero_context_is_refused`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

/// How many tokens the Engine's Main KV cache holds, shared by every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCapacity
{
    /// This many tokens, at least the context ceiling.
    Tokens(core::num::NonZeroU32),
    /// As many as device memory holds after weights, runtime and graph
    /// allowance, less ninfer's sizing headroom.
    Automatic,
}

impl core::str::FromStr for KvCapacity
{
    type Err = MalformedKvCapacity;

    /// Parse `auto` or a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `auto` is [`KvCapacity::Automatic`]; a positive `u32` is that
    ///   many tokens.
    /// - provides: the command-line spelling of a KV capacity, as ninfer's
    ///   server spells it.
    /// - fails: on zero, a negative value, anything above `u32::MAX`, or any
    ///   other text, case included.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`MalformedKvCapacity`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary and over both spellings.
    /// - witness: `tests::kv_capacity_parses_auto_and_positive_counts`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        if text == "auto" {
            return Ok(Self::Automatic);
        }
        return text
            .parse::<core::num::NonZeroU32>()
            .map(Self::Tokens)
            .map_err(|_malformed| return MalformedKvCapacity);
    }
}

/// A KV capacity that is neither `auto` nor a positive token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalformedKvCapacity;

impl core::fmt::Display for MalformedKvCapacity
{
    /// Render the failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("KV capacity is `auto` or a positive token count");
    }
}

impl core::error::Error for MalformedKvCapacity
{
}

/// A KV capacity in tokens below the context ceiling, which one request
/// could outgrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBelowContext;

impl core::fmt::Display for KvBelowContext
{
    /// Render the failure, in ninfer's words.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("--kv-capacity must be at least --max-context");
    }
}

impl core::error::Error for KvBelowContext
{
}

/// How the KV cache stores keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStorage
{
    /// `bf16`: bfloat16 keys and values.
    BFloat16,
    /// `int8`: 8-bit integers in groups of 64.
    Int8,
    /// `fp8`: FP8 E4M3 with one scale per 256-element row.
    Fp8,
    /// `nvfp4`: NVFP4 in groups of 16, as the served engine stores it.
    Nvfp4,
    /// `k8v4`: FP8 keys and NVFP4 values.
    Fp8KeyNvfp4Value,
}

impl core::str::FromStr for KvStorage
{
    type Err = UnknownKvStorage;

    /// Parse ninfer's spelling of a KV storage.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `bf16`, `int8`, `fp8`, `nvfp4` and `k8v4` are the storages
    ///   ninfer's `--kv-dtype` names by them.
    /// - provides: the command-line spelling of a KV storage.
    /// - fails: on any other text, case included.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`UnknownKvStorage`]: `text` is none of the five spellings.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over the five spellings and one refusal.
    /// - witness: `tests::kv_storage_parses_ninfers_five_spellings`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return match text {
            | "bf16" => Ok(Self::BFloat16),
            | "int8" => Ok(Self::Int8),
            | "fp8" => Ok(Self::Fp8),
            | "nvfp4" => Ok(Self::Nvfp4),
            | "k8v4" => Ok(Self::Fp8KeyNvfp4Value),
            | _ => Err(UnknownKvStorage),
        };
    }
}

/// A KV storage ninfer does not name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownKvStorage;

impl core::fmt::Display for UnknownKvStorage
{
    /// Render the failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("KV storage is `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`");
    }
}

impl core::error::Error for UnknownKvStorage
{
}

/// How many prompt tokens one prefill step processes: a positive multiple of
/// 128.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunk(core::num::NonZeroU32);

impl PrefillChunk
{
    /// ninfer's default: 1024 tokens.
    pub const DEFAULT: Self = Self(match core::num::NonZeroU32::new(1024) {
        | Some(tokens) => tokens,
        | None => core::num::NonZeroU32::MIN,
    });
}

impl From<PrefillChunk> for core::num::NonZeroU32
{
    /// Unwrap the chunk's tokens.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(chunk: PrefillChunk) -> Self
    {
        return chunk.0;
    }
}

impl core::str::FromStr for PrefillChunk
{
    type Err = MalformedPrefillChunk;

    /// Parse a positive decimal multiple of 128.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the chunk is the parsed value, a positive multiple
    ///   of 128.
    /// - provides: the command-line spelling of a prefill chunk, with ninfer's
    ///   server's range.
    /// - fails: on zero, a value that is not a multiple of 128, anything above
    ///   `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`MalformedPrefillChunk`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at zero and either side of 128.
    /// - witness: `tests::a_prefill_chunk_is_a_positive_multiple_of_128`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let tokens = text
            .parse::<core::num::NonZeroU32>()
            .map_err(|_malformed| return MalformedPrefillChunk)?;
        if tokens.get() % 128 != 0 {
            return Err(MalformedPrefillChunk);
        }
        return Ok(Self(tokens));
    }
}

/// A prefill chunk that is not a positive multiple of 128.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalformedPrefillChunk;

impl core::fmt::Display for MalformedPrefillChunk
{
    /// Render the failure, in ninfer's words.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("--prefill-chunk must be a positive multiple of 128");
    }
}

impl core::error::Error for MalformedPrefillChunk
{
}

/// How many requests the Engine runs at once, each on its own lane: one to
/// eight, ninfer's range.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Concurrency(core::num::NonZeroU32);

impl Concurrency
{
    /// One request at a time, ninfer's default.
    pub const ONE: Self = Self(core::num::NonZeroU32::MIN);
}

impl From<Concurrency> for core::num::NonZeroU32
{
    /// Unwrap the lane count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(concurrency: Concurrency) -> Self
    {
        return concurrency.0;
    }
}

impl core::str::FromStr for Concurrency
{
    type Err = ConcurrencyOutOfRange;

    /// Parse a decimal lane count from one to eight.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the concurrency is the parsed value, in `[1,8]`.
    /// - provides: the command-line spelling of a concurrency, with ninfer's
    ///   range.
    /// - fails: on zero, anything above eight, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ConcurrencyOutOfRange`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of both ends of the range.
    /// - witness: `tests::concurrency_is_one_to_eight`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let lanes = text
            .parse::<core::num::NonZeroU32>()
            .map_err(|_malformed| return ConcurrencyOutOfRange)?;
        if lanes.get() > 8 {
            return Err(ConcurrencyOutOfRange);
        }
        return Ok(Self(lanes));
    }
}

/// A concurrency outside one to eight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcurrencyOutOfRange;

impl core::fmt::Display for ConcurrencyOutOfRange
{
    /// Render the failure, in ninfer's words.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("--max-concurrency must be in [1,8]");
    }
}

impl core::error::Error for ConcurrencyOutOfRange
{
}

/// A count of recurrent-state checkpoint slots; zero is admitted.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateSlots(pub u32);

impl StateSlots
{
    /// ninfer's default host state slots: eight.
    pub const HOST_DEFAULT: Self = Self(8);
}

impl core::str::FromStr for StateSlots
{
    type Err = core::num::ParseIntError;

    /// Parse a non-negative decimal slot count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the count is the parsed value.
    /// - provides: the command-line spelling of a slot count.
    /// - fails: with the integer parser's error on a negative value, anything
    ///   above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the sign boundary — zero admitted, minus one
    ///   refused.
    /// - witness: `tests::context_cache_capacities_parse_ninfers_ranges`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<u32>().map(Self);
    }
}

/// How many device checkpoint slots the context cache keeps beyond the
/// active lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStateSlots
{
    /// The Engine's default: one per lane.
    PerLane,
    /// This many.
    Exactly(StateSlots),
}

/// The pinned host memory the context cache keeps for KV, in bytes.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostKvBytes(pub usize);

impl HostKvBytes
{
    /// ninfer's default: 8192 MiB.
    pub const DEFAULT: Self = Self(8192_usize << 20_u32);
}

impl core::str::FromStr for HostKvBytes
{
    type Err = HostKvOutOfRange;

    /// Parse a non-negative decimal count of MiB whose bytes fit `usize`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the capacity is the parsed MiB in bytes.
    /// - provides: `--host-kv-mib`, with ninfer's range.
    /// - fails: on a negative value, a non-numeric string, or a count whose
    ///   bytes overflow `usize`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`HostKvOutOfRange`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at zero and at the overflow boundary.
    /// - witness: `tests::context_cache_capacities_parse_ninfers_ranges`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text
            .parse::<usize>()
            .ok()
            .and_then(|mib| return mib.checked_mul(1 << 20_u32))
            .map(Self)
            .ok_or(HostKvOutOfRange);
    }
}

/// A host KV capacity that is not a MiB count whose bytes fit `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostKvOutOfRange;

impl core::fmt::Display for HostKvOutOfRange
{
    /// Render the failure, in ninfer's words.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("--host-kv-mib is out of range");
    }
}

impl core::error::Error for HostKvOutOfRange
{
}

/// How far `RoPE` positions stretch past the native threshold: a finite
/// factor from one to sixteen, one leaving positions unscaled.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeFactor(f32);

impl RopeFactor
{
    /// Unscaled positions, ninfer's default.
    pub const ONE: Self = Self(1.0);
}

// The factor is never NaN: `ONE` is one and parsing admits only `[1,16]`, so
// equality is reflexive.
impl Eq for RopeFactor
{
}

impl From<RopeFactor> for f32
{
    /// Unwrap the factor.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(factor: RopeFactor) -> Self
    {
        return factor.0;
    }
}

impl core::str::FromStr for RopeFactor
{
    type Err = RopeFactorOutOfRange;

    /// Parse a decimal factor from one to sixteen.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the factor is `text` rounded to `f32`, and `text`
    ///   read as `f64` lies in `[1,16]`, the range ninfer's server checks in
    ///   double precision.
    /// - provides: `--rope-scaling-factor`.
    /// - fails: on a value outside the range, NaN, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`RopeFactorOutOfRange`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of both ends of the range.
    /// - witness: `tests::rope_scaling_parses_ninfers_ranges`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let wide = text
            .parse::<f64>()
            .map_err(|_malformed| return RopeFactorOutOfRange)?;
        if !(1.0_f64 ..= 16.0_f64).contains(&wide) {
            return Err(RopeFactorOutOfRange);
        }
        return text
            .parse::<f32>()
            .map(Self)
            .map_err(|_malformed| return RopeFactorOutOfRange);
    }
}

/// A `RoPE` scaling factor outside one to sixteen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeFactorOutOfRange;

impl core::fmt::Display for RopeFactorOutOfRange
{
    /// Render the failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("rope-scaling-factor must be in [1,16]");
    }
}

impl core::error::Error for RopeFactorOutOfRange
{
}

/// The native position threshold past which `RoPE` positions are scaled.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeThreshold(u32);

impl RopeThreshold
{
    /// ninfer's default: 262144 positions.
    pub const DEFAULT: Self = Self(1_u32 << 18_u32);
}

impl From<RopeThreshold> for u32
{
    /// Unwrap the threshold.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(threshold: RopeThreshold) -> Self
    {
        return threshold.0;
    }
}

impl core::str::FromStr for RopeThreshold
{
    type Err = core::num::TryFromIntError;

    /// Parse a non-negative decimal position count no larger than
    /// `i32::MAX`, ninfer's server's range.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the threshold is the parsed value.
    /// - provides: `--rope-scaling-original-context`.
    /// - fails: on a negative value, anything above `i32::MAX`, or a
    ///   non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::TryFromIntError`]: as stated; a non-numeric string is
    ///   reported as out of range too.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at zero, minus one, and either side of `i32::MAX`.
    /// - witness: `tests::rope_scaling_parses_ninfers_ranges`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let signed = text.parse::<i32>().unwrap_or(-1_i32);
        return u32::try_from(signed).map(Self);
    }
}

/// How long a request may wait for admission, in milliseconds; past it the
/// Engine refuses the request with a queue timeout.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingTimeout(core::num::NonZeroU32);

impl PendingTimeout
{
    /// ninfer's default: thirty seconds.
    pub const DEFAULT: Self = Self(match core::num::NonZeroU32::new(30_000) {
        | Some(milliseconds) => milliseconds,
        | None => core::num::NonZeroU32::MIN,
    });
}

impl From<PendingTimeout> for core::num::NonZeroU32
{
    /// Unwrap the timeout's milliseconds.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(timeout: PendingTimeout) -> Self
    {
        return timeout.0;
    }
}

impl core::str::FromStr for PendingTimeout
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal count of milliseconds.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the timeout is the parsed value, at least one.
    /// - provides: the command-line spelling of a pending timeout, refusing
    ///   zero as ninfer's server does.
    /// - fails: with the integer parser's error on zero, a negative value,
    ///   anything above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary.
    /// - witness: `tests::a_zero_pending_timeout_is_refused`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

/// A CUDA device ordinal, between zero and `u16::MAX`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOrdinal(u16);

impl From<DeviceOrdinal> for i32
{
    /// Widen the ordinal to the Engine's `int`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(ordinal: DeviceOrdinal) -> Self
    {
        return Self::from(ordinal.0);
    }
}

impl core::str::FromStr for DeviceOrdinal
{
    type Err = core::num::ParseIntError;

    /// Parse a non-negative decimal device ordinal.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the ordinal is the parsed value; a negative
    ///   ordinal names no device and is refused here rather than by the Engine.
    /// - provides: the command-line spelling of a device choice.
    /// - fails: with the integer parser's error on a negative value, anything
    ///   above `u16::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a `u16`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the sign boundary — zero admitted, minus one
    ///   refused.
    /// - witness: `tests::a_negative_device_is_refused`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<u16>().map(Self);
    }
}

/// Whether the Engine captures its decode rounds as CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraph
{
    /// Launch every kernel directly.
    Off,
    /// Capture decode rounds as CUDA graphs, as the served engine does.
    On,
}

impl core::str::FromStr for CudaGraph
{
    type Err = UnknownCudaGraph;

    /// Parse `off` or `on`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `off` is [`CudaGraph::Off`] and `on` is [`CudaGraph::On`].
    /// - provides: the command-line spelling of the choice.
    /// - fails: on any other text, case included.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`UnknownCudaGraph`]: `text` is neither spelling.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over both spellings and one refusal.
    /// - witness: `tests::cuda_graph_parses_its_two_spellings`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return match text {
            | "off" => Ok(Self::Off),
            | "on" => Ok(Self::On),
            | _ => Err(UnknownCudaGraph),
        };
    }
}

/// A CUDA graph choice that is neither `off` nor `on`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownCudaGraph;

impl core::fmt::Display for UnknownCudaGraph
{
    /// Render the failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("CUDA graph capture is `off` or `on`");
    }
}

impl core::error::Error for UnknownCudaGraph
{
}

/// Which chat template renders a chat prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate
{
    /// The template the artifact embeds.
    Artifact,
    /// The Jinja template in this file.
    File(std::path::PathBuf),
}

/// Everything an Engine is opened with except the round, which the plan
/// supplies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions
{
    /// The `.ninfer` artifact entry file.
    artifact: std::path::PathBuf,
    /// The CUDA device.
    device: DeviceOrdinal,
    /// The context ceiling.
    context: ContextLimit,
    /// The Main KV capacity.
    kv_capacity: KvCapacity,
    /// The KV storage.
    kv_storage: KvStorage,
    /// The prefill chunk.
    prefill_chunk: PrefillChunk,
    /// The concurrency.
    concurrency: Concurrency,
    /// Device checkpoint slots beyond the lanes.
    device_state: DeviceStateSlots,
    /// Host checkpoint slots.
    host_state: StateSlots,
    /// Host KV capacity.
    host_kv: HostKvBytes,
    /// `RoPE` scaling factor.
    rope_factor: RopeFactor,
    /// `RoPE` native threshold.
    rope_threshold: RopeThreshold,
    /// The pending timeout.
    pending_timeout: PendingTimeout,
    /// CUDA graph capture.
    cuda_graph: CudaGraph,
    /// The chat template.
    chat_template: ChatTemplate,
}

impl EngineOptions
{
    /// Gather the options, with the artifact's own chat template and ninfer's
    /// default pending timeout.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        artifact: std::path::PathBuf,
        device: DeviceOrdinal,
        context: ContextLimit,
        cuda_graph: CudaGraph,
    ) -> Self
    {
        return Self {
            artifact,
            device,
            context,
            kv_capacity: KvCapacity::Tokens(context.0),
            kv_storage: KvStorage::BFloat16,
            prefill_chunk: PrefillChunk::DEFAULT,
            concurrency: Concurrency::ONE,
            device_state: DeviceStateSlots::PerLane,
            host_state: StateSlots::HOST_DEFAULT,
            host_kv: HostKvBytes::DEFAULT,
            rope_factor: RopeFactor::ONE,
            rope_threshold: RopeThreshold::DEFAULT,
            pending_timeout: PendingTimeout::DEFAULT,
            cuda_graph,
            chat_template: ChatTemplate::Artifact,
        };
    }

    /// The same options with `timeout` bounding each request's wait for
    /// admission.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_pending_timeout(
        self,
        timeout: PendingTimeout,
    ) -> Self
    {
        return Self {
            pending_timeout: timeout,
            ..self
        };
    }

    /// The pending timeout.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn pending_timeout(&self) -> PendingTimeout
    {
        return self.pending_timeout;
    }

    /// The same options with `capacity` sizing the Main KV cache.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the capacity is `capacity`, every other option
    ///   unchanged.
    /// - provides: `--kv-capacity`, with ninfer's server's cross-check.
    /// - fails: when `capacity` is a token count below the context ceiling.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`KvBelowContext`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of the context ceiling, and automatic
    ///   capacity exempt.
    /// - witness: `tests::a_kv_capacity_below_the_context_is_refused`
    #[inline]
    pub fn with_kv_capacity(
        self,
        capacity: KvCapacity,
    ) -> Result<Self, KvBelowContext>
    {
        if let KvCapacity::Tokens(tokens) = capacity
            && tokens < self.context.0
        {
            return Err(KvBelowContext);
        }
        return Ok(Self {
            kv_capacity: capacity,
            ..self
        });
    }

    /// The Main KV capacity; the context ceiling unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn kv_capacity(&self) -> KvCapacity
    {
        return self.kv_capacity;
    }

    /// The same options with `storage` storing the KV cache.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_kv_storage(
        self,
        storage: KvStorage,
    ) -> Self
    {
        return Self {
            kv_storage: storage,
            ..self
        };
    }

    /// The KV storage; bfloat16 unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn kv_storage(&self) -> KvStorage
    {
        return self.kv_storage;
    }

    /// The same options with `chunk` tokens prefilled per step.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_prefill_chunk(
        self,
        chunk: PrefillChunk,
    ) -> Self
    {
        return Self {
            prefill_chunk: chunk,
            ..self
        };
    }

    /// The prefill chunk; ninfer's 1024 unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn prefill_chunk(&self) -> PrefillChunk
    {
        return self.prefill_chunk;
    }

    /// The same options with `concurrency` requests running at once.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_concurrency(
        self,
        concurrency: Concurrency,
    ) -> Self
    {
        return Self {
            concurrency,
            ..self
        };
    }

    /// The concurrency; one unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn concurrency(&self) -> Concurrency
    {
        return self.concurrency;
    }

    /// The same options with the context cache keeping `device` device
    /// checkpoint slots beyond the lanes, `host` host slots, and `host_kv`
    /// bytes of host KV.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_context_cache(
        self,
        device: DeviceStateSlots,
        host: StateSlots,
        host_kv: HostKvBytes,
    ) -> Self
    {
        return Self {
            device_state: device,
            host_state: host,
            host_kv,
            ..self
        };
    }

    /// Device checkpoint slots beyond the lanes; one per lane unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn device_state(&self) -> DeviceStateSlots
    {
        return self.device_state;
    }

    /// Host checkpoint slots; ninfer's eight unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn host_state(&self) -> StateSlots
    {
        return self.host_state;
    }

    /// Host KV capacity; ninfer's 8192 MiB unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn host_kv(&self) -> HostKvBytes
    {
        return self.host_kv;
    }

    /// The same options with `RoPE` positions past `threshold` scaled by
    /// `factor`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_rope_scaling(
        self,
        factor: RopeFactor,
        threshold: RopeThreshold,
    ) -> Self
    {
        return Self {
            rope_factor: factor,
            rope_threshold: threshold,
            ..self
        };
    }

    /// The `RoPE` scaling factor; one unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn rope_factor(&self) -> RopeFactor
    {
        return self.rope_factor;
    }

    /// The `RoPE` native threshold; ninfer's 262144 unless set.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn rope_threshold(&self) -> RopeThreshold
    {
        return self.rope_threshold;
    }

    /// The same options with `template` rendering chat prompts.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_chat_template(
        self,
        template: ChatTemplate,
    ) -> Self
    {
        return Self {
            chat_template: template,
            ..self
        };
    }

    /// The chat template.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn chat_template(&self) -> &ChatTemplate
    {
        return &self.chat_template;
    }

    /// The artifact.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn artifact(&self) -> &std::path::Path
    {
        return &self.artifact;
    }

    /// The device.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn device(&self) -> DeviceOrdinal
    {
        return self.device;
    }

    /// The context ceiling.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn context(&self) -> ContextLimit
    {
        return self.context;
    }

    /// CUDA graph capture.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn cuda_graph(&self) -> CudaGraph
    {
        return self.cuda_graph;
    }
}

/// Tests for the options' parsing boundaries.
#[cfg(test)]
mod tests
{
    use core::str::FromStr as _;

    use super::Concurrency;
    use super::ConcurrencyOutOfRange;
    use super::ContextLimit;
    use super::CudaGraph;
    use super::DeviceOrdinal;
    use super::EngineOptions;
    use super::HostKvBytes;
    use super::HostKvOutOfRange;
    use super::KvBelowContext;
    use super::KvCapacity;
    use super::KvStorage;
    use super::MalformedKvCapacity;
    use super::PendingTimeout;
    use super::PrefillChunk;
    use super::RopeFactor;
    use super::RopeFactorOutOfRange;
    use super::RopeThreshold;
    use super::StateSlots;
    use super::UnknownCudaGraph;
    use super::UnknownKvStorage;

    /// A zero ceiling is refused and one is admitted.
    #[test]
    fn a_zero_context_is_refused()
    {
        assert!(
            ContextLimit::from_str("0").is_err(),
            "a request needs at least one token"
        );
        assert!(
            ContextLimit::from_str("1").is_ok(),
            "one token is a ceiling"
        );
    }

    /// A zero timeout is refused, as ninfer's server refuses it; one
    /// millisecond is admitted.
    #[test]
    fn a_zero_pending_timeout_is_refused()
    {
        assert!(
            PendingTimeout::from_str("0").is_err(),
            "a request needs time to be admitted"
        );
        assert!(
            PendingTimeout::from_str("1").is_ok(),
            "one millisecond is a timeout"
        );
    }

    /// `auto` and a positive count parse; zero and another spelling do not.
    #[test]
    fn kv_capacity_parses_auto_and_positive_counts()
    {
        assert_eq!(
            KvCapacity::from_str("auto"),
            Ok(KvCapacity::Automatic),
            "auto"
        );
        assert_eq!(
            KvCapacity::from_str("1"),
            Ok(KvCapacity::Tokens(core::num::NonZeroU32::MIN)),
            "one token"
        );
        assert_eq!(KvCapacity::from_str("0"), Err(MalformedKvCapacity), "zero");
        assert_eq!(
            KvCapacity::from_str("Auto"),
            Err(MalformedKvCapacity),
            "case matters"
        );
    }

    /// A token capacity below the context is refused; one equal to it, and an
    /// automatic one, are kept.
    #[test]
    fn a_kv_capacity_below_the_context_is_refused()
    {
        let options = EngineOptions::new(
            std::path::PathBuf::from("model.ninfer"),
            DeviceOrdinal::from_str("0").unwrap(),
            ContextLimit::from_str("256").unwrap(),
            CudaGraph::On,
        );
        assert_eq!(
            options.kv_capacity(),
            KvCapacity::from_str("256").unwrap(),
            "the capacity defaults to the context"
        );
        assert_eq!(
            options
                .clone()
                .with_kv_capacity(KvCapacity::from_str("255").unwrap()),
            Err(KvBelowContext),
            "one token short"
        );
        let equal = KvCapacity::from_str("256").unwrap();
        assert_eq!(
            options
                .clone()
                .with_kv_capacity(equal)
                .map(|kept| return kept.kv_capacity()),
            Ok(equal),
            "equal to the context"
        );
        assert_eq!(
            options
                .with_kv_capacity(KvCapacity::Automatic)
                .map(|kept| return kept.kv_capacity()),
            Ok(KvCapacity::Automatic),
            "automatic is sized by the Engine"
        );
    }

    /// ninfer's five spellings parse; another does not.
    #[test]
    fn kv_storage_parses_ninfers_five_spellings()
    {
        for (text, storage) in [
            ("bf16", KvStorage::BFloat16),
            ("int8", KvStorage::Int8),
            ("fp8", KvStorage::Fp8),
            ("nvfp4", KvStorage::Nvfp4),
            ("k8v4", KvStorage::Fp8KeyNvfp4Value),
        ] {
            assert_eq!(KvStorage::from_str(text), Ok(storage), "{text}");
        }
        assert_eq!(
            KvStorage::from_str("NVFP4"),
            Err(UnknownKvStorage),
            "case matters"
        );
    }

    /// 128 and 256 are chunks; zero, 127 and 129 are not.
    #[test]
    fn a_prefill_chunk_is_a_positive_multiple_of_128()
    {
        for text in ["128", "256"] {
            assert!(PrefillChunk::from_str(text).is_ok(), "{text} is a chunk");
        }
        for text in ["0", "127", "129"] {
            assert!(
                PrefillChunk::from_str(text).is_err(),
                "{text} is not a chunk"
            );
        }
    }

    /// One and eight are concurrencies; zero and nine are not.
    #[test]
    fn concurrency_is_one_to_eight()
    {
        assert_eq!(Concurrency::from_str("1"), Ok(Concurrency::ONE), "one lane");
        assert!(Concurrency::from_str("8").is_ok(), "eight lanes");
        assert_eq!(
            Concurrency::from_str("0"),
            Err(ConcurrencyOutOfRange),
            "no lane"
        );
        assert_eq!(
            Concurrency::from_str("9"),
            Err(ConcurrencyOutOfRange),
            "past ninfer's range"
        );
    }

    /// Slot counts admit zero and refuse minus one; host KV admits zero, reads
    /// MiB, and refuses a count whose bytes overflow.
    #[test]
    fn context_cache_capacities_parse_ninfers_ranges()
    {
        assert_eq!(StateSlots::from_str("0"), Ok(StateSlots(0)), "no slots");
        assert!(StateSlots::from_str("-1").is_err(), "no negative count");
        assert_eq!(HostKvBytes::from_str("0"), Ok(HostKvBytes(0)), "no host KV");
        assert_eq!(
            HostKvBytes::from_str("36864"),
            Ok(HostKvBytes(36864 << 20_u32)),
            "MiB are read as bytes"
        );
        let largest = (usize::MAX >> 20_u32).to_string();
        assert!(
            HostKvBytes::from_str(&largest).is_ok(),
            "the largest fitting count"
        );
        let overflowing = ((usize::MAX >> 20_u32) + 1).to_string();
        assert_eq!(
            HostKvBytes::from_str(&overflowing),
            Err(HostKvOutOfRange),
            "one MiB more overflows"
        );
    }

    /// The factor admits one and sixteen and refuses just outside them and
    /// NaN; the threshold admits zero and `i32::MAX` and refuses minus one and
    /// one past it.
    #[test]
    fn rope_scaling_parses_ninfers_ranges()
    {
        assert_eq!(
            RopeFactor::from_str("1").map(f32::from),
            Ok(1.0_f32),
            "unscaled"
        );
        assert_eq!(
            RopeFactor::from_str("16").map(f32::from),
            Ok(16.0_f32),
            "the widest"
        );
        assert_eq!(
            RopeFactor::from_str("2.4371").map(f32::from),
            Ok(2.4371_f32),
            "the served factor"
        );
        for text in ["0.999", "16.001", "NaN", "x"] {
            assert_eq!(
                RopeFactor::from_str(text),
                Err(RopeFactorOutOfRange),
                "{text}"
            );
        }
        assert_eq!(
            RopeThreshold::from_str("0").map(u32::from),
            Ok(0_u32),
            "zero"
        );
        let widest = i32::MAX.to_string();
        assert!(RopeThreshold::from_str(&widest).is_ok(), "i32::MAX");
        let past = (i64::from(i32::MAX) + 1).to_string();
        assert!(RopeThreshold::from_str(&past).is_err(), "one past i32::MAX");
        assert!(
            RopeThreshold::from_str("-1").is_err(),
            "no negative threshold"
        );
    }

    /// Minus one names no device; zero does.
    #[test]
    fn a_negative_device_is_refused()
    {
        assert!(
            DeviceOrdinal::from_str("-1").is_err(),
            "no device is negative"
        );
        assert_eq!(
            DeviceOrdinal::from_str("0").map(i32::from),
            Ok(0_i32),
            "device zero"
        );
    }

    /// The two spellings parse; anything else is refused.
    #[test]
    fn cuda_graph_parses_its_two_spellings()
    {
        assert_eq!(CudaGraph::from_str("off"), Ok(CudaGraph::Off), "off");
        assert_eq!(CudaGraph::from_str("on"), Ok(CudaGraph::On), "on");
        assert_eq!(
            CudaGraph::from_str("On"),
            Err(UnknownCudaGraph),
            "case matters"
        );
    }
}
