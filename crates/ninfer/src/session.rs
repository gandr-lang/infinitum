//! An open ninfer Engine, driven through the bridge.
//!
//! [`Session::generate`] hands ninfer a round controller that forwards every
//! round to an [`infinitum_round::Preview`]: infinitum makes the output
//! decision, ninfer applies it and commits.

use infinitum_round::CountOverflow;
use infinitum_round::Maybe;
use infinitum_round::Preview;
use infinitum_round::RoundKind;
use infinitum_round::RoundOffer;
use infinitum_round::RoundRecord;
use infinitum_round::RoundVerdict;
use infinitum_round::TokenCount;
use infinitum_round::TokenId;

use crate::bridge::ffi;
use crate::options::ChatTemplate;
use crate::options::CudaGraph;
use crate::options::DeviceStateSlots;
use crate::options::EngineOptions;
use crate::options::KvCapacity;
use crate::options::KvStorage;
use crate::plan::DFlash2Plan;
use crate::text::RawText;
use crate::text::RenderedBytes;

/// The Engine operation a failure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation
{
    /// Opening the Engine.
    Open,
    /// Tokenizing the prompt.
    Tokenize,
    /// Generating.
    Generate,
    /// Rendering ids to bytes.
    Detokenize,
}

impl core::fmt::Display for Operation
{
    /// Render the operation.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(match *self {
            | Self::Open => "opening the engine",
            | Self::Tokenize => "tokenizing",
            | Self::Generate => "generating",
            | Self::Detokenize => "detokenizing",
        });
    }
}

/// How ninfer failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrownKind
{
    /// It threw `std::invalid_argument`.
    InvalidArgument,
    /// It threw another `std::exception`.
    Runtime,
    /// It threw a value that is not a `std::exception`.
    Unknown,
}

impl core::fmt::Display for ThrownKind
{
    /// Render the exception's kind.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(match *self {
            | Self::InvalidArgument => "std::invalid_argument",
            | Self::Runtime => "a std::exception",
            | Self::Unknown => "a non-exception value",
        });
    }
}

/// A failure opening or driving the Engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineFailure
{
    /// The artifact path is not Unicode, which the bridge's string cannot
    /// carry.
    ArtifactPath(std::path::PathBuf),
    /// The chat template path is not Unicode.
    TemplatePath(std::path::PathBuf),
    /// ninfer threw.
    Thrown
    {
        /// What was being done.
        operation: Operation,
        /// What kind of exception.
        kind: ThrownKind,
        /// Its message.
        message: String,
    },
    /// A count left its range while the preview reviewed a round.
    Preview(CountOverflow),
    /// A prompt longer than the bridge's count.
    PromptLength,
}

impl core::fmt::Display for EngineFailure
{
    /// Render the failure for an operator.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::ArtifactPath(ref path) => {
                write!(f, "the artifact path {} is not Unicode", path.display())
            },
            | Self::TemplatePath(ref path) => {
                write!(
                    f,
                    "the chat template path {} is not Unicode",
                    path.display()
                )
            },
            | Self::Thrown {
                operation,
                kind,
                ref message,
            } => write!(f, "ninfer failed {operation}, throwing {kind}: {message}"),
            | Self::Preview(ref overflow) => write!(f, "the round preview failed: {overflow}"),
            | Self::PromptLength => f.write_str("the prompt is longer than a count can hold"),
        };
    }
}

impl core::error::Error for EngineFailure
{
}

/// An outcome the adapter has not written yet.
///
/// # Specification
/// trivial.
pub fn pending() -> ffi::Outcome
{
    return ffi::Outcome {
        status: ffi::Status::Unknown,
        refusal: ffi::Refusal::None,
        message: String::new(),
    };
}

/// Read an adapter outcome.
///
/// # Specification
/// - requires: the adapter wrote `outcome`.
/// - ensures: `Ok` exactly for [`ffi::Status::Completed`]; otherwise the
///   failure carries the operation, the exception's kind, and its message.
/// - provides: the one reading of the bridge's status.
/// - fails: as stated.
/// - panics: none.
///
/// # Errors
/// - [`EngineFailure::Thrown`]: ninfer threw.
pub fn check(
    operation: Operation,
    outcome: ffi::Outcome,
) -> Result<(), EngineFailure>
{
    let kind = match outcome.status {
        | ffi::Status::Completed => return Ok(()),
        | ffi::Status::InvalidArgument | ffi::Status::Refused => ThrownKind::InvalidArgument,
        | ffi::Status::Runtime => ThrownKind::Runtime,
        | _ => ThrownKind::Unknown,
    };
    return Err(EngineFailure::Thrown {
        operation,
        kind,
        message: outcome.message,
    });
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish
{
    /// ninfer reported no reason.
    Unfinished,
    /// The budget, or a round verdict's limit, was reached.
    OutputLimit,
    /// The context ceiling was reached.
    ContextCapacity,
    /// A stop token was generated.
    StopToken,
    /// A stop string was generated.
    StopString,
    /// The request was cancelled.
    Cancelled,
    /// A reason the bridge does not know.
    Unrecognized,
}

impl core::fmt::Display for Finish
{
    /// Render the reason.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(match *self {
            | Self::Unfinished => "unfinished",
            | Self::OutputLimit => "output limit",
            | Self::ContextCapacity => "context capacity",
            | Self::StopToken => "stop token",
            | Self::StopString => "stop string",
            | Self::Cancelled => "cancelled",
            | Self::Unrecognized => "unrecognized",
        });
    }
}

/// How the speculation of one generation went, as ninfer counted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation
{
    /// Speculative rounds run.
    rounds: RoundTally,
    /// Tokens drafted.
    drafted: RoundTally,
    /// Drafted tokens accepted.
    accepted: RoundTally,
    /// Steps that fell back to plain decode.
    fallback_steps: RoundTally,
}

/// A tally ninfer keeps over one generation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoundTally(u64);

impl From<RoundTally> for u64
{
    /// Unwrap the tally.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(tally: RoundTally) -> Self
    {
        return tally.0;
    }
}

impl core::fmt::Display for RoundTally
{
    /// Render the tally.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return core::fmt::Display::fmt(&self.0, f);
    }
}

impl Speculation
{
    /// Speculative rounds run.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn rounds(&self) -> RoundTally
    {
        return self.rounds;
    }

    /// Tokens drafted.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn drafted(&self) -> RoundTally
    {
        return self.drafted;
    }

    /// Drafted tokens accepted.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn accepted(&self) -> RoundTally
    {
        return self.accepted;
    }

    /// Steps that fell back to plain decode.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn fallback_steps(&self) -> RoundTally
    {
        return self.fallback_steps;
    }
}

/// One generation's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation
{
    /// The generated ids.
    generated: Vec<TokenId>,
    /// Why it ended.
    finish: Finish,
    /// ninfer's speculation tallies.
    speculation: Speculation,
    /// Wall time from the first to the last generated token.
    wall: core::time::Duration,
    /// Every round infinitum's preview reviewed.
    rounds: Vec<RoundRecord>,
}

impl Generation
{
    /// The generated ids.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn generated(&self) -> &[TokenId]
    {
        return &self.generated;
    }

    /// Why it ended.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn finish(&self) -> Finish
    {
        return self.finish;
    }

    /// ninfer's speculation tallies.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn speculation(&self) -> Speculation
    {
        return self.speculation;
    }

    /// Wall time from the first to the last generated token.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn wall(&self) -> core::time::Duration
    {
        return self.wall;
    }

    /// Every round the preview reviewed, in order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn rounds(&self) -> &[RoundRecord]
    {
        return &self.rounds;
    }
}

/// The Rust side of a generation's round reviews.
pub struct ReviewSink
{
    /// infinitum's decision and ledger.
    preview: Preview,
    /// The offer's ids, reused across rounds.
    scratch: Vec<TokenId>,
    /// The first failure, which ends every later review with no decision.
    failure: Maybe<CountOverflow, Reviewing>,
}

impl ReviewSink
{
    /// A sink whose preview admits at most `budget` tokens.
    ///
    /// # Specification
    /// trivial.
    pub const fn new(budget: TokenCount) -> Self
    {
        return Self {
            preview: Preview::new(budget),
            scratch: Vec::new(),
            failure: Maybe::Absent(Reviewing::Clean),
        };
    }

    /// The first failure a review met, or why there is none.
    ///
    /// # Specification
    /// trivial.
    pub const fn failure(&self) -> Maybe<CountOverflow, Reviewing>
    {
        return self.failure;
    }
}

/// Why a sink holds no failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reviewing
{
    /// Every review so far succeeded.
    Clean,
}

/// Review one round: the bridge's entry into the preview.
///
/// # Specification
/// - requires: rounds arrive in order, one at a time.
/// - ensures: the answer is the preview's verdict for the offer; after a
///   preview failure the failure is kept and every answer is continue, so the
///   generation completes and [`Session::generate`] reports the failure.
/// - provides: the host decision ninfer's Engine asks for each round.
/// - fails: never across the boundary; a failure is held in the sink.
/// - panics: none.
// The `cxx` border: the bridge declares this signature and generates its C++
// caller, and a `cxx` shared signature carries only primitives, slices of
// them, and bridge-declared types, so no nominal token type fits here. The ids
// are wrapped as `TokenId` on the first line past the border.
pub fn review_round(
    sink: &mut ReviewSink,
    licensed: &[i32],
    kind: ffi::RoundKind,
) -> ffi::ReviewAnswer
{
    let proceed = ffi::ReviewAnswer {
        verdict: ffi::Verdict::Continue,
        limit: 0,
    };
    if let Maybe::Present(_) = sink.failure {
        return proceed;
    }
    sink.scratch.clear();
    sink.scratch
        .extend(licensed.iter().copied().map(TokenId::from));
    let round = if kind == ffi::RoundKind::Decode {
        RoundKind::Decode
    }
    else {
        RoundKind::PrefillFinalization
    };
    return match sink.preview.review(RoundOffer::new(&sink.scratch, round)) {
        | Ok(RoundVerdict::Continue) => proceed,
        | Ok(RoundVerdict::Limit(limit)) => ffi::ReviewAnswer {
            verdict: ffi::Verdict::Limit,
            limit: core::num::NonZeroU32::from(limit).get(),
        },
        | Err(overflow) => {
            sink.failure = Maybe::Present(overflow);
            proceed
        },
    };
}

/// An open ninfer Engine, running the DFlash2 round a plan chose.
#[repr(transparent)]
pub struct Session
{
    /// The adapter's Engine.
    engine: cxx::UniquePtr<ffi::Session>,
}

impl core::fmt::Debug for Session
{
    /// Render the session without its Engine.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.debug_struct("Session").finish_non_exhaustive();
    }
}

impl Session
{
    /// Open an Engine for `options` running `plan`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the Engine runs DFlash2 at the plan's width with
    ///   the optimized proposal head and an explicit KV capacity equal to the
    ///   context ceiling.
    /// - provides: the only way to reach the Engine.
    /// - fails: when the artifact path is not Unicode, or ninfer throws while
    ///   opening.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure::ArtifactPath`], [`EngineFailure::Thrown`]: as named.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — the Engine's own startup validation, exercised on a
    ///   GPU host by the `generate` acceptance run.
    #[inline]
    pub fn open(
        options: &EngineOptions,
        plan: DFlash2Plan,
    ) -> Result<Self, EngineFailure>
    {
        let artifact = options
            .artifact()
            .to_str()
            .ok_or_else(|| return EngineFailure::ArtifactPath(options.artifact().to_path_buf()))?;
        let chat_template = match *options.chat_template() {
            | ChatTemplate::Artifact => String::new(),
            | ChatTemplate::File(ref path) => {
                let text = path
                    .to_str()
                    .ok_or_else(|| return EngineFailure::TemplatePath(path.clone()))?;
                String::from(text)
            },
        };
        let config = ffi::EngineConfig {
            artifact: String::from(artifact),
            device: i32::from(options.device()),
            max_context: core::num::NonZeroU32::from(options.context()).get(),
            kv_sizing: match options.kv_capacity() {
                | KvCapacity::Tokens(_) => ffi::KvSizing::Explicit,
                | KvCapacity::Automatic => ffi::KvSizing::Automatic,
            },
            kv_capacity: match options.kv_capacity() {
                | KvCapacity::Tokens(tokens) => tokens.get(),
                | KvCapacity::Automatic => 0,
            },
            kv_storage: match options.kv_storage() {
                | KvStorage::BFloat16 => ffi::KvStorage::BFloat16,
                | KvStorage::Int8 => ffi::KvStorage::Int8,
                | KvStorage::Fp8 => ffi::KvStorage::Fp8,
                | KvStorage::Nvfp4 => ffi::KvStorage::Nvfp4,
                | KvStorage::Fp8KeyNvfp4Value => ffi::KvStorage::Fp8KeyNvfp4Value,
            },
            prefill_chunk: core::num::NonZeroU32::from(options.prefill_chunk()).get(),
            max_concurrency: core::num::NonZeroU32::from(options.concurrency()).get(),
            device_state_set: matches!(options.device_state(), DeviceStateSlots::Exactly(_)),
            device_state_slots: match options.device_state() {
                | DeviceStateSlots::PerLane => 0,
                | DeviceStateSlots::Exactly(slots) => slots.0,
            },
            host_state_slots: options.host_state().0,
            host_kv_bytes: options.host_kv().0,
            pending_timeout_ms: core::num::NonZeroU32::from(options.pending_timeout()).get(),
            draft_width: core::num::NonZeroU32::from(plan.width()).get(),
            cuda_graph: match options.cuda_graph() {
                | CudaGraph::Off => ffi::CudaGraph::Off,
                | CudaGraph::On => ffi::CudaGraph::On,
            },
            chat_template,
        };
        let mut outcome = pending();
        let engine = ffi::open_session(&config, &mut outcome);
        check(Operation::Open, outcome)?;
        return Ok(Self { engine });
    }

    /// The adapter's Engine.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the Engine [`Session::open`] stored, which is non-null
    ///   because `open` returns a session only on success.
    /// - provides: shared access for the read-only calls.
    /// - fails: never.
    /// - panics: none; a null Engine is unreachable, and reading one would be
    ///   reported by `cxx` as an abort rather than a panic.
    pub(crate) fn engine(&self) -> &ffi::Session
    {
        let Some(engine) = self.engine.as_ref()
        else {
            std::process::abort();
        };
        return engine;
    }

    /// Encode `text` with the artifact's tokenizer; no template or special
    /// token is added.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the ids ninfer's tokenizer produced.
    /// - provides: the prompt's ids.
    /// - fails: when ninfer throws.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure::Thrown`]: ninfer threw.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — exercised on a GPU host by the `generate` acceptance
    ///   run.
    #[inline]
    pub fn tokenize(
        &self,
        text: RawText<'_>,
    ) -> Result<Vec<TokenId>, EngineFailure>
    {
        let mut ids = Vec::new();
        let mut outcome = pending();
        ffi::tokenize(self.engine(), text.as_ref(), &mut ids, &mut outcome);
        check(Operation::Tokenize, outcome)?;
        return Ok(ids.into_iter().map(TokenId::from).collect());
    }

    /// Generate greedily from `prompt`, at most `budget` tokens, with every
    /// round reviewed by an [`infinitum_round::Preview`] over that budget.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the generation carries ninfer's ids, finish
    ///   reason, tallies and wall time, and the preview's ledger of every
    ///   round.
    /// - provides: the infinitum-owned DFlash2 round on ninfer.
    /// - fails: when ninfer throws, the prompt is longer than a count, or the
    ///   preview's counts overflow.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure::Thrown`], [`EngineFailure::PromptLength`],
    ///   [`EngineFailure::Preview`]: as named.
    ///
    /// # Adequacy
    /// - hypothesis: L2 — greedy ids equal ninfer's own DFlash2 greedy run on
    ///   the same artifact, checked on a GPU host by the `generate` acceptance
    ///   run.
    #[inline]
    pub fn generate(
        &mut self,
        prompt: &[TokenId],
        budget: core::num::NonZeroU32,
    ) -> Result<Generation, EngineFailure>
    {
        TokenCount::of(prompt).map_err(|_overflow| return EngineFailure::PromptLength)?;
        let ids: Vec<i32> = prompt.iter().copied().map(i32::from).collect();
        let mut sink = ReviewSink {
            preview: Preview::new(TokenCount::from(budget)),
            scratch: Vec::new(),
            failure: Maybe::Absent(Reviewing::Clean),
        };
        let mut record = ffi::GenerationRecord {
            generated: Vec::new(),
            finish: ffi::Finish::Unrecognized,
            decode_rounds: 0,
            drafted: 0,
            accepted: 0,
            fallback_steps: 0,
            generation_wall_ns: 0,
        };
        let mut outcome = pending();
        let Some(engine) = self.engine.as_mut()
        else {
            std::process::abort();
        };
        ffi::generate(
            engine,
            &ids,
            budget.get(),
            &mut sink,
            &mut record,
            &mut outcome,
        );
        check(Operation::Generate, outcome)?;
        if let Maybe::Present(overflow) = sink.failure {
            return Err(EngineFailure::Preview(overflow));
        }
        let finish = match record.finish {
            | ffi::Finish::Unfinished => Finish::Unfinished,
            | ffi::Finish::OutputLimit => Finish::OutputLimit,
            | ffi::Finish::ContextCapacity => Finish::ContextCapacity,
            | ffi::Finish::StopToken => Finish::StopToken,
            | ffi::Finish::StopString => Finish::StopString,
            | ffi::Finish::Cancelled => Finish::Cancelled,
            | _ => Finish::Unrecognized,
        };
        return Ok(Generation {
            generated: record.generated.into_iter().map(TokenId::from).collect(),
            finish,
            speculation: Speculation {
                rounds: RoundTally(record.decode_rounds),
                drafted: RoundTally(record.drafted),
                accepted: RoundTally(record.accepted),
                fallback_steps: RoundTally(record.fallback_steps),
            },
            wall: core::time::Duration::from_nanos(record.generation_wall_ns),
            rounds: sink.preview.rounds().to_vec(),
        });
    }

    /// Render `ids` to bytes: each id's bytes in order, special tokens
    /// included, not UTF-8 validated.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success ninfer's rendering.
    /// - provides: the generated text.
    /// - fails: when ninfer throws, as it does for an id outside the
    ///   vocabulary.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure::Thrown`]: ninfer threw.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — exercised on a GPU host by the `generate` acceptance
    ///   run.
    #[inline]
    pub fn detokenize(
        &self,
        ids: &[TokenId],
    ) -> Result<RenderedBytes, EngineFailure>
    {
        let raw: Vec<i32> = ids.iter().copied().map(i32::from).collect();
        let mut bytes = Vec::new();
        let mut outcome = pending();
        ffi::detokenize(self.engine(), &raw, &mut bytes, &mut outcome);
        check(Operation::Detokenize, outcome)?;
        return Ok(RenderedBytes::from(bytes));
    }
}
