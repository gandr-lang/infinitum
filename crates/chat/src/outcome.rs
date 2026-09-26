//! What a chat request produced.

use infinitum_round::TokenCount;
use infinitum_round::TokenId;

use crate::request::ThinkingBudget;

/// The output channel a text delta belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel
{
    /// The answer.
    Content,
    /// Thinking.
    Reasoning,
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Finish
{
    /// The output limit bound.
    OutputLimit,
    /// The context ceiling bound.
    ContextCapacity,
    /// The model produced a stop token.
    StopToken,
    /// A stop string matched.
    StopString,
    /// The request was cancelled.
    Cancelled,
    /// The backend reported no reason.
    Unfinished,
}

/// One tool call the model generated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedToolCall
{
    /// The called function's name.
    pub name: String,
    /// The call's arguments, as JSON text.
    pub arguments: String,
}

/// The prompt accounting fixed at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Admission
{
    /// Tokens in the rendered prompt.
    pub prompt_tokens: TokenCount,
    /// Prompt tokens served from a cached prefix rather than prefilled.
    pub reused_tokens: TokenCount,
}

/// How far prefill has come, cumulative, as the backend's request clock
/// measures it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptProgress
{
    /// Tokens in the rendered prompt.
    pub total: TokenCount,
    /// Prompt tokens served from a cached prefix.
    pub reused: TokenCount,
    /// Prompt tokens processed so far, the reused ones included.
    pub processed: TokenCount,
    /// Time since the request's clock started.
    pub elapsed: core::time::Duration,
}

/// Cumulative timings at one output commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingObservation
{
    /// Tokens accepted into the sequence so far, injected control tokens
    /// included.
    pub generated: TokenCount,
    /// Time the prompt took.
    pub prompt_elapsed: core::time::Duration,
    /// Time generation has taken so far.
    pub generation_elapsed: core::time::Duration,
}

/// A count a backend keeps over one generation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Tally(pub u64);

/// Where an admitted prompt's reused prefix came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReusePath
{
    /// Nothing cached matched; the prompt prefilled from the root.
    Root,
    /// The end of a private continuation.
    PrivateEndpoint,
    /// A private continuation's closed turn.
    TurnClosure,
    /// A replayed private response.
    ResponseReplay,
    /// A private long anchor.
    LongAnchor,
    /// A shared stable prefix.
    SharedPrefix,
}

/// A thinking budget and what the model spent of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThinkingSpend
{
    /// The budget the request ran under.
    pub budget: ThinkingBudget,
    /// Model tokens accepted while budgeted thinking stayed open.
    pub model_tokens: TokenCount,
    /// Control tokens the backend injected to close thinking.
    pub injected_tokens: TokenCount,
}

/// How one request ran, for the operational log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Telemetry
{
    /// From the request's arrival to its first output token.
    pub first_token: core::time::Duration,
    /// From the request's arrival to its end.
    pub total: core::time::Duration,
    /// Prefill execution time.
    pub prefill: core::time::Duration,
    /// Decode execution time.
    pub decode: core::time::Duration,
    /// Time spent waiting for admission.
    pub queue_wait: core::time::Duration,
    /// Where the reused prefix came from.
    pub reuse: ReusePath,
    /// The thinking budget and its spend.
    pub thinking: ThinkingSpend,
    /// Drafted tokens accepted at each draft position, first position first.
    pub accepted_per_position: Vec<Tally>,
}

/// What one request produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatOutcome
{
    /// The answer text.
    pub content: String,
    /// The thinking text.
    pub reasoning: String,
    /// Tool calls parsed from the output, in order.
    pub tool_calls: Vec<GeneratedToolCall>,
    /// Why generation ended.
    pub finish: Finish,
    /// The prompt accounting.
    pub admission: Admission,
    /// Every generated id, injected control tokens included.
    pub generated: Vec<TokenId>,
    /// Generated tokens that belong to thinking.
    pub reasoning_tokens: TokenCount,
    /// From admission to the first accepted output token.
    pub prompt_wall: core::time::Duration,
    /// From the first to the last accepted output token.
    pub generation_wall: core::time::Duration,
    /// Tokens drafted over every speculative round.
    pub drafted: Tally,
    /// Drafted tokens accepted.
    pub accepted: Tally,
    /// How the request ran.
    pub telemetry: Telemetry,
}
