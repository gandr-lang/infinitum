//! What a chat request produced.

use infinitum_round::TokenCount;
use infinitum_round::TokenId;

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

/// A count a backend keeps over one generation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Tally(pub u64);

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
}
