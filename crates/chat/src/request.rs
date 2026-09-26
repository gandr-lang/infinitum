//! What a chat request asks for: the conversation, the prompt options, and
//! the generation settings.

use infinitum_round::TokenCount;

/// Who wrote a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role
{
    /// A system instruction.
    System,
    /// A developer instruction.
    Developer,
    /// The user.
    User,
    /// The model.
    Assistant,
    /// A tool's result.
    Tool,
}

/// One tool call an assistant turn made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall
{
    /// The call's wire identity, which a later tool turn answers.
    pub id: String,
    /// The called function's name.
    pub name: String,
    /// The call's arguments, as the JSON text the model wrote.
    pub arguments: String,
}

/// One turn of the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message
{
    /// Who wrote it.
    pub role: Role,
    /// Its text parts, in order.
    pub parts: Vec<String>,
    /// The reasoning an assistant turn carried; empty for every other role.
    pub reasoning: String,
    /// The calls an assistant turn made; empty for every other role.
    pub tool_calls: Vec<ToolCall>,
    /// The call a tool turn answers; empty for every other role.
    pub tool_call_id: String,
}

/// A chat-template switch the request set or left to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Switch
{
    /// Unset: the template's own default applies.
    ModelDefault,
    /// Set on.
    On,
    /// Set off.
    Off,
}

/// The reasoning effort a request asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effort
{
    /// The request named no effort.
    Unrequested,
    /// No reasoning.
    None,
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning.
    Low,
    /// Medium reasoning.
    Medium,
    /// High reasoning.
    High,
    /// Extra-high reasoning.
    XHigh,
    /// The most reasoning.
    Max,
}

/// Where a prompt-cache marker sits in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheBoundary
{
    /// After this many bytes of the leading system or developer message's
    /// text.
    LeadingInstruction(InstructionBytes),
    /// After this many parts of the message at this one-based position.
    MessagePart
    {
        /// The message's one-based position.
        message: Count,
        /// The parts before the boundary.
        parts: Count,
    },
    /// After this many messages.
    Message(Count),
    /// After this many tool definitions.
    Tool(Count),
}

/// A byte count within the leading instruction.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstructionBytes(pub u32);

/// A count of messages, parts or tools; a count of messages also names the
/// last one's one-based position.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Count(pub u32);

/// Whether a client marked a boundary explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Marked
{
    /// No explicit breakpoint.
    Unmarked,
    /// An explicit breakpoint.
    Explicit,
}

/// Whether a boundary is the request's automatic cache write, and on whose
/// word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Automatic
{
    /// Not the automatic write.
    Not,
    /// The protocol's default automatic write.
    Default,
    /// An automatic write the client asked for.
    Requested,
}

/// A boundary at which the backend may publish the prompt prefix for reuse
/// by later requests, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheMarker
{
    /// Where it sits.
    pub boundary: CacheBoundary,
    /// Whether the client marked it.
    pub marked: Marked,
    /// Whether it is the automatic write.
    pub automatic: Automatic,
}

/// Whether the backend may also choose shared prefixes from the prompt's
/// structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralPrefixes
{
    /// It may.
    Allowed,
    /// Only the markers name shared prefixes, as a protocol with its own
    /// write policy requires.
    Withheld,
}

/// The prompt's cache hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCache
{
    /// The marked boundaries, in the order the backend lowers them.
    pub markers: Vec<CacheMarker>,
    /// Whether the backend may add structural shared prefixes.
    pub structural: StructuralPrefixes,
}

/// What the prompt renders from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt
{
    /// The conversation, in order.
    pub messages: Vec<Message>,
    /// Each offered tool's definition, as a JSON object's text, in order.
    pub tools: Vec<String>,
    /// Extra chat-template arguments, as a JSON object's text.
    pub template_arguments: String,
    /// Whether the assistant turn opens in thinking.
    pub thinking: Switch,
    /// Whether closed turns keep their reasoning in the prompt.
    pub preserve_thinking: Switch,
    /// The reasoning effort asked for.
    pub effort: Effort,
    /// Where the prompt may be published for reuse.
    pub cache: PromptCache,
}

/// A sampling field the request set or left to the model's default for the
/// phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting<Value>
{
    /// Unset: the model's default for the phase applies.
    ModelDefault,
    /// Set to this value.
    Set(Value),
}

/// Sampling overrides for one phase of generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling
{
    /// Softmax temperature; zero is exact argmax.
    pub temperature: Setting<f32>,
    /// Top-k cutoff; zero disables it.
    pub top_k: Setting<i32>,
    /// Nucleus cutoff.
    pub top_p: Setting<f32>,
    /// Minimum probability relative to the most likely token.
    pub min_p: Setting<f32>,
    /// Presence penalty.
    pub presence_penalty: Setting<f32>,
    /// Frequency penalty.
    pub frequency_penalty: Setting<f32>,
}

impl Sampling
{
    /// Every field left to the model.
    pub const MODEL_DEFAULT: Self = Self {
        temperature: Setting::ModelDefault,
        top_k: Setting::ModelDefault,
        top_p: Setting::ModelDefault,
        min_p: Setting::ModelDefault,
        presence_penalty: Setting::ModelDefault,
        frequency_penalty: Setting::ModelDefault,
    };
}

/// The seed a later phase samples with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Seed
{
    /// The request's initial seed.
    Inherited,
    /// This seed.
    Fixed(u64),
}

/// The most model-origin tokens thinking may spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThinkingBudget
{
    /// Thinking runs until the model closes it or the output limit binds.
    Unlimited,
    /// Thinking closes after this many tokens.
    Tokens(core::num::NonZeroU32),
}

/// Which output channels a stop string ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StopScope
{
    /// Content only.
    Content,
    /// Content and reasoning.
    ContentAndReasoning,
}

/// Whether special tokens survive into the rendered output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialTokens
{
    /// Removed.
    Trimmed,
    /// Kept, as tool-call parsing needs.
    Preserved,
}

/// Whether the request may reuse and publish cached prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefixReuse
{
    /// Read and write the prefix cache.
    ReadWrite,
    /// Neither read nor write it.
    Disabled,
}

/// How the backend delivers output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Delivery
{
    /// Only the terminal outcome.
    Aggregate,
    /// Admission and every text delta as it commits, then the outcome.
    Streaming,
}

/// How to generate.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation
{
    /// The most tokens to generate, injected control tokens included.
    pub output_tokens: TokenCount,
    /// Overrides for the initial phase.
    pub sampling: Sampling,
    /// The initial phase's seed.
    pub seed: u64,
    /// Overrides for the phase after thinking closes; unset fields take the
    /// model's post-thinking default.
    pub post_thinking: Sampling,
    /// The post-thinking phase's seed.
    pub post_thinking_seed: Seed,
    /// The thinking budget.
    pub thinking_budget: ThinkingBudget,
    /// Stop strings, in order.
    pub stops: Vec<String>,
    /// Which channels the stop strings end.
    pub stop_scope: StopScope,
    /// Whether special tokens survive into the output.
    pub special_tokens: SpecialTokens,
    /// The longest function name tool-call parsing accepts.
    pub tool_name_limit: core::num::NonZeroU32,
    /// Prefix-cache participation.
    pub prefix_reuse: PrefixReuse,
}

/// One chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest
{
    /// What the prompt renders from.
    pub prompt: Prompt,
    /// How to generate.
    pub generation: Generation,
    /// How output is delivered.
    pub delivery: Delivery,
}
