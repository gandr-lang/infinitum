//! `OpenAI` Chat Completions request bodies, validated and translated into the
//! protocol-neutral [`ChatRequest`].
//!
//! Validation and translation follow ninfer's server field for field, so the
//! same body renders the same prompt and runs the same request on either
//! server, and a body one refuses the other refuses with the same parameter
//! and code.

use infinitum_chat::Automatic;
use infinitum_chat::CacheBoundary;
use infinitum_chat::CacheMarker;
use infinitum_chat::ChatRequest;
use infinitum_chat::Count;
use infinitum_chat::Delivery;
use infinitum_chat::Effort;
use infinitum_chat::Generation;
use infinitum_chat::InstructionBytes;
use infinitum_chat::Marked;
use infinitum_chat::Message;
use infinitum_chat::PrefixReuse;
use infinitum_chat::Prompt;
use infinitum_chat::PromptCache;
use infinitum_chat::Role;
use infinitum_chat::Sampling;
use infinitum_chat::Seed;
use infinitum_chat::Setting;
use infinitum_chat::SpecialTokens;
use infinitum_chat::StopScope;
use infinitum_chat::StructuralPrefixes;
use infinitum_chat::Switch;
use infinitum_chat::ThinkingBudget;
use infinitum_chat::ToolCall;
use infinitum_round::Maybe;
use infinitum_round::TokenCount;
use serde_json::Value;

use crate::error::ApiError;
use crate::error::Code;
use crate::error::Param;
use crate::server::ModelId;

/// A JSON object in wire order.
type Object = serde_json::Map<String, Value>;

/// Why a field has no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unset
{
    /// Absent or `null`.
    Omitted,
}

/// The longest function name the chat surface accepts.
pub const TOOL_NAME_LIMIT: core::num::NonZeroU32 = match core::num::NonZeroU32::new(64) {
    | Some(limit) => limit,
    | None => core::num::NonZeroU32::MIN,
};

/// The most stop strings one request may carry.
const MAXIMUM_STOPS: usize = 4;

/// What a parsed body asks the server for, beside the chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedChat
{
    /// The requested model id.
    pub model: ModelId,
    /// Whether to stream.
    pub delivery: Delivery,
    /// Whether a streamed response ends with a usage chunk.
    pub include_usage: Usage,
    /// The request.
    pub request: ChatRequest,
}

/// Whether a stream carries a final usage chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Usage
{
    /// No usage chunk; the finish chunk carries the timings.
    Omitted,
    /// A usage chunk after the finish chunk.
    Included,
}

/// A fresh seed for a request that names none.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreshSeed(pub u64);

/// Server-side settings a request falls back on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Defaults
{
    /// Output tokens when the request names none.
    pub output_tokens: TokenCount,
    /// The thinking budget of a request that does not turn thinking off.
    pub thinking_budget: ThinkingBudget,
}

/// A JSON member name.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key<'name>(&'name str);

/// An optional boolean field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flag
{
    /// Absent or `null`.
    Unset,
    /// `true`.
    True,
    /// `false`.
    False,
}

/// A JSON number, as a double.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct Number(f64);

/// A JSON integer within `i32`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Integer(i32);

/// A message's position in `messages`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Position(usize);

impl core::fmt::Display for Position
{
    /// Render the index.
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

/// A field's value, or why it has none.
///
/// # Specification
/// - requires: nothing.
/// - ensures: [`Maybe::Present`] exactly when the key is present and not
///   `null`.
/// - provides: the "omitted or null" reading every optional field shares;
///   [`Unset::Omitted`] for both.
/// - fails: never.
/// - panics: none.
fn field<'body>(
    object: &'body Object,
    key: Key<'_>,
) -> Maybe<&'body Value, Unset>
{
    return match object.get(key.0) {
        | Some(&Value::Null) | None => Maybe::Absent(Unset::Omitted),
        | Some(value) => Maybe::Present(value),
    };
}

/// An optional boolean.
///
/// # Specification
/// - requires: nothing.
/// - ensures: [`Flag::True`] or [`Flag::False`] when present; [`Flag::Unset`]
///   when absent or null.
/// - provides: boolean fields.
/// - fails: when present and not a boolean.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: `"<key> must be a boolean"` naming `key`.
fn boolean(
    object: &Object,
    key: Key<'_>,
) -> Result<Flag, ApiError>
{
    return match field(object, key) {
        | Maybe::Absent(Unset::Omitted) => Ok(Flag::Unset),
        | Maybe::Present(&Value::Bool(true)) => Ok(Flag::True),
        | Maybe::Present(&Value::Bool(false)) => Ok(Flag::False),
        | Maybe::Present(_) => Err(ApiError::invalid(
            format!("{} must be a boolean", key.0),
            Param(key.0),
            Code::NONE,
        )),
    };
}

/// An optional template switch.
///
/// # Specification
/// - requires: nothing.
/// - ensures: as [`boolean`], with ninfer's message for the switches it reads
///   as optional.
/// - provides: `enable_thinking` and `preserve_thinking`, typed and nested.
/// - fails: when present and not a boolean.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: `"<key> must be a boolean or null"` naming `key`.
fn nullable_boolean(
    object: &Object,
    key: Key<'_>,
) -> Result<Flag, ApiError>
{
    return boolean(object, key).map_err(|_mistyped| {
        return ApiError::invalid(
            format!("{} must be a boolean or null", key.0),
            Param(key.0),
            Code::NONE,
        );
    });
}

/// An optional number.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the value when present; omitted when absent or null.
/// - provides: sampling fields.
/// - fails: when present and not a number.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: `"<key> must be a number"` naming `key`.
fn number(
    object: &Object,
    key: Key<'_>,
) -> Result<Maybe<Number, Unset>, ApiError>
{
    return match field(object, key) {
        | Maybe::Absent(reason) => Ok(Maybe::Absent(reason)),
        | Maybe::Present(value) => match value.as_f64() {
            | Some(number) => Ok(Maybe::Present(Number(number))),
            | None => Err(ApiError::invalid(
                format!("{} must be a number", key.0),
                Param(key.0),
                Code::NONE,
            )),
        },
    };
}

/// An optional 32-bit integer.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the value when present; omitted when absent or null.
/// - provides: integer fields.
/// - fails: when present and not an integer, or outside `i32`.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: `"<key> must be an integer"` or `"<key> is out of range"`,
///   naming `key`.
///
/// # Adequacy
/// - hypothesis: L3 at the `i32` boundary and the integer check.
/// - witness: `tests::integers_are_checked_and_bounded`
fn integer(
    object: &Object,
    key: Key<'_>,
) -> Result<Maybe<Integer, Unset>, ApiError>
{
    let Maybe::Present(value) = field(object, key)
    else {
        return Ok(Maybe::Absent(Unset::Omitted));
    };
    let out_of_range = || {
        return ApiError::invalid(
            format!("{} is out of range", key.0),
            Param(key.0),
            Code::NONE,
        );
    };
    if let Some(signed) = value.as_i64() {
        return i32::try_from(signed)
            .map(|narrow| return Maybe::Present(Integer(narrow)))
            .map_err(|_overflow| return out_of_range());
    }
    if value.is_u64() {
        return Err(out_of_range());
    }
    return Err(ApiError::invalid(
        format!("{} must be an integer", key.0),
        Param(key.0),
        Code::NONE,
    ));
}

/// An optional seed: any integer, a negative one taken as its two's
/// complement bits.
///
/// # Specification
/// - requires: nothing.
/// - ensures: [`Seed::Fixed`] with the seed's bits when present;
///   [`Seed::Inherited`] when absent or null.
/// - provides: seed fields.
/// - fails: when present and not an integer.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: `"<param> must be an integer"` naming `param`.
fn seed(
    object: &Object,
    param: Param<'_>,
) -> Result<Seed, ApiError>
{
    let Maybe::Present(value) = field(object, Key("seed"))
    else {
        return Ok(Seed::Inherited);
    };
    if let Some(unsigned) = value.as_u64() {
        return Ok(Seed::Fixed(unsigned));
    }
    if let Some(signed) = value.as_i64() {
        return Ok(Seed::Fixed(signed.cast_unsigned()));
    }
    return Err(ApiError::invalid(
        format!("{} must be an integer", param.0),
        param,
        Code::NONE,
    ));
}

/// A function object's validated name.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the name when it is a string matching `[A-Za-z0-9_-]{1,64}`.
/// - provides: tool, tool-call and tool-choice names.
/// - fails: otherwise.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `param`.
fn function_name(
    object: &Object,
    param: Param<'_>,
) -> Result<String, ApiError>
{
    let Some(name) = object.get("name").and_then(Value::as_str)
    else {
        return Err(ApiError::invalid(
            String::from("function name must be a string"),
            param,
            Code::NONE,
        ));
    };
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| return byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if !valid {
        return Err(ApiError::invalid(
            String::from("function name must match [A-Za-z0-9_-]{1,64}"),
            param,
            Code::NONE,
        ));
    }
    return Ok(String::from(name));
}

/// Refuse output controls the Engine cannot honour, accepting their neutral
/// values.
///
/// # Specification
/// - requires: nothing.
/// - ensures: nothing on success.
/// - provides: the refusals ninfer's server makes for `functions`,
///   `function_call`, `logit_bias`, `logprobs`, `top_logprobs`,
///   `response_format`, `modalities`, `web_search_options`, `moderation`,
///   `verbosity`, `store`, constrained-decoding extensions,
///   `repetition_penalty`, `mm_processor_kwargs`, and the live-observation
///   extensions this surface does not publish.
/// - fails: on the first non-neutral control, with ninfer's parameter and code.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: the refusal.
fn refuse_unsupported(body: &Object) -> Result<(), ApiError>
{
    if let Maybe::Present(functions) = field(body, Key("functions")) {
        let Some(list) = functions.as_array()
        else {
            return Err(ApiError::invalid(
                String::from("functions must be an array"),
                Param("functions"),
                Code::NONE,
            ));
        };
        if !list.is_empty() {
            return Err(ApiError::invalid(
                String::from("non-empty legacy functions are not supported; use tools instead"),
                Param("functions"),
                Code("legacy_tools_not_supported"),
            ));
        }
    }
    if let Maybe::Present(choice) = field(body, Key("function_call")) {
        match *choice {
            | Value::String(ref value) if value == "none" || value == "auto" => {},
            | Value::Object(_) => {
                return Err(ApiError::invalid(
                    String::from("a named legacy function_call requires forced tool invocation"),
                    Param("function_call"),
                    Code("legacy_tools_not_supported"),
                ));
            },
            | _ => {
                return Err(ApiError::invalid(
                    String::from("function_call must be 'none', 'auto', or an object"),
                    Param("function_call"),
                    Code::NONE,
                ));
            },
        }
    }
    if let Maybe::Present(biases) = field(body, Key("logit_bias")) {
        let Some(map) = biases.as_object()
        else {
            return Err(ApiError::invalid(
                String::from("logit_bias must be an object"),
                Param("logit_bias"),
                Code::NONE,
            ));
        };
        for bias in map.values() {
            let Some(value) = bias.as_f64()
            else {
                return Err(ApiError::invalid(
                    String::from("logit_bias values must be numbers"),
                    Param("logit_bias"),
                    Code::NONE,
                ));
            };
            if value != 0.0_f64 {
                return Err(ApiError::invalid(
                    String::from("nonzero logit_bias requires per-token logit modification"),
                    Param("logit_bias"),
                    Code("logit_bias_not_supported"),
                ));
            }
        }
    }
    if boolean(body, Key("logprobs"))? == Flag::True {
        return Err(ApiError::invalid(
            String::from("logprobs=true requires per-token log probabilities"),
            Param("logprobs"),
            Code("logprobs_not_supported"),
        ));
    }
    if let Maybe::Present(count) = integer(body, Key("top_logprobs"))?
        && count != Integer(0_i32)
    {
        return Err(ApiError::invalid(
            String::from("nonzero top_logprobs requires alternative-token probabilities"),
            Param("top_logprobs"),
            Code("logprobs_not_supported"),
        ));
    }
    if let Maybe::Present(format) = field(body, Key("response_format")) {
        let kind = format.get("type").and_then(Value::as_str);
        match kind {
            | Some("text") => {},
            | Some(_) => {
                return Err(ApiError::invalid(
                    String::from("only {\"type\":\"text\"} response_format is available"),
                    Param("response_format"),
                    Code("response_format_not_supported"),
                ));
            },
            | None => {
                return Err(ApiError::invalid(
                    String::from("response_format must contain a string type"),
                    Param("response_format"),
                    Code::NONE,
                ));
            },
        }
    }
    if let Maybe::Present(modalities) = field(body, Key("modalities")) {
        let Some(list) = modalities.as_array().filter(|list| return !list.is_empty())
        else {
            return Err(ApiError::invalid(
                String::from("modalities must be a non-empty array"),
                Param("modalities"),
                Code::NONE,
            ));
        };
        for modality in list {
            match modality.as_str() {
                | Some("text") => {},
                | Some(_) => {
                    return Err(ApiError::invalid(
                        String::from("only text output is produced"),
                        Param("modalities"),
                        Code("modality_not_supported"),
                    ));
                },
                | None => {
                    return Err(ApiError::invalid(
                        String::from("modalities entries must be strings"),
                        Param("modalities"),
                        Code::NONE,
                    ));
                },
            }
        }
    }
    for (key, code) in [
        ("web_search_options", "web_search_not_supported"),
        ("moderation", "moderation_not_supported"),
    ] {
        if let Maybe::Present(_) = field(body, Key(key)) {
            return Err(ApiError::invalid(
                format!("{key} is not provided"),
                Param(key),
                Code(code),
            ));
        }
    }
    if let Maybe::Present(verbosity) = field(body, Key("verbosity")) {
        match verbosity.as_str() {
            | Some("medium") => {},
            | Some("low" | "high") => {
                return Err(ApiError::invalid(
                    String::from("only the default verbosity 'medium' is accepted"),
                    Param("verbosity"),
                    Code("verbosity_not_supported"),
                ));
            },
            | _ => {
                return Err(ApiError::invalid(
                    String::from("verbosity must be 'low', 'medium', or 'high'"),
                    Param("verbosity"),
                    Code::NONE,
                ));
            },
        }
    }
    if boolean(body, Key("store"))? == Flag::True {
        return Err(ApiError::invalid(
            String::from("store=true requires a retrievable stored Chat Completion"),
            Param("store"),
            Code("store_not_supported"),
        ));
    }
    for key in [
        "grammar",
        "structured_outputs",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
    ] {
        match field(body, Key(key)) {
            | Maybe::Absent(_) => {},
            | Maybe::Present(&Value::String(ref text)) if key == "grammar" && text.is_empty() => {},
            | Maybe::Present(_) => {
                return Err(ApiError::invalid(
                    format!("{key} requests constrained decoding"),
                    Param(key),
                    Code("constrained_decoding_not_supported"),
                ));
            },
        }
    }
    if let Maybe::Present(penalty) = number(body, Key("repetition_penalty"))?
        && (penalty.0 - 1.0_f64).abs() > 0.0_f64
    {
        return Err(ApiError::invalid(
            String::from("only repetition_penalty=1 is accepted"),
            Param("repetition_penalty"),
            Code("repetition_penalty_not_supported"),
        ));
    }
    if let Maybe::Present(kwargs) = field(body, Key("mm_processor_kwargs")) {
        let Some(map) = kwargs.as_object()
        else {
            return Err(ApiError::invalid(
                String::from("mm_processor_kwargs must be an object"),
                Param("mm_processor_kwargs"),
                Code::NONE,
            ));
        };
        if map.values().any(|value| return !value.is_null()) {
            return Err(ApiError::invalid(
                String::from(
                    "mm_processor_kwargs overrides preprocessing, which vision-disabled serving cannot apply",
                ),
                Param("mm_processor_kwargs"),
                Code("mm_processor_kwargs_not_supported"),
            ));
        }
    }
    for key in ["timings_per_token", "return_progress"] {
        if boolean(body, Key(key))? == Flag::True {
            return Err(ApiError::invalid(
                format!("{key} live observations are not published"),
                Param(key),
                Code("observation_not_supported"),
            ));
        }
    }
    return Ok(());
}

/// Validate the prompt-cache hints and read the automatic-write policy.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success, the policy `prompt_cache_options` names: absent, the
///   default automatic write; `mode` `explicit`, none; otherwise the requested
///   automatic write.
/// - provides: ninfer's checks on `prompt_cache_key`, `safety_identifier`,
///   `user`, `prompt_cache_retention` and `prompt_cache_options`.
/// - fails: on a malformed hint.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming the hint.
fn cache_policy(body: &Object) -> Result<AutomaticWrite, ApiError>
{
    for (key, limit) in [
        ("prompt_cache_key", 64_usize),
        ("safety_identifier", 64),
        ("user", usize::MAX),
    ] {
        if let Maybe::Present(value) = field(body, Key(key)) {
            let Some(text) = value.as_str()
            else {
                return Err(ApiError::invalid(
                    format!("{key} must be a string"),
                    Param(key),
                    Code::NONE,
                ));
            };
            if text.len() > limit {
                return Err(ApiError::invalid(
                    format!("{key} is too long"),
                    Param(key),
                    Code::NONE,
                ));
            }
        }
    }
    if let Maybe::Present(retention) = field(body, Key("prompt_cache_retention")) {
        match retention.as_str() {
            | Some("in_memory" | "24h") => {},
            | Some(_) => {
                return Err(ApiError::invalid(
                    String::from("prompt_cache_retention must be 'in_memory' or '24h'"),
                    Param("prompt_cache_retention"),
                    Code::NONE,
                ));
            },
            | None => {
                return Err(ApiError::invalid(
                    String::from("prompt_cache_retention must be a string"),
                    Param("prompt_cache_retention"),
                    Code::NONE,
                ));
            },
        }
    }
    if let Maybe::Present(options) = field(body, Key("prompt_cache_options")) {
        let Some(map) = options.as_object()
        else {
            return Err(ApiError::invalid(
                String::from("prompt_cache_options must be an object"),
                Param("prompt_cache_options"),
                Code::NONE,
            ));
        };
        if let Maybe::Present(mode) = field(map, Key("mode"))
            && !matches!(mode.as_str(), Some("implicit" | "explicit"))
        {
            return Err(ApiError::invalid(
                String::from("prompt_cache_options.mode must be 'implicit' or 'explicit'"),
                Param("prompt_cache_options"),
                Code::NONE,
            ));
        }
        if let Maybe::Present(ttl) = field(map, Key("ttl"))
            && ttl.as_str() != Some("30m")
        {
            return Err(ApiError::invalid(
                String::from("prompt_cache_options.ttl must be '30m'"),
                Param("prompt_cache_options"),
                Code::NONE,
            ));
        }
        if let Maybe::Present(mode) = field(map, Key("mode"))
            && mode.as_str() == Some("explicit")
        {
            return Ok(AutomaticWrite::Disabled);
        }
        return Ok(AutomaticWrite::Requested);
    }
    return Ok(AutomaticWrite::Default);
}

/// Which automatic cache write a body asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticWrite
{
    /// None: only explicit breakpoints write.
    Disabled,
    /// The protocol's default, absent `prompt_cache_options`.
    Default,
    /// The one `prompt_cache_options` asks for.
    Requested,
}

/// A translated turn with the explicit breakpoint after each of its parts.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Turn
{
    /// The turn.
    message: Message,
    /// Whether each part, in order, carries an explicit breakpoint.
    marks: Vec<Marked>,
}

/// A cache slot as ninfer's policy sees it: empty, or a boundary with its
/// evidence.
type Slot = Option<(Marked, Automatic)>;

/// The most explicit cache writes one request keeps, ninfer's
/// `kMaximumExplicitPromptCacheMarkers`.
const EXPLICIT_WRITES: usize = 4;

/// The explicit cache writes kept beside an automatic write that needs its
/// own slot: one fewer than [`EXPLICIT_WRITES`].
const EXPLICIT_WRITES_BESIDE_AUTOMATIC: usize = 3;

/// A count of messages, parts, tools or bytes, before it is narrowed to a
/// marker's `u32`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tally(usize);

impl Tally
{
    /// Narrow the count to the marker's `u32`, as ninfer's lowering does.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success, the count unchanged.
    /// - provides: every marker count.
    /// - fails: when the count exceeds `u32::MAX`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ApiError`]: ninfer's overflow message, naming `messages`.
    fn narrow(self) -> Result<Count, ApiError>
    {
        return u32::try_from(self.0).map(Count).map_err(|_overflow| {
            return ApiError::invalid(
                String::from("conversation cache boundary exceeds uint32"),
                Param("messages"),
                Code::NONE,
            );
        });
    }
}

/// The prompt's cache markers, as ninfer's server derives them for a Chat
/// Completions body.
///
/// # Specification
/// - requires: `tools` are the tool definitions the prompt offers.
/// - ensures: the automatic write targets the last turn, from the end, with
///   tool calls (its boundary) or parts (its last part), else the last offered
///   tool; it is enabled unless `policy` is [`AutomaticWrite::Disabled`] or
///   there is no target. Of the explicit breakpoints only the last four are
///   kept, three when an enabled automatic write needs a slot of its own. The
///   automatic write joins an explicit breakpoint at its target or stands
///   alone. Markers are listed per turn, parts before the turn's own boundary,
///   then per tool; a part of a leading system or developer turn is placed by
///   its cumulative text bytes, any other part by its turn's one-based position
///   and its part count. Structural shared prefixes are withheld, since the
///   protocol has its own write policy.
/// - provides: the Chat Completions cache hints.
/// - fails: when a count or byte offset exceeds `u32`.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: ninfer's overflow message, naming `messages`.
///
/// # Adequacy
/// - hypothesis: L3 — the default write on the last part, an assistant turn's
///   tool calls taking the boundary, the explicit-mode switch-off, the
///   four-write cap with and without a merged target, and leading-instruction
///   byte placement.
/// - witness: `tests::cache_markers_follow_ninfers_policy`
fn prompt_cache(
    turns: &[Turn],
    tools: &[String],
    policy: AutomaticWrite,
) -> Result<PromptCache, ApiError>
{
    let mut slots: Vec<TurnSlots> = turns
        .iter()
        .map(|turn| {
            let parts = turn
                .marks
                .iter()
                .map(|mark| {
                    return match *mark {
                        | Marked::Explicit => Some((Marked::Explicit, Automatic::Not)),
                        | Marked::Unmarked => None,
                    };
                })
                .collect();
            return TurnSlots {
                parts,
                boundary: None,
            };
        })
        .collect();
    let mut last_tool: Slot = None;
    let target = turns
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, turn)| {
            if !turn.message.tool_calls.is_empty() {
                return Some(Target::Turn(index));
            }
            return turn
                .message
                .parts
                .len()
                .checked_sub(1)
                .map(|last| return Target::Part(index, last));
        })
        .or_else(|| return (!tools.is_empty()).then_some(Target::Tool));
    let automatic = match policy {
        | AutomaticWrite::Disabled => Automatic::Not,
        | AutomaticWrite::Default => Automatic::Default,
        | AutomaticWrite::Requested => Automatic::Requested,
    };
    let enabled = automatic != Automatic::Not && target.is_some();
    let merges = enabled
        && match target {
            | Some(Target::Part(turn, part)) => slots
                .get(turn)
                .and_then(|turn| return turn.parts.get(part))
                .is_some_and(Option::is_some),
            | Some(Target::Turn(_) | Target::Tool) | None => false,
        };
    let keep = if !enabled || merges {
        EXPLICIT_WRITES
    }
    else {
        EXPLICIT_WRITES_BESIDE_AUTOMATIC
    };
    let explicit = slots
        .iter()
        .flat_map(|turn| return turn.parts.iter())
        .filter(|slot| return slot.is_some())
        .count();
    for slot in slots
        .iter_mut()
        .flat_map(|turn| return turn.parts.iter_mut())
        .filter(|slot| return slot.is_some())
        .take(explicit.saturating_sub(keep))
    {
        *slot = None;
    }
    if enabled {
        let slot = match target {
            | Some(Target::Part(turn, part)) => slots
                .get_mut(turn)
                .and_then(|turn| return turn.parts.get_mut(part)),
            | Some(Target::Turn(turn)) => slots.get_mut(turn).map(|turn| return &mut turn.boundary),
            | Some(Target::Tool) => Some(&mut last_tool),
            | None => None,
        };
        if let Some(slot) = slot {
            *slot = Some(slot.map_or((Marked::Unmarked, automatic), |(marked, _)| {
                return (marked, automatic);
            }));
        }
    }
    let mut markers = Vec::new();
    for (index, (turn, turn_slots)) in turns.iter().zip(&slots).enumerate() {
        let leading = index == 0 && matches!(turn.message.role, Role::System | Role::Developer);
        let message = Tally(index.saturating_add(1)).narrow()?;
        let mut bytes = 0_usize;
        for (part, (text, slot)) in turn.message.parts.iter().zip(&turn_slots.parts).enumerate() {
            bytes = bytes.saturating_add(text.len());
            if let Some((marked, automatic)) = *slot {
                let boundary = if leading {
                    CacheBoundary::LeadingInstruction(InstructionBytes(Tally(bytes).narrow()?.0))
                }
                else {
                    CacheBoundary::MessagePart {
                        message,
                        parts: Tally(part.saturating_add(1)).narrow()?,
                    }
                };
                markers.push(CacheMarker {
                    boundary,
                    marked,
                    automatic,
                });
            }
        }
        if let Some((marked, automatic)) = turn_slots.boundary {
            markers.push(CacheMarker {
                boundary: CacheBoundary::Message(message),
                marked,
                automatic,
            });
        }
    }
    if let Some((marked, automatic)) = last_tool {
        markers.push(CacheMarker {
            boundary: CacheBoundary::Tool(Tally(tools.len()).narrow()?),
            marked,
            automatic,
        });
    }
    return Ok(PromptCache {
        markers,
        structural: StructuralPrefixes::Withheld,
    });
}

/// One turn's cache slots: one per part, and the turn's own boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TurnSlots
{
    /// After each part, in order.
    parts: Vec<Slot>,
    /// After the turn.
    boundary: Slot,
}

/// Where a body's automatic cache write lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target
{
    /// After this part of this turn.
    Part(usize, usize),
    /// After this turn, whose tool calls end it.
    Turn(usize),
    /// After the last offered tool.
    Tool,
}

/// A content part's cache breakpoint.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success, whether the part carries an explicit breakpoint.
/// - provides: ninfer's check and reading of `prompt_cache_breakpoint`.
/// - fails: when present and not `{"mode":"explicit"}`.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: code `invalid_cache_breakpoint`.
fn breakpoint(part: &Object) -> Result<Marked, ApiError>
{
    let Maybe::Present(breakpoint) = field(part, Key("prompt_cache_breakpoint"))
    else {
        return Ok(Marked::Unmarked);
    };
    if breakpoint.get("mode").and_then(Value::as_str) != Some("explicit") {
        return Err(ApiError::invalid(
            String::from("prompt_cache_breakpoint must be {mode:'explicit'}"),
            Param("messages"),
            Code("invalid_cache_breakpoint"),
        ));
    }
    return Ok(Marked::Explicit);
}

/// A message's content as text parts, each with its breakpoint.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a string is one unmarked part; an array yields its `text` parts
///   and, on assistant turns, its `refusal` parts, in order, each marked as its
///   `prompt_cache_breakpoint` says.
/// - provides: every turn's parts.
/// - fails: on a malformed part, a refusal off an assistant turn, or a media
///   part, which this server refuses because vision is disabled.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `messages`.
fn content_parts(
    content: &Value,
    role: Role,
    index: Position,
) -> Result<Vec<(String, Marked)>, ApiError>
{
    if let Value::String(ref text) = *content {
        return Ok(vec![(text.clone(), Marked::Unmarked)]);
    }
    let Some(parts) = content.as_array()
    else {
        return Err(ApiError::invalid(
            format!("message {index} content must be a string or array"),
            Param("messages"),
            Code::NONE,
        ));
    };
    let mut texts = Vec::with_capacity(parts.len());
    for part in parts {
        let Some(object) = part.as_object()
        else {
            return Err(ApiError::invalid(
                String::from("message content parts must contain a string type"),
                Param("messages"),
                Code::NONE,
            ));
        };
        let Some(kind) = object.get("type").and_then(Value::as_str)
        else {
            return Err(ApiError::invalid(
                String::from("message content parts must contain a string type"),
                Param("messages"),
                Code::NONE,
            ));
        };
        let text = match kind {
            | "text" => object.get("text").and_then(Value::as_str).ok_or_else(|| {
                return ApiError::invalid(
                    String::from("text content part must contain a string text"),
                    Param("messages"),
                    Code::NONE,
                );
            })?,
            | "refusal" => {
                if role != Role::Assistant {
                    return Err(ApiError::invalid(
                        String::from("refusal content is only valid on assistant messages"),
                        Param("messages"),
                        Code::NONE,
                    ));
                }
                object
                    .get("refusal")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        return ApiError::invalid(
                            String::from("refusal content part must contain a string refusal"),
                            Param("messages"),
                            Code::NONE,
                        );
                    })?
            },
            | "image_url" | "video_url" => {
                return Err(ApiError::invalid(
                    String::from("Vision is disabled for this server"),
                    Param("messages"),
                    Code("vision_disabled"),
                ));
            },
            | other => {
                return Err(ApiError::invalid(
                    format!("content type '{other}' is not supported"),
                    Param("messages"),
                    Code("modality_not_supported"),
                ));
            },
        };
        let marked = breakpoint(object)?;
        texts.push((String::from(text), marked));
    }
    return Ok(texts);
}

/// A turn's role.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the five `OpenAI` roles map to their namesakes, and the legacy
///   `function` role to [`Role::Tool`].
/// - provides: every turn's role.
/// - fails: on any other role.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: code `unsupported_role`.
fn role_of(name: RoleName<'_>) -> Result<Role, ApiError>
{
    return match name.0 {
        | "system" => Ok(Role::System),
        | "developer" => Ok(Role::Developer),
        | "user" => Ok(Role::User),
        | "assistant" => Ok(Role::Assistant),
        | "tool" | "function" => Ok(Role::Tool),
        | other => Err(ApiError::invalid(
            format!("unsupported role: {other}"),
            Param("messages"),
            Code("unsupported_role"),
        )),
    };
}

/// A turn's `role` string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoleName<'name>(&'name str);

/// A string field that must be absent, null, or empty.
///
/// # Specification
/// - requires: nothing.
/// - ensures: nothing on success.
/// - provides: the fields a role may not carry.
/// - fails: when present and not a string (`"<key> must be a string"`), or
///   non-empty (`message`).
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `messages`.
fn require_empty(
    item: &Object,
    key: Key<'_>,
    message: String,
) -> Result<(), ApiError>
{
    let Key(key) = key;
    if let Maybe::Present(value) = field(item, Key(key)) {
        let Some(text) = value.as_str()
        else {
            return Err(ApiError::invalid(
                format!("{key} must be a string"),
                Param("messages"),
                Code::NONE,
            ));
        };
        if !text.is_empty() {
            return Err(ApiError::invalid(message, Param("messages"), Code::NONE));
        }
    }
    return Ok(());
}

/// An assistant turn's tool calls, the legacy `function_call` first.
///
/// # Specification
/// - requires: nothing.
/// - ensures: every `tool_calls` entry becomes a call with its id, name and
///   argument text; a legacy `function_call` becomes a first call with an empty
///   id.
/// - provides: assistant tool-call history.
/// - fails: on a malformed call.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `messages`.
fn assistant_calls(
    item: &Object,
    index: Position,
) -> Result<Vec<ToolCall>, ApiError>
{
    let mut calls = Vec::new();
    if let Maybe::Present(call) = field(item, Key("function_call")) {
        let Some(object) = call.as_object()
        else {
            return Err(ApiError::invalid(
                format!("assistant message {index} function_call must be an object"),
                Param("messages"),
                Code::NONE,
            ));
        };
        let Some(arguments) = object.get("arguments").and_then(Value::as_str)
        else {
            return Err(ApiError::invalid(
                String::from("assistant function_call must contain string arguments"),
                Param("messages"),
                Code::NONE,
            ));
        };
        calls.push(ToolCall {
            id: String::new(),
            name: function_name(object, Param("messages"))?,
            arguments: String::from(arguments),
        });
    }
    let Maybe::Present(values) = field(item, Key("tool_calls"))
    else {
        return Ok(calls);
    };
    let Some(list) = values.as_array()
    else {
        return Err(ApiError::invalid(
            format!("assistant message {index} tool_calls must be an array"),
            Param("messages"),
            Code::NONE,
        ));
    };
    for value in list {
        let Some(id) = value.get("id").and_then(Value::as_str)
        else {
            return Err(ApiError::invalid(
                String::from("tool_calls entries must contain a string id"),
                Param("messages"),
                Code::NONE,
            ));
        };
        if value.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::invalid(
                String::from("only function tool_calls are supported"),
                Param("messages"),
                Code("tool_type_not_supported"),
            ));
        }
        let Some(function) = value.get("function").and_then(Value::as_object)
        else {
            return Err(ApiError::invalid(
                String::from("tool_calls entries must contain a function object"),
                Param("messages"),
                Code::NONE,
            ));
        };
        let Some(arguments) = function.get("arguments").and_then(Value::as_str)
        else {
            return Err(ApiError::invalid(
                String::from("function tool_calls must contain string arguments"),
                Param("messages"),
                Code::NONE,
            ));
        };
        calls.push(ToolCall {
            id: String::from(id),
            name: function_name(function, Param("messages"))?,
            arguments: String::from(arguments),
        });
    }
    return Ok(calls);
}

/// An assistant turn's carried reasoning, from `reasoning_content` or its
/// alias `reasoning`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the non-empty value of either field, empty when neither has one.
/// - provides: assistant reasoning history.
/// - fails: when a field is not a string, or both are non-empty and differ.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `messages`.
fn assistant_reasoning(
    item: &Object,
    index: Position,
) -> Result<String, ApiError>
{
    let mut found = Vec::with_capacity(2);
    for key in ["reasoning_content", "reasoning"] {
        if let Maybe::Present(value) = field(item, Key(key)) {
            let Some(text) = value.as_str()
            else {
                return Err(ApiError::invalid(
                    format!("assistant message {index} {key} must be a string"),
                    Param("messages"),
                    Code::NONE,
                ));
            };
            if !text.is_empty() {
                found.push(text);
            }
        }
    }
    return match *found.as_slice() {
        | [] => Ok(String::new()),
        | [one] => Ok(String::from(one)),
        | [first, second, ..] if first == second => Ok(String::from(first)),
        | _ => Err(ApiError::invalid(
            String::from("conflicting assistant reasoning and reasoning_content values"),
            Param("messages"),
            Code("conflicting_template_option"),
        )),
    };
}

/// One turn.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the turn carries its role, parts, and the reasoning, tool calls
///   or tool-call id its role admits, with each part's explicit breakpoint
///   beside it.
/// - provides: the conversation.
/// - fails: on any check ninfer's server makes on a message.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `messages`.
fn message(
    value: &Value,
    index: Position,
) -> Result<Turn, ApiError>
{
    let (Some(item), Some(role_name)) =
        (value.as_object(), value.get("role").and_then(Value::as_str))
    else {
        return Err(ApiError::invalid(
            format!("message {index} must be an object with a string role"),
            Param("messages"),
            Code::NONE,
        ));
    };
    let role = role_of(RoleName(role_name))?;
    let legacy_function = role_name == "function";
    if let Maybe::Present(name) = field(item, Key("name")) {
        let Some(text) = name.as_str()
        else {
            return Err(ApiError::invalid(
                String::from("message name must be a string"),
                Param("messages"),
                Code::NONE,
            ));
        };
        if !text.is_empty() && role != Role::Tool {
            return Err(ApiError::invalid(
                String::from("a non-empty message name changes participant identity"),
                Param("messages"),
                Code("message_name_not_supported"),
            ));
        }
    }
    if legacy_function {
        function_name(item, Param("messages"))?;
    }
    if role != Role::Assistant {
        if let Maybe::Present(_) = field(item, Key("function_call")) {
            return Err(ApiError::invalid(
                String::from("function_call is only valid on assistant messages"),
                Param("messages"),
                Code::NONE,
            ));
        }
        for key in ["reasoning", "reasoning_content"] {
            require_empty(
                item,
                Key(key),
                format!("{key} is only valid on assistant messages"),
            )?;
        }
    }
    let mut turn = Message {
        role,
        parts: Vec::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        tool_call_id: String::new(),
    };
    if role == Role::Tool {
        if let Maybe::Present(calls) = field(item, Key("tool_calls"))
            && calls.as_array().is_none_or(|list| return !list.is_empty())
        {
            return Err(ApiError::invalid(
                String::from("tool messages cannot contain tool_calls"),
                Param("messages"),
                Code::NONE,
            ));
        }
        match field(item, Key("tool_call_id")) {
            | Maybe::Present(&Value::String(ref id)) => turn.tool_call_id.clone_from(id),
            | Maybe::Present(_) => {
                return Err(ApiError::invalid(
                    String::from("tool_call_id must be a string"),
                    Param("messages"),
                    Code::NONE,
                ));
            },
            | Maybe::Absent(_) if legacy_function => {},
            | Maybe::Absent(_) => {
                return Err(ApiError::invalid(
                    String::from("tool messages must contain a string tool_call_id"),
                    Param("messages"),
                    Code::NONE,
                ));
            },
        }
        let Maybe::Present(content) = field(item, Key("content"))
        else {
            return Err(ApiError::invalid(
                String::from("tool messages must contain content"),
                Param("messages"),
                Code::NONE,
            ));
        };
        return Ok(with_parts(turn, content_parts(content, role, index)?));
    }
    require_empty(
        item,
        Key("tool_call_id"),
        String::from("a non-empty tool_call_id is only valid on tool messages"),
    )?;
    if role == Role::Assistant {
        if let Maybe::Present(_) = field(item, Key("audio")) {
            return Err(ApiError::invalid(
                String::from("assistant audio history is not supported"),
                Param("messages"),
                Code("assistant_history_not_supported"),
            ));
        }
        turn.tool_calls = assistant_calls(item, index)?;
        turn.reasoning = assistant_reasoning(item, index)?;
        let mut parts = match field(item, Key("content")) {
            | Maybe::Present(content) => content_parts(content, role, index)?,
            | Maybe::Absent(_) => Vec::new(),
        };
        if let Maybe::Present(refusal) = field(item, Key("refusal")) {
            let Some(text) = refusal.as_str()
            else {
                return Err(ApiError::invalid(
                    String::from("assistant refusal must be a string"),
                    Param("messages"),
                    Code::NONE,
                ));
            };
            if !text.is_empty() {
                parts.push((String::from(text), Marked::Unmarked));
            }
        }
        return Ok(with_parts(turn, parts));
    }
    if let Maybe::Present(calls) = field(item, Key("tool_calls"))
        && calls.as_array().is_none_or(|list| return !list.is_empty())
    {
        return Err(ApiError::invalid(
            String::from("non-empty tool_calls are only valid on assistant messages"),
            Param("messages"),
            Code::NONE,
        ));
    }
    let Maybe::Present(content) = field(item, Key("content"))
    else {
        return Err(ApiError::invalid(
            format!("message {index} must have content"),
            Param("messages"),
            Code::NONE,
        ));
    };
    return Ok(with_parts(turn, content_parts(content, role, index)?));
}

/// A turn given its parts, with their breakpoints kept beside it.
///
/// # Specification
/// trivial.
fn with_parts(
    mut message: Message,
    parts: Vec<(String, Marked)>,
) -> Turn
{
    let (texts, marks) = parts.into_iter().unzip();
    message.parts = texts;
    return Turn { message, marks };
}

/// One declared tool: its name and rendered definition.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Tool
{
    /// The function name.
    name: String,
    /// The definition as the chat template receives it.
    definition: String,
}

/// The declared tools, each rendered as
/// `{"type":"function","function":{"name","parameters","strict":false,"
/// description"}}`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: every tool keeps its name, its parameter schema in wire order (an
///   empty object schema when absent), `strict` false, and its description when
///   given, in that key order.
/// - provides: the prompt's tool definitions.
/// - fails: on a malformed tool or `strict: true`.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `tools`.
///
/// # Adequacy
/// - hypothesis: L3 — key order with and without a description, and the default
///   schema.
/// - witness: `tests::tools_render_in_ninfer_key_order`
fn tools(body: &Object) -> Result<Vec<Tool>, ApiError>
{
    let Maybe::Present(value) = field(body, Key("tools"))
    else {
        return Ok(Vec::new());
    };
    let Some(list) = value.as_array()
    else {
        return Err(ApiError::invalid(
            String::from("tools must be an array"),
            Param("tools"),
            Code::NONE,
        ));
    };
    let mut declared = Vec::with_capacity(list.len());
    for item in list {
        match item.get("type").and_then(Value::as_str) {
            | Some("function") => {},
            | Some(other) => {
                return Err(ApiError::invalid(
                    format!("tool type '{other}' requires a non-function output contract"),
                    Param("tools"),
                    Code("tool_type_not_supported"),
                ));
            },
            | None => {
                return Err(ApiError::invalid(
                    String::from("tools entries must contain a string type"),
                    Param("tools"),
                    Code::NONE,
                ));
            },
        }
        let Some(function) = item.get("function").and_then(Value::as_object)
        else {
            return Err(ApiError::invalid(
                String::from("function tools must contain a function object"),
                Param("tools"),
                Code::NONE,
            ));
        };
        let name = function_name(function, Param("tools"))?;
        let description = match field(function, Key("description")) {
            | Maybe::Absent(_) => String::new(),
            | Maybe::Present(&Value::String(ref text)) => text.clone(),
            | Maybe::Present(_) => {
                return Err(ApiError::invalid(
                    String::from("function description must be a string"),
                    Param("tools"),
                    Code::NONE,
                ));
            },
        };
        let parameters = match field(function, Key("parameters")) {
            | Maybe::Absent(_) => {
                let mut schema = Object::new();
                schema.insert(String::from("type"), Value::from("object"));
                schema.insert(String::from("properties"), Value::Object(Object::new()));
                Value::Object(schema)
            },
            | Maybe::Present(schema) if schema.is_object() => schema.clone(),
            | Maybe::Present(_) => {
                return Err(ApiError::invalid(
                    String::from("function parameters must be a JSON object"),
                    Param("tools"),
                    Code::NONE,
                ));
            },
        };
        let strict = boolean(function, Key("strict")).map_err(|_malformed| {
            return ApiError::invalid(
                String::from("function strict must be a boolean"),
                Param("tools"),
                Code::NONE,
            );
        })?;
        if strict == Flag::True {
            return Err(ApiError::invalid(
                String::from("strict=true requires schema-constrained arguments"),
                Param("tools"),
                Code("strict_tools_not_supported"),
            ));
        }
        let mut rendered = Object::new();
        rendered.insert(String::from("name"), Value::from(name.as_str()));
        rendered.insert(String::from("parameters"), parameters);
        rendered.insert(String::from("strict"), Value::Bool(false));
        if !description.is_empty() {
            rendered.insert(String::from("description"), Value::from(description));
        }
        let mut definition = Object::new();
        definition.insert(String::from("type"), Value::from("function"));
        definition.insert(String::from("function"), Value::Object(rendered));
        declared.push(Tool {
            name,
            definition: Value::Object(definition).to_string(),
        });
    }
    return Ok(declared);
}

/// Whether the tools reach the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolUse
{
    /// Offered.
    Auto,
    /// Withheld: `tool_choice` is `none`.
    Withheld,
}

/// Apply `tool_choice` to the declared tools.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `auto` or omission offers every tool; `none` withholds them;
///   `allowed_tools` in `auto` mode offers only the named ones, in their
///   declared order.
/// - provides: the tools the prompt offers.
/// - fails: on `required`, a named function, a custom choice, `allowed_tools`
///   in `required` mode, or a malformed choice.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `tool_choice`.
fn tool_choice(
    body: &Object,
    declared: &mut Vec<Tool>,
) -> Result<ToolUse, ApiError>
{
    let Maybe::Present(choice) = field(body, Key("tool_choice"))
    else {
        return Ok(ToolUse::Auto);
    };
    if let Value::String(ref value) = *choice {
        return match value.as_str() {
            | "auto" => Ok(ToolUse::Auto),
            | "none" => Ok(ToolUse::Withheld),
            | "required" => Err(ApiError::invalid(
                String::from("tool_choice='required' requires at least one tool call"),
                Param("tool_choice"),
                Code("tool_choice_not_supported"),
            )),
            | _ => Err(ApiError::invalid(
                String::from(
                    "tool_choice must be 'auto', 'none', 'required', or a function choice",
                ),
                Param("tool_choice"),
                Code::NONE,
            )),
        };
    }
    let Some(object) = choice.as_object()
    else {
        return Err(ApiError::invalid(
            String::from("tool_choice must be a string or object"),
            Param("tool_choice"),
            Code::NONE,
        ));
    };
    match object.get("type").and_then(Value::as_str) {
        | Some("allowed_tools") => {},
        | Some("function") => {
            return Err(ApiError::invalid(
                String::from("a named tool_choice requires that exact function to be called"),
                Param("tool_choice"),
                Code("tool_choice_not_supported"),
            ));
        },
        | Some("custom") => {
            return Err(ApiError::invalid(
                String::from("custom tool_choice requires custom tool output"),
                Param("tool_choice"),
                Code("tool_type_not_supported"),
            ));
        },
        | Some(other) => {
            return Err(ApiError::invalid(
                format!("unsupported tool_choice type: {other}"),
                Param("tool_choice"),
                Code::NONE,
            ));
        },
        | None => {
            return Err(ApiError::invalid(
                String::from("tool_choice objects must contain a string type"),
                Param("tool_choice"),
                Code::NONE,
            ));
        },
    }
    let config = object.get("allowed_tools").unwrap_or(choice);
    let Some(config) = config.as_object()
    else {
        return Err(ApiError::invalid(
            String::from("tool_choice.allowed_tools must be an object"),
            Param("tool_choice"),
            Code::NONE,
        ));
    };
    let mode = config.get("mode").and_then(Value::as_str);
    if !matches!(mode, Some("auto" | "required")) {
        return Err(ApiError::invalid(
            String::from("tool_choice.allowed_tools.mode must be 'auto' or 'required'"),
            Param("tool_choice"),
            Code::NONE,
        ));
    }
    let Some(entries) = config.get("tools").and_then(Value::as_array)
    else {
        return Err(ApiError::invalid(
            String::from("tool_choice.allowed_tools.tools must be an array"),
            Param("tool_choice"),
            Code::NONE,
        ));
    };
    let mut allowed = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(item) = entry.as_object()
        else {
            return Err(ApiError::invalid(
                String::from("allowed tool entries must contain a string type"),
                Param("tool_choice"),
                Code::NONE,
            ));
        };
        if item.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::invalid(
                String::from("allowed_tools can select only function tools"),
                Param("tool_choice"),
                Code("tool_type_not_supported"),
            ));
        }
        let name = function_name(item, Param("tool_choice"))?;
        if !declared.iter().any(|tool| return tool.name == name) {
            return Err(ApiError::invalid(
                format!("allowed tool '{name}' is not present in tools"),
                Param("tool_choice"),
                Code::NONE,
            ));
        }
        allowed.push(name);
    }
    if mode == Some("required") {
        return Err(ApiError::invalid(
            String::from(
                "tool_choice.allowed_tools mode='required' requires at least one tool call",
            ),
            Param("tool_choice"),
            Code("tool_choice_not_supported"),
        ));
    }
    declared.retain(|tool| return allowed.contains(&tool.name));
    return Ok(ToolUse::Auto);
}

/// The stop strings.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a string or an array of at most four non-empty strings, in order;
///   stop strings end reasoning as well as content whenever `stop` is given.
/// - provides: the request's stops.
/// - fails: otherwise.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming `stop`.
fn stops(body: &Object) -> Result<(Vec<String>, StopScope), ApiError>
{
    let Maybe::Present(value) = field(body, Key("stop"))
    else {
        return Ok((Vec::new(), StopScope::Content));
    };
    let entries: Vec<&Value> = match *value {
        | Value::String(_) => vec![value],
        | Value::Array(ref list) if list.len() <= MAXIMUM_STOPS => list.iter().collect(),
        | Value::Array(_) => {
            return Err(ApiError::invalid(
                String::from("stop supports at most four strings"),
                Param("stop"),
                Code::NONE,
            ));
        },
        | _ => {
            return Err(ApiError::invalid(
                String::from("stop must be a string or array of strings"),
                Param("stop"),
                Code::NONE,
            ));
        },
    };
    let mut texts = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(text) = entry.as_str()
        else {
            return Err(ApiError::invalid(
                String::from("stop entries must be strings"),
                Param("stop"),
                Code::NONE,
            ));
        };
        if text.is_empty() {
            return Err(ApiError::invalid(
                String::from("stop strings must not be empty"),
                Param("stop"),
                Code::NONE,
            ));
        }
        texts.push(String::from(text));
    }
    return Ok((texts, StopScope::ContentAndReasoning));
}

/// Which phase a set of sampling fields governs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase
{
    /// The request's top level: the whole generation, or thinking when a
    /// post-thinking phase is given.
    Initial,
    /// `post_thinking`: the answer after thinking closes.
    PostThinking,
}

/// One phase's sampling overrides.
///
/// # Specification
/// - requires: nothing.
/// - ensures: each present field is carried within its range: temperature
///   `[0,2]`, top-p and min-p `[0,1]`, top-k `[0,20]`, penalties `[-2,2]`; the
///   value is narrowed to `f32`, as ninfer's server narrows it.
/// - provides: the overrides of either [`Phase`].
/// - fails: when a field is malformed, not finite (naming `sampling` at the top
///   level, `post_thinking.temperature` in the post-thinking phase), or out of
///   range (naming the field, with the phase's prefix).
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: as stated.
///
/// # Adequacy
/// - hypothesis: L3 at the range boundaries.
/// - witness: `tests::sampling_ranges_are_inclusive`
fn sampling(
    object: &Object,
    phase: Phase,
) -> Result<Sampling, ApiError>
{
    let prefix = match phase {
        | Phase::Initial => "",
        | Phase::PostThinking => "post_thinking.",
    };
    let finite = match phase {
        | Phase::Initial => String::from("sampling"),
        | Phase::PostThinking => String::from("post_thinking.temperature"),
    };
    let bounded =
        |key: &str, minimum: f64, maximum: f64, bounds: &str| -> Result<Setting<f32>, ApiError> {
            let Maybe::Present(Number(value)) = number(object, Key(key))?
            else {
                return Ok(Setting::ModelDefault);
            };
            if !value.is_finite() {
                return Err(ApiError::invalid(
                    String::from("sampling parameters must be finite"),
                    Param(&finite),
                    Code::NONE,
                ));
            }
            if value < minimum || value > maximum {
                let message = match phase {
                    | Phase::Initial => format!("{key} must be in {bounds}"),
                    | Phase::PostThinking => format!("{prefix}{key} is out of range"),
                };
                return Err(ApiError::invalid(
                    message,
                    Param(&format!("{prefix}{key}")),
                    Code::NONE,
                ));
            }
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "the Engine samples in f32, and ninfer's server narrows the same way"
            )]
            let narrowed = value as f32;
            return Ok(Setting::Set(narrowed));
        };
    let top_k = match integer(object, Key("top_k"))? {
        | Maybe::Absent(_) => Setting::ModelDefault,
        | Maybe::Present(Integer(value)) if (0_i32 ..= 20_i32).contains(&value) => {
            Setting::Set(value)
        },
        | Maybe::Present(_) => {
            return Err(ApiError::invalid(
                format!("{prefix}top_k must be in [0,20]"),
                Param(&format!("{prefix}top_k")),
                Code::NONE,
            ));
        },
    };
    let temperature = bounded("temperature", 0.0_f64, 2.0_f64, "[0,2]")?;
    let top_p = bounded("top_p", 0.0_f64, 1.0_f64, "[0,1]")?;
    let min_p = bounded("min_p", 0.0_f64, 1.0_f64, "[0,1]")?;
    let presence_penalty = bounded("presence_penalty", -2.0_f64, 2.0_f64, "[-2,2]")?;
    let frequency_penalty = bounded("frequency_penalty", -2.0_f64, 2.0_f64, "[-2,2]")?;
    return Ok(Sampling {
        temperature,
        top_k,
        top_p,
        min_p,
        presence_penalty,
        frequency_penalty,
    });
}

/// A reasoning effort's name.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the seven named efforts map to their namesakes.
/// - provides: effort parsing for both spellings; [`Unnamed::Unknown`] for any
///   other value, a string or not.
/// - fails: never.
/// - panics: none.
fn effort_named(value: &Value) -> Maybe<Effort, Unnamed>
{
    let Some(name) = value.as_str()
    else {
        return Maybe::Absent(Unnamed::Unknown);
    };
    return match name {
        | "none" => Maybe::Present(Effort::None),
        | "minimal" => Maybe::Present(Effort::Minimal),
        | "low" => Maybe::Present(Effort::Low),
        | "medium" => Maybe::Present(Effort::Medium),
        | "high" => Maybe::Present(Effort::High),
        | "xhigh" => Maybe::Present(Effort::XHigh),
        | "max" => Maybe::Present(Effort::Max),
        | _ => Maybe::Absent(Unnamed::Unknown),
    };
}

/// Why a name names no effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unnamed
{
    /// The value is not one of the seven names.
    Unknown,
}

/// Merge a typed template switch with its `chat_template_kwargs` spelling,
/// removing the key from the arguments.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the nested value when present, else the typed one; the key is
///   gone from `kwargs`.
/// - provides: `enable_thinking` and `preserve_thinking`.
/// - fails: when the nested value is not a boolean, or both are set and differ.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: [`nullable_boolean`]'s, or code
///   `conflicting_template_option`, naming `key`.
fn merge_switch(
    kwargs: &mut Object,
    key: Key<'_>,
    typed: Flag,
) -> Result<Flag, ApiError>
{
    let nested = nullable_boolean(kwargs, key)?;
    let Key(key) = key;
    let _merged = kwargs.shift_remove(key);
    if nested == Flag::Unset {
        return Ok(typed);
    }
    if typed != Flag::Unset && typed != nested {
        return Err(ApiError::invalid(
            format!("conflicting {key} values"),
            Param(key),
            Code("conflicting_template_option"),
        ));
    }
    return Ok(nested);
}

/// A switch's neutral form.
///
/// # Specification
/// trivial.
const fn switch_of(value: Flag) -> Switch
{
    return match value {
        | Flag::True => Switch::On,
        | Flag::False => Switch::Off,
        | Flag::Unset => Switch::ModelDefault,
    };
}

/// The template choices: thinking, preserved thinking, effort, and the
/// remaining template arguments.
///
/// # Specification
/// - requires: nothing.
/// - ensures: top-level and `chat_template_kwargs` spellings of
///   `enable_thinking`, `preserve_thinking` and `reasoning_effort` merge into
///   one value each and leave the arguments; an effort sets thinking on unless
///   it is `none`; the remaining arguments keep their wire order and serialize
///   as a JSON object, `{}` when there are none.
/// - provides: the prompt options.
/// - fails: on a malformed or conflicting choice.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming the choice.
///
/// # Adequacy
/// - hypothesis: L3 — nested and typed spellings, their conflict, and an effort
///   implying thinking.
/// - witness: `tests::template_choices_merge_and_leave_the_arguments`
fn template_choices(body: &Object) -> Result<(Switch, Switch, Effort, String), ApiError>
{
    let mut kwargs = match field(body, Key("chat_template_kwargs")) {
        | Maybe::Absent(_) => Object::new(),
        | Maybe::Present(&Value::Object(ref map)) => map.clone(),
        | Maybe::Present(_) => {
            return Err(ApiError::invalid(
                String::from("chat_template_kwargs must be an object"),
                Param("chat_template_kwargs"),
                Code::NONE,
            ));
        },
    };
    let typed_thinking = nullable_boolean(body, Key("enable_thinking"))?;
    let thinking = merge_switch(&mut kwargs, Key("enable_thinking"), typed_thinking)?;
    let typed_preserve = nullable_boolean(body, Key("preserve_thinking"))?;
    let preserve = merge_switch(&mut kwargs, Key("preserve_thinking"), typed_preserve)?;
    let mut effort = match field(body, Key("reasoning_effort")) {
        | Maybe::Absent(_) => Effort::Unrequested,
        | Maybe::Present(value) => {
            if !value.is_string() {
                return Err(ApiError::invalid(
                    String::from("reasoning_effort must be a string or null"),
                    Param("reasoning_effort"),
                    Code::NONE,
                ));
            }
            let Maybe::Present(effort) = effort_named(value)
            else {
                return Err(ApiError::invalid(
                    String::from(
                        "reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or max",
                    ),
                    Param("reasoning_effort"),
                    Code::NONE,
                ));
            };
            effort
        },
    };
    match kwargs.shift_remove("reasoning_effort") {
        | None | Some(Value::Null) => {},
        | Some(ref name @ Value::String(_)) => {
            let Maybe::Present(nested) = effort_named(name)
            else {
                return Err(ApiError::invalid(
                    String::from("invalid reasoning_effort"),
                    Param("reasoning_effort"),
                    Code("invalid_template_option"),
                ));
            };
            if effort != Effort::Unrequested && effort != nested {
                return Err(ApiError::invalid(
                    String::from("conflicting reasoning_effort values"),
                    Param("reasoning_effort"),
                    Code("conflicting_template_option"),
                ));
            }
            effort = nested;
        },
        | Some(_) => {
            return Err(ApiError::invalid(
                String::from("reasoning_effort must be a string"),
                Param("reasoning_effort"),
                Code("invalid_template_option"),
            ));
        },
    }
    let mut thinking = switch_of(thinking);
    if effort != Effort::Unrequested {
        let enables = if effort == Effort::None {
            Switch::Off
        }
        else {
            Switch::On
        };
        if thinking != Switch::ModelDefault && thinking != enables {
            return Err(ApiError::invalid(
                String::from("reasoning effort conflicts with enable_thinking"),
                Param("reasoning_effort"),
                Code("conflicting_template_option"),
            ));
        }
        thinking = enables;
    }
    return Ok((
        thinking,
        switch_of(preserve),
        effort,
        Value::Object(kwargs).to_string(),
    ));
}

/// The output limit.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `max_completion_tokens`, else `max_tokens`, else the default.
/// - provides: the request's output limit.
/// - fails: when the named field is malformed or negative.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: naming the field.
fn output_tokens(
    body: &Object,
    defaults: Defaults,
) -> Result<TokenCount, ApiError>
{
    for key in ["max_completion_tokens", "max_tokens"] {
        if let Maybe::Present(Integer(limit)) = integer(body, Key(key))? {
            return u32::try_from(limit)
                .map(TokenCount::from)
                .map_err(|_negative| {
                    return ApiError::invalid(
                        format!("{key} must be nonnegative"),
                        Param(key),
                        Code::NONE,
                    );
                });
        }
    }
    return Ok(defaults.output_tokens);
}

/// Translate a Chat Completions body.
///
/// # Specification
/// - requires: `seed_if_unset` is the fresh seed to use when the body names
///   none.
/// - ensures: on success the chat request renders and runs as ninfer's server
///   renders and runs the same body: the same turns, tool definitions, template
///   arguments and choices, output limit, sampling for both phases, seeds,
///   stops, special-token handling and tool-name limit; the default thinking
///   budget applies unless thinking is turned off, by `enable_thinking` or by
///   effort `none`.
/// - provides: the chat surface's request translation.
/// - fails: with ninfer's parameter and code for every body it refuses.
/// - panics: none.
///
/// # Errors
/// - [`ApiError`]: the body is refused.
///
/// # Adequacy
/// - hypothesis: L2 — a streamed tool-using body translates to the documented
///   fields; the served comparison against `ninfer-serve` checks rendering end
///   to end.
/// - witness: `tests::a_tool_turn_translates_whole`
#[inline]
pub fn chat_request(
    body: &Value,
    defaults: Defaults,
    seed_if_unset: FreshSeed,
) -> Result<ParsedChat, ApiError>
{
    let Some(body) = body.as_object()
    else {
        return Err(ApiError::invalid(
            String::from("request body must be a JSON object"),
            Param::NONE,
            Code::NONE,
        ));
    };
    refuse_unsupported(body)?;
    let model = match body.get("model").and_then(Value::as_str) {
        | Some(name) if !name.is_empty() => ModelId(String::from(name)),
        | _ => {
            return Err(ApiError::invalid(
                String::from("missing required field: model"),
                Param("model"),
                Code::NONE,
            ));
        },
    };
    let policy = cache_policy(body)?;
    let mut declared = tools(body)?;
    let tool_use = tool_choice(body, &mut declared)?;
    if boolean(body, Key("parallel_tool_calls"))? == Flag::False
        && tool_use == ToolUse::Auto
        && !declared.is_empty()
    {
        return Err(ApiError::invalid(
            String::from("parallel_tool_calls=false requires at most one tool call"),
            Param("parallel_tool_calls"),
            Code("parallel_tool_calls_not_supported"),
        ));
    }
    let Some(list) = body.get("messages")
    else {
        return Err(ApiError::invalid(
            String::from("missing required field: messages"),
            Param("messages"),
            Code::NONE,
        ));
    };
    let Some(list) = list.as_array().filter(|list| return !list.is_empty())
    else {
        return Err(ApiError::invalid(
            String::from("messages must be a non-empty array"),
            Param("messages"),
            Code::NONE,
        ));
    };
    let mut turns = Vec::with_capacity(list.len());
    for (index, value) in list.iter().enumerate() {
        turns.push(message(value, Position(index))?);
    }
    let (stops, stop_scope) = stops(body)?;
    let post_thinking = match field(body, Key("post_thinking")) {
        | Maybe::Absent(_) => (Sampling::MODEL_DEFAULT, Seed::Inherited),
        | Maybe::Present(&Value::Object(ref object)) => {
            let phase = sampling(object, Phase::PostThinking)?;
            let seed = seed(object, Param("post_thinking.seed"))?;
            (phase, seed)
        },
        | Maybe::Present(_) => {
            return Err(ApiError::invalid(
                String::from("post_thinking must be an object"),
                Param("post_thinking"),
                Code::NONE,
            ));
        },
    };
    let initial = sampling(body, Phase::Initial)?;
    let seed = match seed(body, Param("seed"))? {
        | Seed::Fixed(bits) => bits,
        | Seed::Inherited => seed_if_unset.0,
    };
    if let Maybe::Present(count) = integer(body, Key("n"))?
        && count != Integer(1_i32)
    {
        return Err(ApiError::invalid(
            String::from(
                "n requests multiple completions, while NInfer produces one completion per \
                 request; only n=1 is supported",
            ),
            Param("n"),
            Code("n_not_supported"),
        ));
    }
    let delivery = match boolean(body, Key("stream"))? {
        | Flag::True => Delivery::Streaming,
        | Flag::False | Flag::Unset => Delivery::Aggregate,
    };
    let include_usage = match field(body, Key("stream_options")) {
        | Maybe::Absent(_) => Usage::Omitted,
        | Maybe::Present(&Value::Object(ref options)) => {
            if let Maybe::Present(value) = field(options, Key("include_obfuscation"))
                && !value.is_boolean()
            {
                return Err(ApiError::invalid(
                    String::from("include_obfuscation must be a boolean"),
                    Param("stream_options"),
                    Code::NONE,
                ));
            }
            match boolean(options, Key("include_usage"))? {
                | Flag::True => Usage::Included,
                | Flag::False | Flag::Unset => Usage::Omitted,
            }
        },
        | Maybe::Present(_) => {
            return Err(ApiError::invalid(
                String::from("stream_options must be an object"),
                Param("stream_options"),
                Code::NONE,
            ));
        },
    };
    let output_tokens = output_tokens(body, defaults)?;
    let (thinking, preserve_thinking, effort, template_arguments) = template_choices(body)?;
    let thinking_budget = if thinking == Switch::Off || effort == Effort::None {
        ThinkingBudget::Unlimited
    }
    else {
        defaults.thinking_budget
    };
    let offered = tool_use == ToolUse::Auto && !declared.is_empty();
    let tool_history = turns.iter().any(|turn| {
        return turn.message.role == Role::Tool || !turn.message.tool_calls.is_empty();
    });
    let special_tokens = if offered || tool_history {
        SpecialTokens::Preserved
    }
    else {
        SpecialTokens::Trimmed
    };
    let tools: Vec<String> = if offered {
        declared
            .into_iter()
            .map(|tool| return tool.definition)
            .collect()
    }
    else {
        Vec::new()
    };
    let cache = prompt_cache(&turns, &tools, policy)?;
    let messages = turns.into_iter().map(|turn| return turn.message).collect();
    return Ok(ParsedChat {
        model,
        delivery,
        include_usage,
        request: ChatRequest {
            prompt: Prompt {
                messages,
                tools,
                template_arguments,
                thinking,
                preserve_thinking,
                effort,
                cache,
            },
            generation: Generation {
                output_tokens,
                sampling: initial,
                seed,
                post_thinking: post_thinking.0,
                post_thinking_seed: post_thinking.1,
                thinking_budget,
                stops,
                stop_scope,
                special_tokens,
                tool_name_limit: TOOL_NAME_LIMIT,
                prefix_reuse: PrefixReuse::ReadWrite,
            },
            delivery,
        },
    });
}

/// Tests for the translation.
#[cfg(test)]
mod tests
{
    use infinitum_chat::Automatic;
    use infinitum_chat::CacheBoundary;
    use infinitum_chat::CacheMarker;
    use infinitum_chat::Count;
    use infinitum_chat::Delivery;
    use infinitum_chat::Effort;
    use infinitum_chat::InstructionBytes;
    use infinitum_chat::Marked;
    use infinitum_chat::Role;
    use infinitum_chat::Setting;
    use infinitum_chat::SpecialTokens;
    use infinitum_chat::StopScope;
    use infinitum_chat::StructuralPrefixes;
    use infinitum_chat::Switch;
    use infinitum_chat::ThinkingBudget;
    use infinitum_round::Maybe;
    use infinitum_round::TokenCount;
    use serde_json::json;

    use super::Defaults;
    use super::FreshSeed;
    use super::Integer;
    use super::Key;
    use super::Phase;
    use super::Usage;
    use super::chat_request;
    use super::integer;
    use super::sampling;
    use super::template_choices;
    use super::tools;

    /// The defaults the tests translate under.
    const DEFAULTS: Defaults = Defaults {
        output_tokens: TokenCount::ZERO,
        thinking_budget: ThinkingBudget::Unlimited,
    };

    /// The cache markers ninfer's server derives for `messages`, with
    /// `options` as `prompt_cache_options` when present.
    ///
    /// # Specification
    /// - requires: the body translates.
    /// - ensures: the translated prompt's markers, after checking that
    ///   structural prefixes are withheld.
    /// - provides: the cache-policy cases' round trip.
    fn markers(
        messages: &serde_json::Value,
        options: Option<serde_json::Value>,
    ) -> Vec<(CacheBoundary, Marked, Automatic)>
    {
        let mut body = json!({"model": "m", "messages": messages});
        if let Some(options) = options {
            body["prompt_cache_options"] = options;
        }
        let cache = chat_request(&body, DEFAULTS, FreshSeed(0))
            .unwrap()
            .request
            .prompt
            .cache;
        assert_eq!(
            cache.structural,
            StructuralPrefixes::Withheld,
            "Chat Completions has its own write policy"
        );
        return cache
            .markers
            .into_iter()
            .map(
                |CacheMarker {
                     boundary,
                     marked,
                     automatic,
                 }| return (boundary, marked, automatic),
            )
            .collect();
    }

    /// The default budget reaches a request that leaves thinking on or to the
    /// model, and not one that turns it off either way.
    #[test]
    fn the_default_thinking_budget_skips_requests_without_thinking()
    {
        let budget = ThinkingBudget::Tokens(core::num::NonZeroU32::MIN);
        let defaults = Defaults {
            thinking_budget: budget,
            ..DEFAULTS
        };
        let cases = [
            (json!({}), budget, "left to the model"),
            (
                json!({"chat_template_kwargs": {"enable_thinking": true}}),
                budget,
                "turned on",
            ),
            (
                json!({"chat_template_kwargs": {"enable_thinking": false}}),
                ThinkingBudget::Unlimited,
                "turned off",
            ),
            (
                json!({"reasoning_effort": "none"}),
                ThinkingBudget::Unlimited,
                "effort none",
            ),
            (json!({"reasoning_effort": "low"}), budget, "effort low"),
        ];
        for (extra, expected, case) in cases {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "u"}]});
            for (key, value) in extra.as_object().unwrap() {
                body[key] = value.clone();
            }
            let parsed = chat_request(&body, defaults, FreshSeed(0)).unwrap();
            assert_eq!(
                parsed.request.generation.thinking_budget, expected,
                "{case}"
            );
        }
    }

    #[test]
    fn cache_markers_follow_ninfers_policy()
    {
        let marked = |text: &str| {
            return json!({"type": "text", "text": text, "prompt_cache_breakpoint": {"mode": "explicit"}});
        };
        let part = |message: u32, parts: u32| {
            return CacheBoundary::MessagePart {
                message: Count(message),
                parts: Count(parts),
            };
        };
        let plain = json!([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]);
        assert_eq!(
            markers(&plain, None),
            vec![(part(2, 1), Marked::Unmarked, Automatic::Default)],
            "the default write lands after the last part"
        );
        assert_eq!(
            markers(&plain, Some(json!({"mode": "implicit"}))),
            vec![(part(2, 1), Marked::Unmarked, Automatic::Requested)],
            "prompt_cache_options requests the write"
        );
        assert!(
            markers(&plain, Some(json!({"mode": "explicit"}))).is_empty(),
            "explicit mode leaves only breakpoints"
        );
        let leading = json!([
            {"role": "system", "content": [{"type": "text", "text": "ab"}, marked("cde")]},
            {"role": "user", "content": "u"}
        ]);
        assert_eq!(
            markers(&leading, None),
            vec![
                (
                    CacheBoundary::LeadingInstruction(InstructionBytes(5)),
                    Marked::Explicit,
                    Automatic::Not
                ),
                (part(2, 1), Marked::Unmarked, Automatic::Default),
            ],
            "a leading instruction's breakpoint is placed by its cumulative bytes"
        );
        let called = json!([
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]}
        ]);
        assert_eq!(
            markers(&called, None),
            vec![(
                CacheBoundary::Message(Count(2)),
                Marked::Unmarked,
                Automatic::Default
            )],
            "a turn ending in tool calls takes the write at its boundary"
        );
        let merged = json!([{"role": "user", "content": [marked("a"), marked("b"), marked("c"), marked("d"), marked("e")]}]);
        assert_eq!(
            markers(&merged, None),
            vec![
                (part(1, 2), Marked::Explicit, Automatic::Not),
                (part(1, 3), Marked::Explicit, Automatic::Not),
                (part(1, 4), Marked::Explicit, Automatic::Not),
                (part(1, 5), Marked::Explicit, Automatic::Default),
            ],
            "four writes are kept when the automatic one joins a breakpoint"
        );
        let apart = json!([{"role": "user", "content": [marked("a"), marked("b"), marked("c"), marked("d"), {"type": "text", "text": "e"}]}]);
        assert_eq!(
            markers(&apart, None),
            vec![
                (part(1, 2), Marked::Explicit, Automatic::Not),
                (part(1, 3), Marked::Explicit, Automatic::Not),
                (part(1, 4), Marked::Explicit, Automatic::Not),
                (part(1, 5), Marked::Unmarked, Automatic::Default),
            ],
            "three breakpoints are kept when the automatic write needs its own slot"
        );
        assert_eq!(
            markers(&apart, Some(json!({"mode": "explicit"}))).len(),
            4,
            "without the automatic write all four breakpoints are kept"
        );
    }

    #[test]
    fn integers_are_checked_and_bounded()
    {
        let body = json!({"a": i32::MAX, "b": i64::from(i32::MAX) + 1, "c": 1.5_f64, "d": null});
        let object = body.as_object().unwrap();
        assert_eq!(
            integer(object, Key("a")),
            Ok(Maybe::Present(Integer(i32::MAX))),
            "i32::MAX is accepted"
        );
        assert!(
            integer(object, Key("b")).is_err(),
            "one past i32::MAX is out of range"
        );
        assert!(
            integer(object, Key("c")).is_err(),
            "a fraction is not an integer"
        );
        assert!(
            matches!(integer(object, Key("d")), Ok(Maybe::Absent(_))),
            "null is omitted"
        );
    }

    #[test]
    fn sampling_ranges_are_inclusive()
    {
        let edges = json!({"temperature": 2.0_f64, "top_p": 0.0_f64, "top_k": 20_i32, "min_p": 1.0_f64, "presence_penalty": -2.0_f64});
        let phase = sampling(edges.as_object().unwrap(), Phase::Initial).unwrap();
        assert_eq!(
            phase.temperature,
            Setting::Set(2.0_f32),
            "temperature 2 is in range"
        );
        assert_eq!(phase.top_k, Setting::Set(20_i32), "top_k 20 is in range");
        for outside in [
            json!({"temperature": 2.001_f64}),
            json!({"top_k": 21_i32}),
            json!({"top_p": -0.001_f64}),
        ] {
            let refused = sampling(outside.as_object().unwrap(), Phase::Initial).unwrap_err();
            assert_eq!(
                refused.status.as_u16(),
                400_u16,
                "{outside} is out of range"
            );
        }
    }

    #[test]
    fn tools_render_in_ninfer_key_order()
    {
        let body = json!({"tools": [
            {"type": "function", "function": {"description": "d", "name": "read", "parameters": {"type": "object", "properties": {"z": {}, "a": {}}}}},
            {"type": "function", "function": {"name": "ls"}}
        ]});
        let declared = tools(body.as_object().unwrap()).unwrap();
        let rendered: Vec<&str> = declared
            .iter()
            .map(|tool| return tool.definition.as_str())
            .collect();
        assert_eq!(
            rendered,
            [
                r#"{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"z":{},"a":{}}},"strict":false,"description":"d"}}"#,
                r#"{"type":"function","function":{"name":"ls","parameters":{"type":"object","properties":{}},"strict":false}}"#,
            ],
            "definitions keep ninfer's key order and the schema's wire order"
        );
    }

    #[test]
    fn template_choices_merge_and_leave_the_arguments()
    {
        let body = json!({"chat_template_kwargs": {"x": 1_i32, "enable_thinking": true, "a": 2_i32}, "reasoning_effort": "low"});
        let (thinking, preserve, effort, arguments) =
            template_choices(body.as_object().unwrap()).unwrap();
        assert_eq!(
            thinking,
            Switch::On,
            "the nested switch and the effort agree on thinking"
        );
        assert_eq!(
            preserve,
            Switch::ModelDefault,
            "an unset switch stays unset"
        );
        assert_eq!(effort, Effort::Low, "the effort is carried");
        assert_eq!(
            arguments, r#"{"x":1,"a":2}"#,
            "merged keys leave the arguments, the rest keep order"
        );
        let conflict = json!({"enable_thinking": false, "reasoning_effort": "high"});
        assert_eq!(
            template_choices(conflict.as_object().unwrap())
                .unwrap_err()
                .code,
            "conflicting_template_option",
            "an effort that implies thinking conflicts with thinking off"
        );
        let none = json!({});
        assert_eq!(
            template_choices(none.as_object().unwrap()).unwrap().3,
            "{}",
            "no arguments serialize as an empty object"
        );
        let mistyped = json!({"chat_template_kwargs": {"enable_thinking": "yes"}});
        let refused = template_choices(mistyped.as_object().unwrap()).unwrap_err();
        assert_eq!(
            (refused.message.as_str(), refused.code.as_str()),
            ("enable_thinking must be a boolean or null", ""),
            "a mistyped nested switch fails as ninfer's parser fails it"
        );
        let unnamed = json!({"reasoning_effort": 3_i32});
        assert_eq!(
            template_choices(unnamed.as_object().unwrap())
                .unwrap_err()
                .message,
            "reasoning_effort must be a string or null",
            "an effort that is not a string"
        );
    }

    #[test]
    fn a_tool_turn_translates_whole()
    {
        let body = json!({
            "model": "m",
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_completion_tokens": 32_i32,
            "temperature": 0_i32,
            "stop": "END",
            "tools": [{"type": "function", "function": {"name": "ls"}}],
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "u"}]},
                {"role": "assistant", "content": null, "reasoning_content": "r", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "out"}
            ]
        });
        let parsed = chat_request(&body, DEFAULTS, FreshSeed(7)).unwrap();
        assert_eq!(parsed.model.0, "m", "the model is carried");
        assert_eq!(
            parsed.delivery,
            Delivery::Streaming,
            "stream selects streaming"
        );
        assert_eq!(
            parsed.include_usage,
            Usage::Included,
            "include_usage is carried"
        );
        let request = parsed.request;
        assert_eq!(
            request.generation.output_tokens,
            TokenCount::from(32_u32),
            "the limit is carried"
        );
        assert_eq!(
            request.generation.seed, 7,
            "an unset seed takes the fresh one"
        );
        assert_eq!(request.generation.stops, ["END"], "the stop is carried");
        assert_eq!(
            request.generation.stop_scope,
            StopScope::ContentAndReasoning,
            "a stop also ends reasoning"
        );
        assert_eq!(
            request.generation.special_tokens,
            SpecialTokens::Preserved,
            "tools preserve special tokens"
        );
        assert_eq!(request.prompt.tools.len(), 1, "the tool is offered");
        let roles: Vec<Role> = request
            .prompt
            .messages
            .iter()
            .map(|turn| return turn.role)
            .collect();
        assert_eq!(
            roles,
            [Role::System, Role::User, Role::Assistant, Role::Tool],
            "turns keep order"
        );
        let assistant = &request.prompt.messages[2];
        assert_eq!(assistant.reasoning, "r", "carried reasoning is kept");
        assert_eq!(
            assistant.tool_calls.first().unwrap().id,
            "c1",
            "the call id is kept"
        );
        assert_eq!(
            request.prompt.messages[3].tool_call_id, "c1",
            "the tool turn names its call"
        );
        let withheld = json!({"model": "m", "tool_choice": "none", "tools": [{"type": "function", "function": {"name": "ls"}}], "messages": [{"role": "user", "content": "u"}]});
        let parsed = chat_request(&withheld, DEFAULTS, FreshSeed(0)).unwrap();
        assert!(
            parsed.request.prompt.tools.is_empty(),
            "tool_choice none withholds the tools"
        );
        assert_eq!(
            parsed.request.generation.special_tokens,
            SpecialTokens::Trimmed,
            "withheld tools trim special tokens"
        );
    }
}
