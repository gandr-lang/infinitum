//! Chat Completions responses: the aggregate body, and the server-sent-event
//! chunks of a streamed one.
//!
//! Both follow ninfer's server field for field: the same message shape, the
//! same finish reasons, the same `usage` and llama.cpp-style `timings`, and
//! the same chunk sequence, so a client reads either server the same way.

use infinitum_chat::Channel;
use infinitum_chat::ChatOutcome;
use infinitum_chat::DeltaText;
use infinitum_chat::Finish;
use infinitum_chat::GeneratedToolCall;
use infinitum_round::TokenCount;
use serde_json::Value;
use serde_json::json;

use crate::error::ApiError;
use crate::parse::Usage;

/// Fresh random bits.
///
/// # Specification
/// - requires: nothing.
/// - ensures: each call draws from a hasher keyed afresh from the standard
///   library's per-process random keys.
/// - provides: response ids, tool-call ids and unset seeds; not cryptographic,
///   as ninfer's are not.
/// - fails: never.
/// - panics: none.
#[inline]
#[must_use]
pub fn fresh_bits() -> Bits
{
    use core::hash::BuildHasher as _;
    return Bits(std::hash::RandomState::new().hash_one(0_u8));
}

/// Random bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bits(pub u64);

/// What an id names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Identified
{
    /// A response: `chatcmpl-`.
    Completion,
    /// A generated tool call: `call_`.
    ToolCall,
}

/// A fresh id: the kind's prefix, then sixteen lowercase hex digits.
///
/// # Specification
/// trivial.
fn identifier(kind: Identified) -> String
{
    let prefix = match kind {
        | Identified::Completion => "chatcmpl-",
        | Identified::ToolCall => "call_",
    };
    return format!("{prefix}{:016x}", fresh_bits().0);
}

/// Seconds since the Unix epoch, zero before it.
///
/// # Specification
/// trivial.
#[inline]
#[must_use]
pub fn unix_now() -> UnixSeconds
{
    return UnixSeconds(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| return since.as_secs()),
    );
}

/// A time in seconds since the Unix epoch.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnixSeconds(pub u64);

/// A payload's `object`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectKind
{
    /// `chat.completion`.
    Completion,
    /// `chat.completion.chunk`.
    Chunk,
}

/// The identity every chunk of one response shares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity
{
    /// `chatcmpl-` and sixteen hex digits.
    id: String,
    /// The model id the request named.
    model: String,
    /// The response's creation time, in Unix seconds.
    created: u64,
}

impl Identity
{
    /// A fresh identity for a response from `model`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn new(model: String) -> Self
    {
        return Self {
            id: identifier(Identified::Completion),
            model,
            created: unix_now().0,
        };
    }

    /// The fields every payload opens with.
    ///
    /// # Specification
    /// trivial.
    fn payload(
        &self,
        object: ObjectKind,
    ) -> serde_json::Map<String, Value>
    {
        let object = match object {
            | ObjectKind::Completion => "chat.completion",
            | ObjectKind::Chunk => "chat.completion.chunk",
        };
        let mut payload = serde_json::Map::new();
        payload.insert(String::from("id"), Value::from(self.id.as_str()));
        payload.insert(String::from("object"), Value::from(object));
        payload.insert(String::from("created"), Value::from(self.created));
        payload.insert(String::from("model"), Value::from(self.model.as_str()));
        return payload;
    }
}

/// The `OpenAI` finish reason of a generation without tool calls.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the output limit and the context ceiling finish `length`; every
///   other reason finishes `stop`.
/// - provides: `finish_reason`.
/// - fails: never.
/// - panics: none.
const fn finish_reason(finish: Finish) -> FinishReason
{
    return match finish {
        | Finish::OutputLimit | Finish::ContextCapacity => FinishReason::Length,
        | Finish::StopToken | Finish::StopString | Finish::Cancelled | Finish::Unfinished => {
            FinishReason::Stop
        },
    };
}

/// A choice's `finish_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinishReason
{
    /// `length`: a limit bound.
    Length,
    /// `stop`: the model or a stop string ended it.
    Stop,
    /// `tool_calls`: the model called tools.
    ToolCalls,
}

impl From<FinishReason> for Value
{
    /// The reason's wire string.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(reason: FinishReason) -> Self
    {
        return Self::from(match reason {
            | FinishReason::Length => "length",
            | FinishReason::Stop => "stop",
            | FinishReason::ToolCalls => "tool_calls",
        });
    }
}

/// Tokens as a JSON number.
///
/// # Specification
/// trivial.
fn tokens(count: TokenCount) -> Value
{
    return Value::from(u32::from(count));
}

/// A token total that may exceed `u32`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Total(u64);

/// The completion's token count.
///
/// # Specification
/// trivial.
fn completion_tokens(outcome: &ChatOutcome) -> Total
{
    return Total(u64::try_from(outcome.generated.len()).unwrap_or(u64::MAX));
}

/// The `usage` object.
///
/// # Specification
/// - requires: nothing.
/// - ensures: prompt, cached (clamped to the prompt), completion, reasoning and
///   total tokens, in ninfer's nesting.
/// - provides: aggregate and stream usage.
/// - fails: never.
/// - panics: none.
fn usage(outcome: &ChatOutcome) -> Value
{
    let prompt = u32::from(outcome.admission.prompt_tokens);
    let cached = u32::from(outcome.admission.reused_tokens).min(prompt);
    let Total(completion) = completion_tokens(outcome);
    return json!({
        "prompt_tokens": prompt,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens": completion,
        "completion_tokens_details": {"reasoning_tokens": tokens(outcome.reasoning_tokens)},
        "total_tokens": u64::from(prompt).saturating_add(completion),
    });
}

/// The llama.cpp-style `timings` object.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `cache_n` is the reused prompt tokens clamped to the prompt;
///   `prompt_n` the rest; `predicted_n` the generated tokens; the per-token and
///   per-second rates divide by `prompt_n` and by the decode intervals (one
///   fewer than the generated tokens), zero when those are zero; the draft
///   counts appear only when something was drafted.
/// - provides: the timings ninfer's server reports.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 — the intervals rule and the draft fields' presence.
/// - witness: `tests::timings_follow_ninfer`
fn timings(outcome: &ChatOutcome) -> Value
{
    let prompt = u32::from(outcome.admission.prompt_tokens);
    let cached = u32::from(outcome.admission.reused_tokens).min(prompt);
    let milliseconds = |duration: core::time::Duration| return duration.as_secs_f64() * 1000.0_f64;
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "a rate is reported as a float, as ninfer's is"
    )]
    let per_token = |count: u64, elapsed: f64| {
        if count == 0 {
            return 0.0_f64;
        }
        return elapsed / count as f64;
    };
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "a rate is reported as a float, as ninfer's is"
    )]
    let per_second = |count: u64, elapsed: f64| {
        if elapsed <= 0.0_f64 {
            return 0.0_f64;
        }
        return 1000.0_f64 * count as f64 / elapsed;
    };
    let prefilled = u64::from(prompt.saturating_sub(cached));
    let prompt_ms = milliseconds(outcome.prompt_wall);
    let Total(predicted) = completion_tokens(outcome);
    let predicted_ms = milliseconds(outcome.generation_wall);
    let intervals = predicted.saturating_sub(1);
    let mut rendered = json!({
        "cache_n": cached,
        "prompt_n": prefilled,
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": per_token(prefilled, prompt_ms),
        "prompt_per_second": per_second(prefilled, prompt_ms),
        "predicted_n": predicted,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": per_token(intervals, predicted_ms),
        "predicted_per_second": per_second(intervals, predicted_ms),
    });
    if outcome.drafted.0 != 0
        && let Value::Object(ref mut fields) = rendered
    {
        fields.insert(String::from("draft_n"), Value::from(outcome.drafted.0));
        fields.insert(
            String::from("draft_n_accepted"),
            Value::from(outcome.accepted.0),
        );
    }
    return rendered;
}

/// The tool calls, each with a fresh `call_` id, and with its position when
/// streamed.
///
/// # Specification
/// trivial.
fn tool_calls(
    calls: &[GeneratedToolCall],
    indexed: Indexing,
) -> Value
{
    return calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let mut value = json!({
                "id": identifier(Identified::ToolCall),
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            });
            if indexed == Indexing::Indexed
                && let Value::Object(ref mut fields) = value
            {
                fields.insert(String::from("index"), Value::from(index));
            }
            return value;
        })
        .collect();
}

/// Whether rendered tool calls carry their position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Indexing
{
    /// The aggregate message's calls.
    Unindexed,
    /// A stream chunk's calls.
    Indexed,
}

/// The aggregate `chat.completion` body.
///
/// # Specification
/// - requires: nothing.
/// - ensures: one choice whose message carries the content, `refusal` null, the
///   reasoning as `reasoning_content` when there is any, and the tool calls
///   when there are any, with content then null when empty; the finish reason
///   is `tool_calls` when there are calls; `usage` and `timings` follow the
///   choices.
/// - provides: the non-streamed response.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 — a plain answer and a tool call with empty content.
/// - witness: `tests::a_completion_carries_ninfers_fields`
#[inline]
#[must_use]
pub fn completion(
    identity: &Identity,
    outcome: &ChatOutcome,
) -> String
{
    let mut message = serde_json::Map::new();
    message.insert(String::from("role"), Value::from("assistant"));
    message.insert(
        String::from("content"),
        Value::from(outcome.content.as_str()),
    );
    message.insert(String::from("refusal"), Value::Null);
    if !outcome.reasoning.is_empty() {
        message.insert(
            String::from("reasoning_content"),
            Value::from(outcome.reasoning.as_str()),
        );
    }
    let finish = if outcome.tool_calls.is_empty() {
        finish_reason(outcome.finish)
    }
    else {
        if outcome.content.is_empty() {
            message.insert(String::from("content"), Value::Null);
        }
        message.insert(
            String::from("tool_calls"),
            tool_calls(&outcome.tool_calls, Indexing::Unindexed),
        );
        FinishReason::ToolCalls
    };
    let mut payload = identity.payload(ObjectKind::Completion);
    payload.insert(
        String::from("choices"),
        json!([{"index": 0_i32, "message": message, "logprobs": null, "finish_reason": Value::from(finish)}]),
    );
    payload.insert(String::from("usage"), usage(outcome));
    payload.insert(String::from("timings"), timings(outcome));
    return Value::Object(payload).to_string();
}

/// One server-sent event carrying `payload`.
///
/// # Specification
/// trivial.
fn event(payload: &Value) -> String
{
    return format!("data: {payload}\n\n");
}

/// A stream's error event.
///
/// # Specification
/// trivial.
#[inline]
#[must_use]
pub fn error_event(error: &ApiError) -> String
{
    return format!("data: {}\n\n", error.body());
}

/// The stream's closing event.
pub const DONE: &str = "data: [DONE]\n\n";

/// A streamed response's encoder: it turns the backend's deltas and outcome
/// into chunks, keeping what it has streamed to check the outcome against.
#[derive(Debug, Clone)]
pub struct ChunkStream
{
    /// The response's identity.
    identity: Identity,
    /// Whether a usage chunk ends the stream.
    usage: Usage,
    /// The reasoning streamed so far.
    reasoning: String,
    /// The content streamed so far.
    content: String,
}

impl ChunkStream
{
    /// An encoder for one response.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        identity: Identity,
        usage: Usage,
    ) -> Self
    {
        return Self {
            identity,
            usage,
            reasoning: String::new(),
            content: String::new(),
        };
    }

    /// A chunk with one choice carrying `delta`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: a `chat.completion.chunk` with the choice, `usage` null when
    ///   a usage chunk will follow, and `timings` when given.
    /// - provides: every choice-bearing chunk.
    /// - fails: never.
    /// - panics: none.
    fn chunk(
        &self,
        delta: &Value,
        finish: &Value,
        timings: Value,
    ) -> String
    {
        let mut payload = self.identity.payload(ObjectKind::Chunk);
        payload.insert(
            String::from("choices"),
            json!([{"index": 0_i32, "delta": delta, "logprobs": null, "finish_reason": finish}]),
        );
        if self.usage == Usage::Included {
            payload.insert(String::from("usage"), Value::Null);
        }
        if !timings.is_null() {
            payload.insert(String::from("timings"), timings);
        }
        return event(&Value::Object(payload));
    }

    /// The opening chunk: the assistant role with empty content.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn start(&self) -> String
    {
        return self.chunk(
            &json!({"role": "assistant", "content": ""}),
            &Value::Null,
            Value::Null,
        );
    }

    /// A delta chunk on `channel`.
    ///
    /// # Specification
    /// - requires: reasoning deltas precede content deltas, as the backend
    ///   publishes them.
    /// - ensures: the text is carried as `reasoning_content` or `content`, and
    ///   recorded for [`ChunkStream::finish`].
    /// - provides: the streamed output.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    #[must_use]
    pub fn delta(
        &mut self,
        channel: Channel,
        text: DeltaText<'_>,
    ) -> String
    {
        let (key, streamed) = match channel {
            | Channel::Reasoning => ("reasoning_content", &mut self.reasoning),
            | Channel::Content => ("content", &mut self.content),
        };
        streamed.push_str(text.0);
        let mut delta = serde_json::Map::new();
        delta.insert(String::from(key), Value::from(text.0));
        return self.chunk(&Value::Object(delta), &Value::Null, Value::Null);
    }

    /// The closing chunks.
    ///
    /// # Specification
    /// - requires: every delta of the request was passed to
    ///   [`ChunkStream::delta`].
    /// - ensures: any unstreamed reasoning, then any unstreamed content, each
    ///   as one delta chunk; the tool calls as one chunk when there are any;
    ///   the finish chunk, with `timings` unless a usage chunk follows; the
    ///   usage chunk when requested; and the closing `[DONE]`.
    /// - provides: the end of a streamed response.
    /// - fails: when the streamed text is not a prefix of the outcome's on
    ///   either channel, or reasoning remains after content was streamed.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ApiError`]: an internal error naming the mismatch.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — the remainder, tool-call and usage chunks, and the
    ///   prefix check.
    /// - witness: `tests::a_stream_finishes_with_the_remainder_and_usage`
    #[inline]
    pub fn finish(
        &self,
        outcome: &ChatOutcome,
    ) -> Result<Vec<String>, ApiError>
    {
        let (Some(reasoning), Some(content)) = (
            outcome.reasoning.strip_prefix(self.reasoning.as_str()),
            outcome.content.strip_prefix(self.content.as_str()),
        )
        else {
            return Err(ApiError::internal(String::from(
                "streamed output does not match terminal output",
            )));
        };
        if !reasoning.is_empty() && !self.content.is_empty() {
            return Err(ApiError::internal(String::from(
                "terminal reasoning appeared after streamed content",
            )));
        }
        let mut events = Vec::with_capacity(6);
        if !reasoning.is_empty() {
            events.push(self.chunk(
                &json!({"reasoning_content": reasoning}),
                &Value::Null,
                Value::Null,
            ));
        }
        if !content.is_empty() {
            events.push(self.chunk(&json!({"content": content}), &Value::Null, Value::Null));
        }
        let finish = if outcome.tool_calls.is_empty() {
            finish_reason(outcome.finish)
        }
        else {
            events.push(self.chunk(
                &json!({"tool_calls": tool_calls(&outcome.tool_calls, Indexing::Indexed)}),
                &Value::Null,
                Value::Null,
            ));
            FinishReason::ToolCalls
        };
        let final_timings = timings(outcome);
        match self.usage {
            | Usage::Omitted => {
                events.push(self.chunk(&json!({}), &Value::from(finish), final_timings));
            },
            | Usage::Included => {
                events.push(self.chunk(&json!({}), &Value::from(finish), Value::Null));
                let mut payload = self.identity.payload(ObjectKind::Chunk);
                payload.insert(String::from("choices"), json!([]));
                payload.insert(String::from("usage"), usage(outcome));
                payload.insert(String::from("timings"), final_timings);
                events.push(event(&Value::Object(payload)));
            },
        }
        events.push(String::from(DONE));
        return Ok(events);
    }
}

/// Tests for the rendering.
#[cfg(test)]
mod tests
{
    use infinitum_chat::Admission;
    use infinitum_chat::Channel;
    use infinitum_chat::ChatOutcome;
    use infinitum_chat::DeltaText;
    use infinitum_chat::Finish;
    use infinitum_chat::GeneratedToolCall;
    use infinitum_chat::Tally;
    use infinitum_round::TokenCount;
    use infinitum_round::TokenId;
    use serde_json::Value;

    use super::ChunkStream;
    use super::DONE;
    use super::Identity;
    use super::completion;
    use super::timings;
    use crate::parse::Usage;

    /// The texts and length of a test outcome.
    struct Shape<'text>
    {
        /// The thinking text.
        reasoning: &'text str,
        /// The answer text.
        content: &'text str,
        /// How many tokens were generated.
        generated: usize,
    }

    /// An outcome of the shape's texts and length.
    ///
    /// # Specification
    /// trivial.
    fn outcome(shape: &Shape<'_>) -> ChatOutcome
    {
        return ChatOutcome {
            content: String::from(shape.content),
            reasoning: String::from(shape.reasoning),
            tool_calls: Vec::new(),
            finish: Finish::StopToken,
            admission: Admission {
                prompt_tokens: TokenCount::from(10_u32),
                reused_tokens: TokenCount::from(4_u32),
            },
            generated: vec![TokenId::from(0_i32); shape.generated],
            reasoning_tokens: TokenCount::from(1_u32),
            prompt_wall: core::time::Duration::from_millis(12),
            generation_wall: core::time::Duration::from_millis(40),
            drafted: Tally(0),
            accepted: Tally(0),
        };
    }

    /// Parse one event's payload.
    ///
    /// # Specification
    /// trivial.
    fn payload(event: &Event) -> Value
    {
        return serde_json::from_str(event.0.strip_prefix("data: ").unwrap().trim_end()).unwrap();
    }

    /// One rendered server-sent event.
    #[repr(transparent)]
    struct Event(String);

    #[test]
    fn timings_follow_ninfer()
    {
        let plain = timings(&outcome(&Shape {
            reasoning: "",
            content: "a",
            generated: 5_usize,
        }));
        assert_eq!(plain["cache_n"], 4_i32, "reused tokens are the cache count");
        assert_eq!(plain["prompt_n"], 6_i32, "the rest were prefilled");
        assert_eq!(
            plain["predicted_per_token_ms"], 10.0_f64,
            "40 ms over four decode intervals"
        );
        assert_eq!(
            plain["predicted_per_second"], 100.0_f64,
            "four intervals in 40 ms"
        );
        assert!(
            plain.get("draft_n").is_none(),
            "no draft fields without drafting"
        );
        let mut drafted = outcome(&Shape {
            reasoning: "",
            content: "a",
            generated: 1_usize,
        });
        drafted.drafted = Tally(14);
        drafted.accepted = Tally(9);
        let drafted = timings(&drafted);
        assert_eq!(
            drafted["predicted_per_second"], 0.0_f64,
            "one token has no decode interval"
        );
        assert_eq!(
            drafted["draft_n_accepted"], 9_i32,
            "draft fields appear once something was drafted"
        );
    }

    #[test]
    fn a_completion_carries_ninfers_fields()
    {
        let identity = Identity::new(String::from("m"));
        let plain: Value = serde_json::from_str(&completion(
            &identity,
            &outcome(&Shape {
                reasoning: "r",
                content: "a",
                generated: 3_usize,
            }),
        ))
        .unwrap();
        let choice = &plain["choices"][0];
        assert_eq!(choice["message"]["content"], "a", "the content is carried");
        assert_eq!(
            choice["message"]["reasoning_content"], "r",
            "the reasoning is carried"
        );
        assert_eq!(
            choice["finish_reason"], "stop",
            "a stop token finishes stop"
        );
        assert_eq!(
            plain["usage"]["total_tokens"], 13_i32,
            "prompt and completion sum"
        );
        let mut called = outcome(&Shape {
            reasoning: "",
            content: "",
            generated: 3_usize,
        });
        called.finish = Finish::OutputLimit;
        called.tool_calls.push(GeneratedToolCall {
            name: String::from("ls"),
            arguments: String::from("{}"),
        });
        let called: Value = serde_json::from_str(&completion(&identity, &called)).unwrap();
        let choice = &called["choices"][0];
        assert_eq!(
            choice["message"]["content"],
            Value::Null,
            "empty content beside calls is null"
        );
        assert_eq!(
            choice["finish_reason"], "tool_calls",
            "calls finish tool_calls"
        );
        assert!(
            choice["message"]["tool_calls"][0].get("index").is_none(),
            "aggregate calls are unindexed"
        );
        assert!(
            choice["message"]["tool_calls"][0]["id"]
                .as_str()
                .unwrap()
                .starts_with("call_"),
            "calls take fresh ids"
        );
    }

    #[test]
    fn a_stream_finishes_with_the_remainder_and_usage()
    {
        let mut stream = ChunkStream::new(Identity::new(String::from("m")), Usage::Included);
        assert_eq!(
            payload(&Event(stream.start()))["choices"][0]["delta"]["role"],
            "assistant",
            "the stream opens"
        );
        let reasoning = stream.delta(Channel::Reasoning, DeltaText("thin"));
        assert_eq!(
            payload(&Event(reasoning))["choices"][0]["delta"]["reasoning_content"],
            "thin",
            "a delta rides its channel"
        );
        let terminal = stream
            .finish(&outcome(&Shape {
                reasoning: "think",
                content: "ok",
                generated: 3_usize,
            }))
            .unwrap();
        let payloads: Vec<Value> = terminal
            .iter()
            .take(terminal.len().saturating_sub(1))
            .map(|event| return payload(&Event(event.clone())))
            .collect();
        assert_eq!(
            payloads[0]["choices"][0]["delta"]["reasoning_content"], "k",
            "unstreamed reasoning follows"
        );
        assert_eq!(
            payloads[1]["choices"][0]["delta"]["content"], "ok",
            "unstreamed content follows"
        );
        assert_eq!(
            payloads[2]["choices"][0]["finish_reason"], "stop",
            "the finish chunk"
        );
        assert!(
            payloads[2].get("timings").is_none(),
            "timings move to the usage chunk"
        );
        assert_eq!(
            payloads[3]["usage"]["completion_tokens"], 3_i32,
            "the usage chunk"
        );
        assert_eq!(terminal.last().unwrap(), DONE, "the stream closes");
        let mut diverged = ChunkStream::new(Identity::new(String::from("m")), Usage::Omitted);
        let _content = diverged.delta(Channel::Content, DeltaText("x"));
        assert_eq!(
            diverged
                .finish(&outcome(&Shape {
                    reasoning: "",
                    content: "y",
                    generated: 1_usize
                }))
                .unwrap_err()
                .status
                .as_u16(),
            500_u16,
            "streamed text must prefix the outcome"
        );
    }
}
