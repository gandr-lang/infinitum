//! One greedy request, end to end, through infinitum's DFlash2 round on
//! ninfer.
//!
//! The request composes the DFlash2 round graph at the requested draft width
//! and has the ninfer backend plan it; a graph ninfer cannot run is refused
//! before anything opens. With the `ninfer` feature the plan then opens
//! ninfer's Engine, encodes the prompt, generates greedily with infinitum's
//! round preview deciding each round's output, renders the generated ids, and
//! prints the ids of both sides beside the text and the round tallies. Greedy
//! generation is deterministic, so the ids lines of two runs of one request
//! are the oracle an implementation of the same model is compared against.

use std::io::Write;
use std::path::PathBuf;

use infinitum_ninfer::ContextLimit;
use infinitum_ninfer::CudaGraph;
use infinitum_ninfer::DFlash2Plan;
use infinitum_ninfer::DeviceOrdinal;
use infinitum_ninfer::Ninfer;
#[cfg(any(feature = "ninfer", test))]
use infinitum_ninfer::RenderedBytes;
use infinitum_round::Backend as _;
use infinitum_round::BuildFailure;
use infinitum_round::DraftWidth;
use infinitum_round::Refusal;
#[cfg(any(feature = "ninfer", test))]
use infinitum_round::TokenId;

/// The positive number of tokens a greedy generation may produce.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget(core::num::NonZeroU32);

impl core::str::FromStr for TokenBudget
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the budget is the parsed value, at least one.
    /// - provides: the command-line spelling of a generation budget.
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
    /// - witness: `crate::tests::a_zero_budget_is_an_argument_error`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

/// Prompt text, encoded as raw text: no chat template and no special token is
/// added.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt(String);

impl core::str::FromStr for Prompt
{
    type Err = core::convert::Infallible;

    /// Take the text as given.
    ///
    /// # Specification
    /// trivial.
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return Ok(Self(String::from(text)));
    }
}

/// Run one prompt through infinitum's DFlash2 round on ninfer: plan, open,
/// tokenize, generate greedily, and print the ids, the text, and the round
/// tallies.
#[derive(Debug, Clone, clap::Args)]
pub struct Request
{
    /// The `.ninfer` artifact to open; it must carry a DFlash2 companion.
    #[arg(long, value_name = "PATH")]
    artifact: PathBuf,
    /// The most tokens to generate; generation also ends at a model stop
    /// token.
    #[arg(long, value_name = "TOKENS", default_value = "32")]
    max_new_tokens: TokenBudget,
    /// DFlash2's draft width `K`: tokens drafted per round, verified in
    /// `K + 1` columns. ninfer runs 1 to 15.
    #[arg(long, value_name = "TOKENS", default_value = "7")]
    draft_width: DraftWidth,
    /// The request's context ceiling in tokens, prompt and generation
    /// together; it also sizes the KV cache.
    #[arg(long, value_name = "TOKENS", default_value = "4096")]
    max_context: ContextLimit,
    /// The CUDA device ordinal.
    #[arg(long, value_name = "ORDINAL", default_value = "0")]
    device: DeviceOrdinal,
    /// Whether decode rounds are captured as CUDA graphs: `off` or `on`.
    #[arg(long, value_name = "CHOICE", default_value = "off")]
    cuda_graph: CudaGraph,
    /// The prompt, encoded as raw text: no chat template and no special token
    /// is added.
    prompt: Prompt,
}

/// A finished request: the prompt's ids, the generated ids, and the bytes the
/// generated ids render as.
#[cfg(any(feature = "ninfer", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion
{
    /// The tokenizer's encoding of the prompt.
    prompt: Vec<TokenId>,
    /// The greedy continuation.
    generated: Vec<TokenId>,
    /// The generated ids rendered as bytes.
    text: RenderedBytes,
}

/// A failure of a request.
#[derive(Debug)]
pub enum RequestFailure
{
    /// The DFlash2 round could not be composed.
    Compose(BuildFailure),
    /// ninfer refused the round.
    Plan(Refusal),
    /// This build has no ninfer Engine: it was built without the `ninfer`
    /// feature.
    #[cfg(not(feature = "ninfer"))]
    NoEngine(DFlash2Plan),
    /// Opening or driving ninfer's Engine failed.
    #[cfg(feature = "ninfer")]
    Engine(infinitum_ninfer::EngineFailure),
    /// Writing the result failed.
    Output(std::io::Error),
}

impl From<std::io::Error> for RequestFailure
{
    /// Wrap an output failure.
    ///
    /// # Specification
    /// trivial.
    fn from(failure: std::io::Error) -> Self
    {
        return Self::Output(failure);
    }
}

#[cfg(feature = "ninfer")]
impl From<infinitum_ninfer::EngineFailure> for RequestFailure
{
    /// Wrap an Engine failure.
    ///
    /// # Specification
    /// trivial.
    fn from(failure: infinitum_ninfer::EngineFailure) -> Self
    {
        return Self::Engine(failure);
    }
}

impl core::fmt::Display for RequestFailure
{
    /// Render the failure through its own rendering.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::Compose(ref failure) => {
                write!(f, "cannot compose the DFlash2 round: {failure}")
            },
            | Self::Plan(ref refusal) => core::fmt::Display::fmt(refusal, f),
            #[cfg(not(feature = "ninfer"))]
            | Self::NoEngine(ref plan) => write!(
                f,
                "ninfer plans the DFlash2 round at draft width {}, but this build has no ninfer \
                 engine; rebuild with `--features ninfer`",
                plan.width()
            ),
            #[cfg(feature = "ninfer")]
            | Self::Engine(ref failure) => core::fmt::Display::fmt(failure, f),
            | Self::Output(ref failure) => write!(f, "cannot write the result: {failure}"),
        };
    }
}

/// Plan the request's round on ninfer.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the plan runs the canonical DFlash2 round at the
///   request's draft width.
/// - provides: the gate every request passes before an Engine opens.
/// - fails: when the round cannot be composed or ninfer refuses it.
/// - panics: none.
///
/// # Errors
/// - [`RequestFailure::Compose`], [`RequestFailure::Plan`]: as named.
///
/// # Adequacy
/// - hypothesis: L3 on the refusal, a width past ninfer's range.
/// - witness: `tests::a_width_past_ninfers_range_is_refused_before_anything_opens`
fn plan(request: &Request) -> Result<DFlash2Plan, RequestFailure>
{
    let graph = infinitum_round::dflash2(request.draft_width).map_err(RequestFailure::Compose)?;
    return Ninfer.plan(&graph).map_err(RequestFailure::Plan);
}

/// Run a request and print its completion to `out`.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: on success `out` holds `render`'s rendering of the completion
///   followed by the round tallies line, and nothing else; the Engine is closed
///   before this returns.
/// - provides: the driver's one engine invocation.
/// - fails: with the first failure, in the order plan, open, tokenize,
///   generate, detokenize, and nothing is written then; without the `ninfer`
///   feature, with [`RequestFailure::NoEngine`] after a successful plan; or
///   with the writer's error.
/// - panics: none.
///
/// # Errors
/// - [`RequestFailure`]: as ordered above.
///
/// # Adequacy
/// - hypothesis: the plan step is L3 in the suite; every later step needs
///   ninfer and a device and is witnessed by the device acceptance run, whose
///   generated ids match ninfer's own DFlash2 greedy run.
/// - witness: `tests::a_width_past_ninfers_range_is_refused_before_anything_opens`
pub fn run<Writer>(
    request: &Request,
    out: &mut Writer,
) -> Result<(), RequestFailure>
where
    Writer: Write,
{
    let plan = plan(request)?;
    return execute(request, plan, out);
}

/// Without an Engine, a planned request goes no further.
///
/// # Specification
/// - requires: nothing.
/// - ensures: nothing is written.
/// - provides: the explicit end of a build without the `ninfer` feature.
/// - fails: always, with [`RequestFailure::NoEngine`].
/// - panics: none.
///
/// # Errors
/// - [`RequestFailure::NoEngine`]: this build has no Engine.
#[cfg(not(feature = "ninfer"))]
const fn execute<Writer>(
    _request: &Request,
    plan: DFlash2Plan,
    _out: &mut Writer,
) -> Result<(), RequestFailure>
where
    Writer: Write,
{
    return Err(RequestFailure::NoEngine(plan));
}

/// Open ninfer's Engine for the plan and run the request on it.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: on success the completion, then one `rounds:` line with ninfer's
///   speculative rounds, drafted and accepted tokens, fallback steps, the
///   generation wall time in microseconds, the number of rounds infinitum's
///   preview reviewed, and the finish reason.
/// - provides: the engine half of [`run`].
/// - fails: with the first failing Engine step, before anything is written; or
///   with the writer's error.
/// - panics: none.
///
/// # Errors
/// - [`RequestFailure::Engine`], [`RequestFailure::Output`]: as named.
///
/// # Adequacy
/// - hypothesis: L2 outside the suite — every step needs ninfer and a device,
///   and the device acceptance run checks the ids against ninfer's own DFlash2
///   greedy run.
#[cfg(feature = "ninfer")]
fn execute<Writer>(
    request: &Request,
    plan: DFlash2Plan,
    out: &mut Writer,
) -> Result<(), RequestFailure>
where
    Writer: Write,
{
    let options = infinitum_ninfer::EngineOptions::new(
        request.artifact.clone(),
        request.device,
        request.max_context,
        request.cuda_graph,
    );
    let mut session =
        infinitum_ninfer::Session::open(&options, plan, &mut infinitum_chat::Unobserved)?;
    let prompt = session.tokenize(infinitum_ninfer::RawText::from(request.prompt.0.as_str()))?;
    let generation = session.generate(&prompt, request.max_new_tokens.0)?;
    let text = session.detokenize(generation.generated())?;
    drop(session);
    let completion = Completion {
        prompt,
        generated: generation.generated().to_vec(),
        text,
    };
    render(&completion, out)?;
    let speculation = generation.speculation();
    writeln!(
        out,
        "rounds: {} drafted: {} accepted: {} fallback steps: {} wall us: {} reviewed rounds: {} \
         finish: {}",
        speculation.rounds(),
        speculation.drafted(),
        speculation.accepted(),
        speculation.fallback_steps(),
        generation.wall().as_micros(),
        generation.rounds().len(),
        generation.finish(),
    )?;
    return Ok(());
}

/// Write one line of ids after a label.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: `out` gains `label`, a colon, each id preceded by one space, and
///   a newline; an empty list writes the label and colon alone.
/// - provides: the ids lines two runs are compared by.
/// - fails: with the writer's error.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: writing failed.
///
/// # Adequacy
/// - hypothesis: L3 on the separator and the empty list, through [`render`]'s
///   exact output.
/// - witness: `tests::a_completion_renders_ids_then_bytes`
/// - witness: `tests::an_empty_generation_renders_empty_lines`
#[cfg(any(feature = "ninfer", test))]
fn render_ids<Writer>(
    label: IdsLabel,
    ids: &[TokenId],
    out: &mut Writer,
) -> Result<(), std::io::Error>
where
    Writer: Write,
{
    write!(out, "{label}:")?;
    for id in ids {
        write!(out, " {id}")?;
    }
    writeln!(out)?;
    return Ok(());
}

/// The label of an ids line.
#[cfg(any(feature = "ninfer", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdsLabel
{
    /// The prompt's encoding.
    Prompt,
    /// The generation.
    Generated,
}

#[cfg(any(feature = "ninfer", test))]
impl core::fmt::Display for IdsLabel
{
    /// Render the label as printed.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::Prompt => f.write_str("prompt ids"),
            | Self::Generated => f.write_str("generated ids"),
        };
    }
}

/// Write a completion: the prompt ids, the generated ids, and the generated
/// bytes.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: `out` gains exactly `prompt ids:` and the prompt's ids on one
///   line, `generated ids:` and the generated ids on the next, a `generated
///   text:` line, the rendered bytes unchanged, and a newline.
/// - provides: the driver's output for a request; the ids lines are
///   byte-comparable between runs, and the bytes are not re-encoded, so a
///   multi-byte character a budget cut short arrives as the bytes it is.
/// - fails: with the writer's error.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: writing failed.
///
/// # Adequacy
/// - hypothesis: L3 — the exact output for an ordinary completion carrying a
///   byte that is not UTF-8, and for an empty generation.
/// - witness: `tests::a_completion_renders_ids_then_bytes`
/// - witness: `tests::an_empty_generation_renders_empty_lines`
#[cfg(any(feature = "ninfer", test))]
fn render<Writer>(
    completion: &Completion,
    out: &mut Writer,
) -> Result<(), std::io::Error>
where
    Writer: Write,
{
    render_ids(IdsLabel::Prompt, &completion.prompt, out)?;
    render_ids(IdsLabel::Generated, &completion.generated, out)?;
    writeln!(out, "generated text:")?;
    out.write_all(completion.text.as_ref())?;
    writeln!(out)?;
    return Ok(());
}

/// Tests for the request's rendering and its planning gate.
#[cfg(test)]
mod tests
{
    use infinitum_round::RefusalReason;
    use infinitum_round::TokenId;

    use super::Completion;
    use super::RenderedBytes;
    use super::Request;
    use super::RequestFailure;
    use super::render;
    use super::run;

    /// The rendering is the ids lines, then the bytes unchanged.
    #[test]
    fn a_completion_renders_ids_then_bytes()
    {
        let mut out = Vec::new();
        let completion = Completion {
            prompt: [9707_i32, 11_i32].map(TokenId::from).to_vec(),
            generated: [1879_i32, 0_i32, 13_i32].map(TokenId::from).to_vec(),
            text: RenderedBytes::from(b"world\xe4".to_vec()),
        };
        render(&completion, &mut out).unwrap();
        assert_eq!(
            out, b"prompt ids: 9707 11\ngenerated ids: 1879 0 13\ngenerated text:\nworld\xe4\n",
            "ids are space-separated and the bytes pass through unchanged"
        );
    }

    /// An empty generation still renders every line.
    #[test]
    fn an_empty_generation_renders_empty_lines()
    {
        let mut out = Vec::new();
        let completion = Completion {
            prompt: vec![TokenId::from(1_i32)],
            generated: Vec::new(),
            text: RenderedBytes::default(),
        };
        render(&completion, &mut out).unwrap();
        assert_eq!(
            out, b"prompt ids: 1\ngenerated ids:\ngenerated text:\n\n",
            "an empty list writes its label alone"
        );
    }

    /// A draft width ninfer cannot run is refused at planning, before any
    /// artifact is opened, and nothing is written.
    #[test]
    fn a_width_past_ninfers_range_is_refused_before_anything_opens()
    {
        let request = <Request as clap::FromArgMatches>::from_arg_matches(
            &<Request as clap::Args>::augment_args(clap::Command::new("generate"))
                .try_get_matches_from([
                    "generate",
                    "--artifact",
                    "no-such-directory/model.ninfer",
                    "--draft-width",
                    "16",
                    "hello",
                ])
                .unwrap(),
        )
        .unwrap();
        let mut out = Vec::new();
        let failed = run(&request, &mut out);
        assert!(
            matches!(
                failed,
                Err(RequestFailure::Plan(ref refusal))
                    if matches!(refusal.reason(), RefusalReason::UnsupportedWidth(_))
            ),
            "the refusal is the width: {failed:?}"
        );
        assert!(out.is_empty(), "a refused request writes nothing");
    }
}
