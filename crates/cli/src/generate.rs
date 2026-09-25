//! One greedy request, end to end, through the ninfer C facade.
//!
//! The request encodes a prompt with the artifact's tokenizer, generates
//! greedily, renders the generated ids back to bytes, and prints the ids of
//! both sides beside the text. The facade's greedy path is deterministic, so
//! the printed ids of two runs of one request are the oracle an
//! implementation of the same model is compared against.

use std::io::Write;
use std::path::PathBuf;

use crate::ninfer::ContextLimit;
use crate::ninfer::CudaGraph;
use crate::ninfer::DeviceOrdinal;
use crate::ninfer::EngineOptions;
use crate::ninfer::Facade;
use crate::ninfer::NinferFailure;
use crate::ninfer::Prompt;
use crate::ninfer::RenderedBytes;
use crate::ninfer::TokenBudget;
use crate::ninfer::TokenId;

/// Run one prompt through ninfer's C facade: tokenize, generate greedily, and
/// print the ids and the generated text.
#[derive(Debug, Clone, clap::Args)]
pub struct Request
{
    /// The ninfer C facade shared library to load (`libninfer_capi.so`). It
    /// must be that facade: loading a library runs its code.
    #[arg(long, value_name = "PATH")]
    library: PathBuf,
    /// The `.ninfer` artifact to open.
    #[arg(long, value_name = "PATH")]
    artifact: PathBuf,
    /// The most tokens to generate; generation also ends at a model stop
    /// token.
    #[arg(long, value_name = "TOKENS", default_value = "32")]
    max_new_tokens: TokenBudget,
    /// The request's context ceiling in tokens, prompt and generation
    /// together; it also sizes the KV cache.
    #[arg(long, value_name = "TOKENS", default_value = "4096")]
    max_context: ContextLimit,
    /// The CUDA device ordinal.
    #[arg(long, value_name = "ORDINAL", default_value = "0")]
    device: DeviceOrdinal,
    /// Whether decode rounds are captured as CUDA graphs.
    #[arg(long, value_enum, default_value_t = CudaGraph::Off)]
    cuda_graph: CudaGraph,
    /// The prompt, encoded as raw text: no chat template and no special token
    /// is added.
    prompt: Prompt,
}

/// A finished request: the prompt's ids, the generated ids, and the bytes the
/// generated ids render as.
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
    /// Loading or driving the facade failed.
    Ninfer(NinferFailure),
    /// Writing the result failed.
    Output(std::io::Error),
}

impl From<NinferFailure> for RequestFailure
{
    /// Wrap a facade failure.
    ///
    /// # Specification
    /// trivial.
    fn from(failure: NinferFailure) -> Self
    {
        return Self::Ninfer(failure);
    }
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
            | Self::Ninfer(ref failure) => core::fmt::Display::fmt(failure, f),
            | Self::Output(ref failure) => write!(f, "cannot write the result: {failure}"),
        };
    }
}

/// Run a request and print its completion to `out`.
///
/// # Specification
/// - requires: `out` accepts bytes, and `request`'s library is the ninfer C
///   facade. The driver cannot check the second from this side of the boundary,
///   so it forwards the operator's claim to [`Facade::load`].
/// - ensures: on success `out` holds [`render`]'s rendering of the completion
///   and nothing else; the engine is closed and the library unloaded before
///   this returns.
/// - provides: the driver's one engine invocation.
/// - fails: with the first facade failure, in the order load, open, tokenize,
///   generate, detokenize, and nothing is written then; or with the writer's
///   error.
/// - panics: none.
///
/// # Errors
/// - [`RequestFailure::Ninfer`]: a facade step failed.
/// - [`RequestFailure::Output`]: writing to `out` failed.
///
/// # Adequacy
/// - hypothesis: the load step is L3 in the suite, a missing library being the
///   first failure reached; every later step needs the facade and a device and
///   is witnessed by the device smoke, whose repeated run is the determinism
///   oracle.
/// - witness: `tests::a_request_without_a_library_fails_to_load`
pub fn run<Writer>(
    request: &Request,
    out: &mut Writer,
) -> Result<(), RequestFailure>
where
    Writer: Write,
{
    // SAFETY: the operator names this library as the ninfer C facade through
    // `--library`, and the driver cannot check that claim from this side of
    // the boundary; the command's documentation states the requirement.
    let facade = unsafe { Facade::load(&request.library) }?;
    let options = EngineOptions::new(
        request.artifact.clone(),
        request.max_context,
        request.device,
        request.cuda_graph,
    );
    let completion = complete(&facade, &options, request)?;
    render(&completion, out)?;
    return Ok(());
}

/// Drive one engine through tokenize, generate, and detokenize.
///
/// # Specification
/// - requires: nothing beyond [`Facade::open`]'s requirement.
/// - ensures: on success the completion's prompt ids are the prompt's encoding,
///   its generated ids the greedy continuation of them within the budget, and
///   its text those ids' rendering; the engine is closed before this returns,
///   on success and on failure.
/// - provides: the completion [`run`] prints.
/// - fails: with the first failing facade step.
/// - panics: none.
///
/// # Errors
/// - [`NinferFailure`]: open, tokenize, generate or detokenize failed.
///
/// # Adequacy
/// - hypothesis: none in the suite; every step needs the facade and a device.
///   The device smoke witnesses the whole path.
fn complete(
    facade: &Facade,
    options: &EngineOptions,
    request: &Request,
) -> Result<Completion, NinferFailure>
{
    let engine = facade.open(options)?;
    let prompt = engine.tokenize(&request.prompt)?;
    let generated = engine.generate_greedy(&prompt, request.max_new_tokens)?;
    let text = engine.detokenize(&generated)?;
    return Ok(Completion {
        prompt,
        generated,
        text,
    });
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdsLabel
{
    /// The prompt's encoding.
    Prompt,
    /// The generation.
    Generated,
}

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

/// Tests for the request's rendering and its first failure.
#[cfg(test)]
mod tests
{
    use super::Completion;
    use super::Request;
    use super::RequestFailure;
    use super::render;
    use super::run;
    use crate::ninfer::NinferFailure;
    use crate::ninfer::RenderedBytes;
    use crate::ninfer::TokenId;

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
            text: RenderedBytes::from(Vec::new()),
        };
        render(&completion, &mut out).unwrap();
        assert_eq!(
            out, b"prompt ids: 1\ngenerated ids:\ngenerated text:\n\n",
            "an empty list writes its label alone"
        );
    }

    /// Without a library the request fails at the load step and writes
    /// nothing.
    #[test]
    fn a_request_without_a_library_fails_to_load()
    {
        let request = <Request as clap::FromArgMatches>::from_arg_matches(
            &<Request as clap::Args>::augment_args(clap::Command::new("generate"))
                .try_get_matches_from([
                    "generate",
                    "--library",
                    "no-such-directory/libninfer_capi.so",
                    "--artifact",
                    "model.ninfer",
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
                Err(RequestFailure::Ninfer(NinferFailure::Load { .. }))
            ),
            "the first failure is the load: {failed:?}"
        );
        assert!(out.is_empty(), "a failed request writes nothing");
    }
}
