//! The ninfer C facade, loaded at run time.
//!
//! The driver reaches ninfer through its C facade: open an artifact, encode
//! text, generate greedily, render ids back to bytes, and close. The facade is
//! a shared library the operator names when a request runs, so building this
//! crate needs no engine on the machine and every line here is checked by the
//! default build.
//!
//! Each foreign prototype is transcribed once, as a function-pointer type
//! below, from the facade's C header. The header is the authority: a change
//! there is a change to these types.

use alloc::ffi::CString;
use core::ffi::CStr;
use core::ffi::c_char;
use core::ffi::c_int;
use core::marker::PhantomData;
use core::marker::PhantomPinned;
use core::num::NonZeroU32;
use core::ptr::NonNull;
use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;

/// Capacity of the buffer each call offers for the facade's failure message.
///
/// A longer message arrives truncated and still NUL-terminated.
const MESSAGE_CAPACITY: Length = Length(1024);

/// The facade's symbol for opening an engine.
const OPEN_SYMBOL: SymbolName = SymbolName("ninfer_engine_open");

/// The facade's symbol for closing an engine.
const CLOSE_SYMBOL: SymbolName = SymbolName("ninfer_engine_close");

/// The facade's symbol for encoding raw text.
const TOKENIZE_SYMBOL: SymbolName = SymbolName("ninfer_tokenize");

/// The facade's symbol for greedy generation.
const GENERATE_SYMBOL: SymbolName = SymbolName("ninfer_generate_greedy");

/// The facade's symbol for rendering ids back to bytes.
const DETOKENIZE_SYMBOL: SymbolName = SymbolName("ninfer_detokenize");

/// A token id in the artifact tokenizer's vocabulary, laid out as the facade's
/// `int32_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(i32);
impl From<i32> for TokenId
{
    /// Wrap an id.
    ///
    /// # Specification
    /// trivial.
    fn from(id: i32) -> Self
    {
        return Self(id);
    }
}

impl core::fmt::Display for TokenId
{
    /// Render the id as its decimal value.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return core::fmt::Display::fmt(&self.0, f);
    }
}

/// A count or capacity, laid out as the facade's `size_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Length(usize);

impl From<usize> for Length
{
    /// Wrap a count.
    ///
    /// # Specification
    /// trivial.
    fn from(count: usize) -> Self
    {
        return Self(count);
    }
}

impl From<Length> for usize
{
    /// Unwrap a count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(length: Length) -> Self
    {
        return length.0;
    }
}

impl core::fmt::Display for Length
{
    /// Render the count as its decimal value.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return core::fmt::Display::fmt(&self.0, f);
    }
}

/// One byte of a C string or byte buffer, laid out as the facade's `char`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
struct CByte(c_char);

/// The positive number of tokens a greedy generation may produce, laid out as
/// the facade's `uint32_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget(NonZeroU32);

impl core::str::FromStr for TokenBudget
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the budget is the parsed value, which is at least
    ///   one.
    /// - provides: the command-line spelling of a generation budget.
    /// - fails: with the integer parser's error on zero, a negative value,
    ///   anything above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary — zero is refused and one is
    ///   admitted — which is the one decision the wrapper adds to `u32`
    ///   parsing.
    /// - witness: `tests::a_zero_budget_is_refused`
    /// - witness: `tests::a_budget_of_one_is_admitted`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let count = text.parse::<NonZeroU32>()?;
        return Ok(Self(count));
    }
}

impl From<TokenBudget> for NonZeroU32
{
    /// Unwrap the budget.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(budget: TokenBudget) -> Self
    {
        return budget.0;
    }
}

impl TryFrom<TokenBudget> for Length
{
    type Error = core::num::TryFromIntError;

    /// Size an output array for a whole generation.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the length equals the budget.
    /// - provides: the first offer for a generation's output.
    /// - fails: where `usize` is narrower than `u32` and the budget exceeds it.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::TryFromIntError`]: the budget does not fit a `usize`.
    fn try_from(budget: TokenBudget) -> Result<Self, Self::Error>
    {
        let count = usize::try_from(budget.0.get())?;
        return Ok(Self(count));
    }
}

/// The logical ceiling of one request in tokens, prompt and generation
/// together, laid out as the facade's `uint32_t`.
///
/// The facade also sizes the KV cache from it, so a small ceiling keeps a
/// one-request run's device footprint small.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLimit(NonZeroU32);

impl core::str::FromStr for ContextLimit
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the limit is the parsed value, which is at least
    ///   one; the facade reads zero as its own default, which this type cannot
    ///   carry.
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
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let count = text.parse::<NonZeroU32>()?;
        return Ok(Self(count));
    }
}

/// A CUDA device ordinal, laid out as the facade's `int32_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOrdinal(i32);

impl core::str::FromStr for DeviceOrdinal
{
    type Err = core::num::ParseIntError;

    /// Parse a non-negative decimal device ordinal.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the ordinal is the parsed value, between zero and
    ///   `u16::MAX`; a negative ordinal names no device and is refused here
    ///   rather than by the facade.
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
    /// - witness: `tests::device_zero_is_admitted`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let ordinal = text.parse::<u16>()?;
        return Ok(Self(i32::from(ordinal)));
    }
}

/// Whether the engine captures its decode rounds as CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CudaGraph
{
    /// Launch every kernel directly.
    Off,
    /// Capture decode rounds as CUDA graphs, as the served engine does.
    On,
}

impl From<CudaGraph> for u8
{
    /// Encode the choice as the facade's `uint8_t` flag.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `Off` encodes as zero and `On` as one, the facade's reading
    ///   of the flag.
    /// - provides: the one place the flag's encoding is written.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over both variants.
    /// - witness: `tests::cuda_graph_encodes_as_the_facade_flag`
    #[inline]
    fn from(choice: CudaGraph) -> Self
    {
        return match choice {
            | CudaGraph::Off => 0,
            | CudaGraph::On => 1,
        };
    }
}

/// Prompt text, encoded by the facade as raw text: no chat template and no
/// special token is added.
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

impl AsRef<str> for Prompt
{
    /// View the text.
    ///
    /// # Specification
    /// trivial.
    fn as_ref(&self) -> &str
    {
        return &self.0;
    }
}

/// Bytes the facade rendered from token ids.
///
/// They are exactly what the ids encode and are not UTF-8 validated: a
/// generation budget can end inside a multi-byte character.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedBytes(Vec<u8>);

impl AsRef<[u8]> for RenderedBytes
{
    /// View the bytes.
    ///
    /// # Specification
    /// trivial.
    fn as_ref(&self) -> &[u8]
    {
        return &self.0;
    }
}

impl From<Vec<u8>> for RenderedBytes
{
    /// Wrap rendered bytes.
    ///
    /// # Specification
    /// trivial.
    fn from(bytes: Vec<u8>) -> Self
    {
        return Self(bytes);
    }
}

/// A facade symbol name, without its terminating NUL.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolName(&'static str);

impl core::fmt::Display for SymbolName
{
    /// Render the symbol as it appears in the library.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(self.0);
    }
}

/// A status word exactly as the facade returns it.
///
/// `ninfer_status` is a C enum whose values are zero to five; the platform C
/// ABIs this crate targets return such an enum as an `int`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawStatus(c_int);

/// The facade's opaque `ninfer_engine`.
///
/// Only pointers to it exist on this side; the marker keeps the type
/// unconstructible, `!Send`, `!Sync` and `!Unpin`, as a foreign object with
/// unknown thread affinity must be.
#[repr(C)]
struct RawEngine
{
    /// Zero-sized storage, so the type has no size this side can rely on.
    _opaque: [u8; 0],
    /// Removes the auto traits the foreign object is not known to have.
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// The facade's `ninfer_engine_options`, field for field.
#[repr(C)]
struct RawOptions
{
    /// NUL-terminated artifact path; required.
    artifact_path: *const CByte,
    /// NUL-terminated chat-template path, or null for the artifact's own.
    chat_template_path: *const CByte,
    /// CUDA device ordinal.
    device: DeviceOrdinal,
    /// Logical ceiling of one request, in tokens.
    max_context: ContextLimit,
    /// KV capacity in tokens; zero follows `max_context`.
    max_kv_tokens: u32,
    /// Concurrent requests admitted.
    max_concurrency: u32,
    /// Nonzero enables CUDA graph capture.
    use_cuda_graph: u8,
}

/// `ninfer_engine_open(options, out, error, error_bytes)`.
type OpenFn =
    unsafe extern "C" fn(*const RawOptions, *mut *mut RawEngine, *mut CByte, Length) -> RawStatus;

/// `ninfer_engine_close(engine)`.
type CloseFn = unsafe extern "C" fn(*mut RawEngine);

/// `ninfer_tokenize(engine, text, out_tokens, capacity, out_count, error,
/// error_bytes)`.
type TokenizeFn = unsafe extern "C" fn(
    *mut RawEngine,
    *const CByte,
    *mut TokenId,
    Length,
    *mut Length,
    *mut CByte,
    Length,
) -> RawStatus;

/// `ninfer_generate_greedy(engine, tokens, token_count, max_new_tokens,
/// out_tokens, capacity, out_count, error, error_bytes)`.
type GenerateFn = unsafe extern "C" fn(
    *mut RawEngine,
    *const TokenId,
    Length,
    TokenBudget,
    *mut TokenId,
    Length,
    *mut Length,
    *mut CByte,
    Length,
) -> RawStatus;

/// `ninfer_detokenize(engine, tokens, token_count, out_bytes, capacity,
/// out_length, error, error_bytes)`.
type DetokenizeFn = unsafe extern "C" fn(
    *mut RawEngine,
    *const TokenId,
    Length,
    *mut CByte,
    Length,
    *mut Length,
    *mut CByte,
    Length,
) -> RawStatus;

/// A facade failure status, one variant per `ninfer_status` failure value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure
{
    /// `NINFER_INVALID_ARGUMENT`: a required pointer was null or a number was
    /// outside its domain.
    InvalidArgument,
    /// `NINFER_NOT_FOUND`: the artifact or a resource it needs was not found.
    NotFound,
    /// `NINFER_BUFFER_TOO_SMALL`: an output array was too small.
    BufferTooSmall,
    /// `NINFER_RUNTIME_ERROR`: the engine rejected the request or failed.
    Runtime,
    /// `NINFER_UNKNOWN_ERROR`: a failure the facade could not interpret.
    Unknown,
    /// A status value the facade header does not define.
    Unrecognized(RawStatus),
}

impl core::fmt::Display for Failure
{
    /// Render the failure as the facade's status name.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::InvalidArgument => f.write_str("NINFER_INVALID_ARGUMENT"),
            | Self::NotFound => f.write_str("NINFER_NOT_FOUND"),
            | Self::BufferTooSmall => f.write_str("NINFER_BUFFER_TOO_SMALL"),
            | Self::Runtime => f.write_str("NINFER_RUNTIME_ERROR"),
            | Self::Unknown => f.write_str("NINFER_UNKNOWN_ERROR"),
            | Self::Unrecognized(RawStatus(value)) => write!(f, "unrecognized status {value}"),
        };
    }
}

/// What a facade call reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome
{
    /// `NINFER_OK`.
    Completed,
    /// Any other status.
    Failed(Failure),
}

impl From<RawStatus> for Outcome
{
    /// Classify a status word.
    ///
    /// # Specification
    /// - requires: nothing; every `int` is classified.
    /// - ensures: zero is `Completed`; one to five are the failure variants in
    ///   the header's order; every other value is `Unrecognized` carrying it.
    /// - provides: the only reading of the facade's status word.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over the six defined values plus the two
    ///   nearest undefined ones, six and minus one, so a swapped arm or a
    ///   collapsed range is distinguished.
    /// - witness: `tests::each_defined_status_is_classified`
    /// - witness: `tests::undefined_statuses_are_kept_as_unrecognized`
    fn from(status: RawStatus) -> Self
    {
        return match status.0 {
            | 0 => Self::Completed,
            | 1 => Self::Failed(Failure::InvalidArgument),
            | 2 => Self::Failed(Failure::NotFound),
            | 3 => Self::Failed(Failure::BufferTooSmall),
            | 4 => Self::Failed(Failure::Runtime),
            | 5 => Self::Failed(Failure::Unknown),
            | _ => Self::Failed(Failure::Unrecognized(status)),
        };
    }
}

/// The facade entry point a failure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation
{
    /// `ninfer_engine_open`.
    Open,
    /// `ninfer_tokenize`.
    Tokenize,
    /// `ninfer_generate_greedy`.
    Generate,
    /// `ninfer_detokenize`.
    Detokenize,
}

impl core::fmt::Display for Operation
{
    /// Render the operation as its entry point's name.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        let symbol = match *self {
            | Self::Open => OPEN_SYMBOL,
            | Self::Tokenize => TOKENIZE_SYMBOL,
            | Self::Generate => GENERATE_SYMBOL,
            | Self::Detokenize => DETOKENIZE_SYMBOL,
        };
        return core::fmt::Display::fmt(&symbol, f);
    }
}

/// A text argument that cannot become a C string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextArgument
{
    /// The artifact path.
    ArtifactPath,
    /// The prompt.
    Prompt,
}

impl core::fmt::Display for TextArgument
{
    /// Render the argument's name.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::ArtifactPath => f.write_str("artifact path"),
            | Self::Prompt => f.write_str("prompt"),
        };
    }
}

/// A failure reaching or driving the facade.
#[derive(Debug)]
pub enum NinferFailure
{
    /// The shared library could not be loaded.
    Load
    {
        /// The path the operator named.
        library: PathBuf,
        /// The platform loader's report.
        cause: libloading::Error,
    },
    /// The library loaded but lacks one of the facade's entry points.
    MissingSymbol
    {
        /// The entry point that was not found.
        symbol: SymbolName,
        /// The platform loader's report.
        cause: libloading::Error,
    },
    /// A text argument contains a NUL byte, which a C string cannot carry.
    InteriorNul
    {
        /// The argument that contains it.
        argument: TextArgument,
    },
    /// An entry point returned a failure status.
    Status
    {
        /// The entry point.
        operation: Operation,
        /// The status it returned.
        failure: Failure,
        /// The facade's message, empty when it wrote none.
        message: String,
    },
    /// An entry point asked for a larger output array twice in a row, so the
    /// length it reports does not settle.
    UnsettledLength
    {
        /// The entry point.
        operation: Operation,
        /// The capacity offered on the second attempt.
        offered: Length,
        /// The length the second attempt asked for.
        requested: Length,
    },
    /// An entry point reported success with a length beyond the array it was
    /// given, which the facade promises never to do.
    LengthBeyondCapacity
    {
        /// The entry point.
        operation: Operation,
        /// The capacity offered.
        capacity: Length,
        /// The length it reported.
        reported: Length,
    },
}

impl core::fmt::Display for NinferFailure
{
    /// Render the failure for an operator: what failed, and the facade's or
    /// the loader's own account of why.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the text names the failed step — the library path, the
    ///   missing symbol, the argument, or the entry point with its status — and
    ///   ends with the underlying cause or message where one exists.
    /// - provides: the line the driver prints on failure.
    /// - fails: only when the formatter does.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on the two renderings an operator acts on: a status
    ///   failure carrying the facade's message, and a load failure naming the
    ///   path.
    /// - witness: `tests::a_status_failure_names_the_entry_point_and_message`
    /// - witness: `tests::a_missing_library_is_a_load_failure`
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::Load {
                ref library,
                ref cause,
            } => {
                let cause = LoaderCause(cause);
                write!(
                    f,
                    "cannot load the ninfer facade `{}`: {cause}",
                    library.display()
                )
            },
            | Self::MissingSymbol { symbol, ref cause } => {
                let cause = LoaderCause(cause);
                write!(
                    f,
                    "the library lacks the ninfer entry point `{symbol}`: {cause}"
                )
            },
            | Self::InteriorNul { argument } => {
                write!(
                    f,
                    "the {argument} contains a NUL byte, which the C facade cannot take"
                )
            },
            | Self::Status {
                operation,
                failure,
                ref message,
            } => {
                if message.is_empty() {
                    write!(f, "{operation} failed with {failure}")
                }
                else {
                    write!(f, "{operation} failed with {failure}: {message}")
                }
            },
            | Self::UnsettledLength {
                operation,
                offered,
                requested,
            } => {
                write!(
                    f,
                    "{operation} asked for {requested} elements after being offered the \
                     {offered} it asked for"
                )
            },
            | Self::LengthBeyondCapacity {
                operation,
                capacity,
                reported,
            } => {
                write!(
                    f,
                    "{operation} reported {reported} elements written into an array of \
                     {capacity}"
                )
            },
        };
    }
}

/// A loader error rendered with the platform's own account of it.
///
/// The loader's error names only the step that failed (`dlopen failed`); the
/// platform's reason, such as a missing file or an unresolved dependency, is
/// its source.
#[repr(transparent)]
struct LoaderCause<'cause>(&'cause libloading::Error);

impl core::fmt::Display for LoaderCause<'_>
{
    /// Render the failed step, then the platform's reason where it gave one.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the text is the loader error's rendering, followed by `: `
    ///   and its source's rendering when it has a source.
    /// - provides: the cause part of a load or symbol failure.
    /// - fails: only when the formatter does.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on the source-present arm, the one a platform loader
    ///   takes for a missing file.
    /// - witness: `tests::a_missing_library_is_a_load_failure`
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        core::fmt::Display::fmt(self.0, f)?;
        if let Some(source) = core::error::Error::source(self.0) {
            write!(f, ": {source}")?;
        }
        return Ok(());
    }
}

/// The buffer a call offers for the facade's failure message.
#[repr(transparent)]
struct MessageBuffer(Vec<u8>);

impl MessageBuffer
{
    /// Allocate a zeroed buffer of [`MESSAGE_CAPACITY`] bytes.
    ///
    /// # Specification
    /// trivial.
    fn new() -> Self
    {
        return Self(vec![0; usize::from(MESSAGE_CAPACITY)]);
    }

    /// The pointer the facade writes the message through.
    ///
    /// # Specification
    /// trivial.
    fn as_mut_ptr(&mut self) -> *mut CByte
    {
        return self.0.as_mut_ptr().cast();
    }

    /// The capacity the facade is told it may write.
    ///
    /// # Specification
    /// trivial.
    fn capacity(&self) -> Length
    {
        return Length(self.0.len());
    }

    /// Read the message the facade left.
    ///
    /// # Specification
    /// - requires: nothing; the facade promises a NUL terminator, but the
    ///   reading does not depend on it.
    /// - ensures: returns the bytes before the first NUL, or the whole buffer
    ///   when there is none, with invalid UTF-8 replaced; an untouched buffer
    ///   reads as empty.
    /// - provides: the facade's own account of a failure.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 over the three buffer shapes — terminated,
    ///   unterminated, untouched — which are the reading's whole decision
    ///   surface.
    /// - witness: `tests::a_terminated_message_stops_at_its_nul`
    /// - witness: `tests::an_unterminated_message_is_read_whole`
    /// - witness: `tests::an_untouched_buffer_reads_as_empty`
    fn message(&self) -> String
    {
        let bytes = match CStr::from_bytes_until_nul(&self.0) {
            | Ok(terminated) => terminated.to_bytes(),
            | Err(_) => self.0.as_slice(),
        };
        return String::from_utf8_lossy(bytes).into_owned();
    }
}

/// What one sized call produced.
enum Attempt<Element>
{
    /// The output fit; the elements the facade wrote.
    Filled(Vec<Element>),
    /// The output did not fit; the length the facade asked for.
    Needs(Length),
}

/// Make one call that writes into a caller-allocated array of `capacity`
/// elements.
///
/// # Specification
/// - requires: `call` passes the pointer, capacity, length slot and message
///   buffer it receives to one facade entry point unchanged, and returns that
///   entry point's status.
/// - ensures: on `NINFER_OK` the returned elements are exactly the first
///   `*out_count` elements written; on `NINFER_BUFFER_TOO_SMALL` the length the
///   facade asked for.
/// - provides: one step of the facade's sizing protocol.
/// - fails: with `Status` on any other failure, carrying the facade's message,
///   and with `LengthBeyondCapacity` when success reports more elements than
///   the array holds.
/// - panics: none.
///
/// # Errors
/// - [`NinferFailure::Status`]: the entry point failed.
/// - [`NinferFailure::LengthBeyondCapacity`]: success reported an impossible
///   length.
///
/// # Adequacy
/// - hypothesis: L3 over the status partition — completed, too small, other
///   failure — and the reported-length boundary at the capacity, exercised
///   through [`fill`] with scripted entry points.
/// - witness: `tests::an_output_that_fits_is_returned_whole`
/// - witness: `tests::a_failure_status_carries_the_message`
/// - witness: `tests::success_beyond_capacity_is_refused`
fn attempt<Element, Call>(
    operation: Operation,
    capacity: Length,
    call: &mut Call,
) -> Result<Attempt<Element>, NinferFailure>
where
    Element: Copy + Default,
    Call: FnMut(*mut Element, Length, &mut Length, &mut MessageBuffer) -> RawStatus,
{
    let mut output = vec![Element::default(); usize::from(capacity)];
    let mut reported = Length(0);
    let mut message = MessageBuffer::new();
    let status = call(output.as_mut_ptr(), capacity, &mut reported, &mut message);
    return match Outcome::from(status) {
        | Outcome::Completed => {
            if reported > capacity {
                return Err(NinferFailure::LengthBeyondCapacity {
                    operation,
                    capacity,
                    reported,
                });
            }
            output.truncate(usize::from(reported));
            Ok(Attempt::Filled(output))
        },
        | Outcome::Failed(Failure::BufferTooSmall) => Ok(Attempt::Needs(reported)),
        | Outcome::Failed(failure) => Err(NinferFailure::Status {
            operation,
            failure,
            message: message.message(),
        }),
    };
}

/// Run the facade's sizing protocol: offer `initial` elements, and if the
/// facade asks for more, offer exactly what it asked for once.
///
/// # Specification
/// - requires: as [`attempt`]; `call` produces the same output length on every
///   invocation with the same inputs, which the facade's determinism promises.
/// - ensures: on success the returned elements are the complete output.
/// - provides: every array-producing facade call, sized without guessing a
///   bound.
/// - fails: as [`attempt`], and with `UnsettledLength` when the second offer is
///   refused as well.
/// - panics: none.
/// - intension: at most two calls are made, so an entry point whose output fits
///   the first offer runs once.
///
/// # Errors
/// - [`NinferFailure::Status`]: an attempt failed.
/// - [`NinferFailure::LengthBeyondCapacity`]: an attempt reported an impossible
///   length.
/// - [`NinferFailure::UnsettledLength`]: the facade refused the length it asked
///   for.
///
/// # Adequacy
/// - hypothesis: L3 over the protocol's three paths — fits first, fits second,
///   refused twice — with the number of calls observed on each, since a skipped
///   retry and an extra call are the plausible faults.
/// - witness: `tests::an_output_that_fits_is_returned_whole`
/// - witness: `tests::a_short_offer_is_retried_with_the_requested_length`
/// - witness: `tests::a_second_refusal_is_unsettled`
fn fill<Element, Call>(
    operation: Operation,
    initial: Length,
    mut call: Call,
) -> Result<Vec<Element>, NinferFailure>
where
    Element: Copy + Default,
    Call: FnMut(*mut Element, Length, &mut Length, &mut MessageBuffer) -> RawStatus,
{
    let first = attempt(operation, initial, &mut call)?;
    let requested = match first {
        | Attempt::Filled(output) => return Ok(output),
        | Attempt::Needs(requested) => requested,
    };
    let second = attempt(operation, requested, &mut call)?;
    return match second {
        | Attempt::Filled(output) => Ok(output),
        | Attempt::Needs(again) => Err(NinferFailure::UnsettledLength {
            operation,
            offered: requested,
            requested: again,
        }),
    };
}

/// Convert text into the NUL-terminated form the facade reads.
///
/// # Specification
/// - requires: nothing; the facade reads paths and text as the platform's byte
///   encoding, which on Unix is the `OsStr`'s own bytes.
/// - ensures: on success the C string holds exactly `text`'s encoded bytes
///   followed by one NUL.
/// - provides: every string argument the facade receives.
/// - fails: with `InteriorNul` naming `argument` when `text` contains a NUL.
/// - panics: none.
///
/// # Errors
/// - [`NinferFailure::InteriorNul`]: `text` contains a NUL byte.
///
/// # Adequacy
/// - hypothesis: L3 on the NUL boundary — a NUL anywhere refused, NUL-free text
///   passed through.
/// - witness: `tests::a_nul_in_the_prompt_is_refused`
fn c_string(
    argument: TextArgument,
    text: &OsStr,
) -> Result<CString, NinferFailure>
{
    return CString::new(text.as_encoded_bytes())
        .map_err(|_nul| return NinferFailure::InteriorNul { argument });
}

/// The loaded facade: its library and the five entry points in it.
///
/// The entry points are plain function pointers copied out of the library;
/// they stay valid exactly as long as `library` stays loaded, which is as long
/// as this value lives. [`Engine`] borrows the facade, so no engine outlives
/// the code it calls.
pub struct Facade
{
    /// `ninfer_engine_open`.
    open: OpenFn,
    /// `ninfer_engine_close`.
    close: CloseFn,
    /// `ninfer_tokenize`.
    tokenize: TokenizeFn,
    /// `ninfer_generate_greedy`.
    generate: GenerateFn,
    /// `ninfer_detokenize`.
    detokenize: DetokenizeFn,
    /// Keeps the entry points mapped; never read, and dropped after every
    /// engine because every engine borrows this value.
    _library: libloading::Library,
}

impl Facade
{
    /// Load the facade from the shared library at `path`.
    ///
    /// # Specification
    /// - requires: nothing checkable; see the unsafe invariants.
    /// - ensures: on success all five entry points are resolved.
    /// - provides: the only way to reach the facade.
    /// - fails: with `Load` when the platform loader cannot load `path`, and
    ///   with `MissingSymbol` naming the first entry point the library lacks.
    /// - panics: none.
    /// - unsafe invariants: the library at `path` is a build of the ninfer C
    ///   facade whose entry points have the prototypes this module transcribes,
    ///   and its initializers are sound to run in this process.
    ///
    /// # Errors
    /// - [`NinferFailure::Load`]: the library could not be loaded.
    /// - [`NinferFailure::MissingSymbol`]: an entry point is absent.
    ///
    /// # Safety
    /// The caller must ensure `path` names the ninfer C facade. Loading runs
    /// the library's initializers, and every later call through the returned
    /// value trusts that each entry point has the prototype transcribed here;
    /// neither can be checked from this side of the boundary. A library that
    /// lacks an entry point is refused, but one that exports the names with
    /// other prototypes is undefined behavior.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on the two failure steps — no library, and a library
    ///   without the facade — each asserted by variant and payload. Success
    ///   needs the facade library and is witnessed outside the suite by the
    ///   device smoke.
    /// - witness: `tests::a_missing_library_is_a_load_failure`
    /// - witness: `tests::a_library_without_the_facade_lacks_its_entry_points`
    pub unsafe fn load(path: &Path) -> Result<Self, NinferFailure>
    {
        // SAFETY: the caller guarantees `path` names the facade, whose
        // initializers are sound to run here.
        let loaded = unsafe { libloading::Library::new(path) };
        let library = loaded.map_err(|cause| {
            return NinferFailure::Load {
                library: path.to_path_buf(),
                cause,
            };
        })?;
        // SAFETY: the caller guarantees the library is the facade, and each
        // lookup here pairs a name with the type transcribed from its
        // prototype in the facade header; this one, `ninfer_engine_open`.
        let open = unsafe { symbol::<OpenFn>(&library, OPEN_SYMBOL) }?;
        // SAFETY: as above, for `ninfer_engine_close`.
        let close = unsafe { symbol::<CloseFn>(&library, CLOSE_SYMBOL) }?;
        // SAFETY: as above, for `ninfer_tokenize`.
        let tokenize = unsafe { symbol::<TokenizeFn>(&library, TOKENIZE_SYMBOL) }?;
        // SAFETY: as above, for `ninfer_generate_greedy`.
        let generate = unsafe { symbol::<GenerateFn>(&library, GENERATE_SYMBOL) }?;
        // SAFETY: as above, for `ninfer_detokenize`.
        let detokenize = unsafe { symbol::<DetokenizeFn>(&library, DETOKENIZE_SYMBOL) }?;
        return Ok(Self {
            open,
            close,
            tokenize,
            generate,
            detokenize,
            _library: library,
        });
    }

    /// Open an engine over one artifact.
    ///
    /// # Specification
    /// - requires: nothing beyond [`Facade::load`]'s requirement.
    /// - ensures: on success the engine holds the handle the facade produced,
    ///   admits one concurrent request, sizes its KV cache from the context
    ///   limit, and uses the artifact's own chat template.
    /// - provides: the engine a request runs on.
    /// - fails: with `InteriorNul` when the artifact path contains a NUL, and
    ///   with `Status` for the facade's own failures, among them a missing
    ///   artifact and an absent device.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`NinferFailure::InteriorNul`]: the path contains a NUL byte.
    /// - [`NinferFailure::Status`]: `ninfer_engine_open` failed.
    ///
    /// # Adequacy
    /// - hypothesis: none in the suite; opening needs the facade library and,
    ///   for success, a device. The device smoke witnesses success, and a run
    ///   with no visible device witnesses the failure path.
    pub fn open(
        &self,
        options: &EngineOptions,
    ) -> Result<Engine<'_>, NinferFailure>
    {
        let artifact = c_string(TextArgument::ArtifactPath, options.artifact.as_os_str())?;
        let settings = RawOptions {
            artifact_path: artifact.as_ptr().cast(),
            chat_template_path: core::ptr::null(),
            device: options.device,
            max_context: options.context,
            max_kv_tokens: 0,
            max_concurrency: 1,
            use_cuda_graph: u8::from(options.cuda_graph),
        };
        let mut handle: *mut RawEngine = core::ptr::null_mut();
        let mut message = MessageBuffer::new();
        let capacity = message.capacity();
        // SAFETY: `settings` and the artifact string it points into live until
        // the call returns; `handle` is a valid slot for the out pointer; the
        // message pointer is valid for `capacity` bytes. The prototype matches
        // `ninfer_engine_open`.
        let status = unsafe {
            (self.open)(
                &raw const settings,
                &raw mut handle,
                message.as_mut_ptr(),
                capacity,
            )
        };
        if let Outcome::Failed(failure) = Outcome::from(status) {
            return Err(NinferFailure::Status {
                operation: Operation::Open,
                failure,
                message: message.message(),
            });
        }
        return match NonNull::new(handle) {
            | Some(handle) => Ok(Engine {
                facade: self,
                handle,
            }),
            | None => Err(NinferFailure::Status {
                operation: Operation::Open,
                failure: Failure::Unknown,
                message: String::from("reported success without producing a handle"),
            }),
        };
    }
}

impl core::fmt::Debug for Facade
{
    /// Render the facade without its function pointers.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.debug_struct("Facade").finish_non_exhaustive();
    }
}

/// Resolve one entry point from the facade library.
///
/// # Specification
/// - requires: nothing checkable; see the unsafe invariants.
/// - ensures: on success the pointer is the library's `name`, valid while
///   `library` stays loaded.
/// - provides: each of [`Facade`]'s entry points.
/// - fails: with `MissingSymbol` naming `name` when the library lacks it.
/// - panics: none.
/// - unsafe invariants: `Entry` is the function-pointer type transcribing
///   `name`'s C prototype in the library.
///
/// # Errors
/// - [`NinferFailure::MissingSymbol`]: the library has no such symbol.
///
/// # Safety
/// The caller must pair `name` with the type of the symbol it names; the
/// loader returns an address, and reading it at another type is undefined
/// behavior.
///
/// # Adequacy
/// - hypothesis: L3 on the absent-symbol step, the one decision this function
///   makes.
/// - witness: `tests::a_library_without_the_facade_lacks_its_entry_points`
unsafe fn symbol<Entry>(
    library: &libloading::Library,
    name: SymbolName,
) -> Result<Entry, NinferFailure>
where
    Entry: Copy,
{
    // SAFETY: the caller pairs `name` with its own type.
    //
    // The name is unpacked by hand: libloading's symbol-name trait is sealed,
    // so no trait implementation of ours can hand it over.
    let found = unsafe { library.get::<Entry>(name.0) };
    let found = found.map_err(|cause| {
        return NinferFailure::MissingSymbol {
            symbol: name,
            cause,
        };
    })?;
    return Ok(*found);
}

/// How to open an engine.
#[derive(Debug, Clone)]
pub struct EngineOptions
{
    /// The `.ninfer` artifact entry file.
    artifact: PathBuf,
    /// Logical ceiling of the request, which also sizes the KV cache.
    context: ContextLimit,
    /// The CUDA device.
    device: DeviceOrdinal,
    /// Whether decode rounds are captured as CUDA graphs.
    cuda_graph: CudaGraph,
}

impl EngineOptions
{
    /// Gather the options for one engine.
    ///
    /// # Specification
    /// trivial.
    pub const fn new(
        artifact: PathBuf,
        context: ContextLimit,
        device: DeviceOrdinal,
        cuda_graph: CudaGraph,
    ) -> Self
    {
        return Self {
            artifact,
            context,
            device,
            cuda_graph,
        };
    }
}

/// An open engine; dropping it closes the handle.
///
/// The handle is a foreign object with no promise of thread safety, so the
/// type is neither `Send` nor `Sync`, and calls on it are serial.
#[derive(Debug)]
pub struct Engine<'facade>
{
    /// The facade whose entry points drive this handle.
    facade: &'facade Facade,
    /// The handle `ninfer_engine_open` produced; closed exactly once, on drop.
    handle: NonNull<RawEngine>,
}

impl Engine<'_>
{
    /// Encode a prompt with the artifact's tokenizer.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the ids are the tokenizer's raw-text encoding of
    ///   the prompt, with no chat template or special token added.
    /// - provides: the prefix a generation starts from.
    /// - fails: with `InteriorNul` when the prompt contains a NUL, and with the
    ///   facade's own failures.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`NinferFailure::InteriorNul`]: the prompt contains a NUL byte.
    /// - [`NinferFailure::Status`], [`NinferFailure::UnsettledLength`],
    ///   [`NinferFailure::LengthBeyondCapacity`]: from `ninfer_tokenize`
    ///   through [`fill`].
    ///
    /// # Adequacy
    /// - hypothesis: the NUL refusal is L3; the encoding itself needs the
    ///   facade and a device and is witnessed by the device smoke.
    /// - witness: `tests::a_nul_in_the_prompt_is_refused`
    pub fn tokenize(
        &self,
        prompt: &Prompt,
    ) -> Result<Vec<TokenId>, NinferFailure>
    {
        let prompt: &str = prompt.as_ref();
        let text = c_string(TextArgument::Prompt, OsStr::new(prompt))?;
        // economy: a byte-level tokenizer yields at most one id per byte, so
        // the prompt's length is offered first; a tokenizer that yields more
        // costs one extra encode.
        let initial = Length(prompt.len());
        return fill(
            Operation::Tokenize,
            initial,
            |output, capacity, reported, message| {
                let message_capacity = message.capacity();
                // SAFETY: `text` is NUL-terminated and outlives the call;
                // `output` is valid for `capacity` ids and `reported` for one
                // length; the message pointer is valid for its capacity; the
                // handle is open. The prototype matches `ninfer_tokenize`.
                return unsafe {
                    (self.facade.tokenize)(
                        self.handle.as_ptr(),
                        text.as_ptr().cast(),
                        output,
                        capacity,
                        reported,
                        message.as_mut_ptr(),
                        message_capacity,
                    )
                };
            },
        );
    }

    /// Generate greedily from a prefix.
    ///
    /// # Specification
    /// - requires: nothing; an empty prefix is the facade's to refuse.
    /// - ensures: on success the ids are the greedy continuation of `prefix`,
    ///   at most `budget` of them, ending early at a model stop token.
    /// - provides: the generation the driver prints.
    /// - fails: with the facade's own failures, among them an empty prefix and
    ///   a prefix plus budget beyond the context limit.
    /// - panics: none.
    /// - intension: the output array is sized to the budget, so the facade
    ///   generates once.
    ///
    /// # Errors
    /// - [`NinferFailure::Status`], [`NinferFailure::UnsettledLength`],
    ///   [`NinferFailure::LengthBeyondCapacity`]: from `ninfer_generate_greedy`
    ///   through [`fill`].
    ///
    /// # Adequacy
    /// - hypothesis: none in the suite; generation needs the facade and a
    ///   device. The device smoke witnesses it, and its determinism is the
    ///   oracle: two runs of one prompt print the same ids.
    pub fn generate_greedy(
        &self,
        prefix: &[TokenId],
        budget: TokenBudget,
    ) -> Result<Vec<TokenId>, NinferFailure>
    {
        // economy: where `usize` is narrower than `u32` the budget may not
        // fit; offering nothing then costs one extra generation, and the
        // facade reports the exact length for the second offer.
        let initial = Length::try_from(budget).unwrap_or_default();
        let prefix_length = Length(prefix.len());
        return fill(
            Operation::Generate,
            initial,
            |output, capacity, reported, message| {
                let message_capacity = message.capacity();
                // SAFETY: `prefix` is valid for `prefix_length` ids and
                // outlives the call; `output` is valid for `capacity` ids and
                // `reported` for one length; the message pointer is valid for
                // its capacity; the handle is open. The prototype matches
                // `ninfer_generate_greedy`.
                return unsafe {
                    (self.facade.generate)(
                        self.handle.as_ptr(),
                        prefix.as_ptr(),
                        prefix_length,
                        budget,
                        output,
                        capacity,
                        reported,
                        message.as_mut_ptr(),
                        message_capacity,
                    )
                };
            },
        );
    }

    /// Render ids as the bytes the artifact's tokenizer gives them.
    ///
    /// # Specification
    /// - requires: nothing; an id outside the vocabulary is the facade's to
    ///   refuse.
    /// - ensures: on success the bytes are each id's tokenizer bytes in order,
    ///   special tokens included, not UTF-8 validated.
    /// - provides: the generated text the driver prints.
    /// - fails: with the facade's own failures, among them an id outside the
    ///   vocabulary.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`NinferFailure::Status`], [`NinferFailure::UnsettledLength`],
    ///   [`NinferFailure::LengthBeyondCapacity`]: from `ninfer_detokenize`
    ///   through [`fill`].
    ///
    /// # Adequacy
    /// - hypothesis: none in the suite; rendering needs an open engine, which
    ///   needs a device. The device smoke witnesses it.
    pub fn detokenize(
        &self,
        ids: &[TokenId],
    ) -> Result<RenderedBytes, NinferFailure>
    {
        let count = Length(ids.len());
        // economy: rendering is cheap, so the length is asked for rather than
        // estimated, and the second call writes it.
        let bytes = fill(
            Operation::Detokenize,
            Length(0),
            |output: *mut u8, capacity, reported, message| {
                let message_capacity = message.capacity();
                // SAFETY: `ids` is valid for `count` ids and outlives the call;
                // `output` is valid for `capacity` bytes and `reported` for one
                // length; the message pointer is valid for its capacity; the
                // handle is open. The prototype matches `ninfer_detokenize`.
                return unsafe {
                    (self.facade.detokenize)(
                        self.handle.as_ptr(),
                        ids.as_ptr(),
                        count,
                        output.cast(),
                        capacity,
                        reported,
                        message.as_mut_ptr(),
                        message_capacity,
                    )
                };
            },
        )?;
        return Ok(RenderedBytes(bytes));
    }
}

impl Drop for Engine<'_>
{
    /// Close the handle.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the facade has released the engine, once.
    /// - provides: the only release of a handle.
    /// - fails: never; `ninfer_engine_close` reports nothing.
    /// - panics: none.
    fn drop(&mut self)
    {
        // SAFETY: the handle came from a successful open and this is its one
        // release; the prototype matches `ninfer_engine_close`.
        unsafe { (self.facade.close)(self.handle.as_ptr()) };
    }
}

/// Tests for the status vocabulary, the argument domains, and the sizing
/// protocol.
#[cfg(test)]
mod tests
{
    use core::str::FromStr as _;
    use std::ffi::OsStr;
    use std::path::Path;

    use super::CudaGraph;
    use super::DeviceOrdinal;
    use super::Facade;
    use super::Failure;
    use super::Length;
    use super::MessageBuffer;
    use super::NinferFailure;
    use super::Operation;
    use super::Outcome;
    use super::RawStatus;
    use super::SymbolName;
    use super::TextArgument;
    use super::TokenBudget;
    use super::c_string;
    use super::fill;

    /// A library every supported platform carries that is not the facade.
    #[cfg(target_os = "linux")]
    const SYSTEM_LIBRARY: &str = "libc.so.6";
    /// A library every supported platform carries that is not the facade.
    #[cfg(target_os = "macos")]
    const SYSTEM_LIBRARY: &str = "/usr/lib/libSystem.B.dylib";
    /// A library every supported platform carries that is not the facade.
    #[cfg(windows)]
    const SYSTEM_LIBRARY: &str = "kernel32.dll";

    /// Each value the header defines maps to its own variant.
    #[test]
    fn each_defined_status_is_classified()
    {
        let expected = [
            (0_i32, Outcome::Completed),
            (1_i32, Outcome::Failed(Failure::InvalidArgument)),
            (2_i32, Outcome::Failed(Failure::NotFound)),
            (3_i32, Outcome::Failed(Failure::BufferTooSmall)),
            (4_i32, Outcome::Failed(Failure::Runtime)),
            (5_i32, Outcome::Failed(Failure::Unknown)),
        ];
        for (value, outcome) in expected {
            assert_eq!(
                Outcome::from(RawStatus(value)),
                outcome,
                "status {value} is classified as the header defines it"
            );
        }
    }

    /// A value the header does not define is kept, not folded into another.
    #[test]
    fn undefined_statuses_are_kept_as_unrecognized()
    {
        for value in [-1_i32, 6_i32] {
            assert_eq!(
                Outcome::from(RawStatus(value)),
                Outcome::Failed(Failure::Unrecognized(RawStatus(value))),
                "status {value} is unrecognized and carries its value"
            );
        }
    }

    /// The message stops at the first NUL.
    #[test]
    fn a_terminated_message_stops_at_its_nul()
    {
        let mut buffer = MessageBuffer::new();
        buffer.0[.. 6].copy_from_slice(b"no gpu");
        buffer.0[7] = b'x';
        assert_eq!(
            buffer.message(),
            "no gpu",
            "the bytes after the NUL are ignored"
        );
    }

    /// A buffer with no NUL is read whole.
    #[test]
    fn an_unterminated_message_is_read_whole()
    {
        let buffer = MessageBuffer(b"abc".to_vec());
        assert_eq!(buffer.message(), "abc", "every byte is read");
    }

    /// A buffer the facade never wrote reads as empty.
    #[test]
    fn an_untouched_buffer_reads_as_empty()
    {
        assert_eq!(
            MessageBuffer::new().message(),
            "",
            "an untouched buffer is empty"
        );
    }

    /// Zero tokens is not a budget.
    #[test]
    fn a_zero_budget_is_refused()
    {
        assert!(TokenBudget::from_str("0").is_err(), "zero is refused");
    }

    /// One token is the smallest budget.
    #[test]
    fn a_budget_of_one_is_admitted()
    {
        let budget = TokenBudget::from_str("1").unwrap();
        assert_eq!(
            u32::from(core::num::NonZeroU32::from(budget)),
            1,
            "one is kept"
        );
    }

    /// Zero tokens is not a context.
    #[test]
    fn a_zero_context_is_refused()
    {
        assert!(
            super::ContextLimit::from_str("0").is_err(),
            "zero is refused"
        );
    }

    /// A negative ordinal names no device.
    #[test]
    fn a_negative_device_is_refused()
    {
        assert!(
            DeviceOrdinal::from_str("-1").is_err(),
            "minus one is refused"
        );
    }

    /// Device zero is the first device.
    #[test]
    fn device_zero_is_admitted()
    {
        assert_eq!(
            DeviceOrdinal::from_str("0").unwrap(),
            DeviceOrdinal(0),
            "zero is kept"
        );
    }

    /// The flag encodes as the facade reads it.
    #[test]
    fn cuda_graph_encodes_as_the_facade_flag()
    {
        assert_eq!(u8::from(CudaGraph::Off), 0, "off is zero");
        assert_eq!(u8::from(CudaGraph::On), 1, "on is one");
    }

    /// A NUL cannot cross as part of a C string.
    #[test]
    fn a_nul_in_the_prompt_is_refused()
    {
        match c_string(TextArgument::Prompt, OsStr::new("a\0b")) {
            | Err(NinferFailure::InteriorNul { argument }) => {
                assert_eq!(
                    argument,
                    TextArgument::Prompt,
                    "the refusal names the prompt"
                );
            },
            | other => panic!("a NUL must be refused, got {other:?}"),
        }
        let passed = c_string(TextArgument::Prompt, OsStr::new("ab")).unwrap();
        assert_eq!(
            passed.as_bytes_with_nul(),
            b"ab\0",
            "NUL-free text gains one NUL"
        );
    }

    /// An output that fits the first offer is returned from one call.
    #[test]
    fn an_output_that_fits_is_returned_whole()
    {
        let mut calls = 0_u32;
        let output = fill(
            Operation::Tokenize,
            Length(4),
            |output: *mut u8, capacity, reported, _| {
                calls += 1;
                assert_eq!(capacity, Length(4), "the first offer is the initial length");
                // SAFETY: `output` is valid for the four bytes offered.
                unsafe { output.copy_from_nonoverlapping(b"hi".as_ptr(), 2) };
                *reported = Length(2);
                return RawStatus(0);
            },
        )
        .unwrap();
        assert_eq!(output, b"hi", "exactly the reported elements are returned");
        assert_eq!(calls, 1, "a fitting output takes one call");
    }

    /// A short offer is retried once, with the length the facade asked for.
    #[test]
    fn a_short_offer_is_retried_with_the_requested_length()
    {
        let mut offers = Vec::new();
        let output = fill(
            Operation::Detokenize,
            Length(0),
            |output: *mut u8, capacity, reported, _| {
                offers.push(capacity);
                *reported = Length(3);
                if capacity < Length(3) {
                    return RawStatus(3);
                }
                // SAFETY: `output` is valid for the three bytes offered.
                unsafe { output.copy_from_nonoverlapping(b"abc".as_ptr(), 3) };
                return RawStatus(0);
            },
        )
        .unwrap();
        assert_eq!(output, b"abc", "the second attempt's output is returned");
        assert_eq!(
            offers,
            [Length(0), Length(3)],
            "the retry offers exactly the requested length"
        );
    }

    /// A facade that refuses the length it asked for is reported, not retried
    /// again.
    #[test]
    fn a_second_refusal_is_unsettled()
    {
        let mut calls = 0_u32;
        let refused = fill(
            Operation::Generate,
            Length(1),
            |_: *mut u8, capacity, reported, _| {
                calls += 1;
                *reported = Length(usize::from(capacity).saturating_add(1));
                return RawStatus(3);
            },
        );
        match refused {
            | Err(NinferFailure::UnsettledLength {
                operation,
                offered,
                requested,
            }) => {
                assert_eq!(operation, Operation::Generate, "the entry point is named");
                assert_eq!(offered, Length(2), "the second offer is the first request");
                assert_eq!(requested, Length(3), "the second request is kept");
            },
            | other => panic!("a second refusal must be unsettled, got {other:?}"),
        }
        assert_eq!(calls, 2, "the protocol stops after two calls");
    }

    /// A failure status stops the protocol and carries the facade's message.
    #[test]
    fn a_failure_status_carries_the_message()
    {
        let failed = fill(
            Operation::Tokenize,
            Length(1),
            |_: *mut u8, _, _, message| {
                message.0[.. 4].copy_from_slice(b"bad!");
                return RawStatus(1);
            },
        );
        match failed {
            | Err(NinferFailure::Status {
                operation,
                failure,
                message,
            }) => {
                assert_eq!(operation, Operation::Tokenize, "the entry point is named");
                assert_eq!(failure, Failure::InvalidArgument, "the status is kept");
                assert_eq!(message, "bad!", "the facade's message is kept");
            },
            | other => panic!("a failure status must be reported, got {other:?}"),
        }
    }

    /// Success claiming more elements than were offered is refused rather
    /// than read past the array.
    #[test]
    fn success_beyond_capacity_is_refused()
    {
        let claimed = fill(
            Operation::Tokenize,
            Length(2),
            |_: *mut u8, _, reported, _| {
                *reported = Length(3);
                return RawStatus(0);
            },
        );
        match claimed {
            | Err(NinferFailure::LengthBeyondCapacity {
                capacity, reported, ..
            }) => {
                assert_eq!(capacity, Length(2), "the offered capacity is kept");
                assert_eq!(reported, Length(3), "the impossible length is kept");
            },
            | other => panic!("an impossible length must be refused, got {other:?}"),
        }
    }

    /// The rendering an operator reads names the entry point, the status and
    /// the facade's message.
    #[test]
    fn a_status_failure_names_the_entry_point_and_message()
    {
        let failure = NinferFailure::Status {
            operation: Operation::Open,
            failure: Failure::Runtime,
            message: String::from("no device"),
        };
        assert_eq!(
            failure.to_string(),
            "ninfer_engine_open failed with NINFER_RUNTIME_ERROR: no device",
            "the rendering names what failed and why"
        );
    }

    /// A path with no library behind it fails at the load step, naming the
    /// path and carrying the platform's reason.
    #[test]
    fn a_missing_library_is_a_load_failure()
    {
        let path = Path::new("no-such-directory/libninfer_capi.so");
        // SAFETY: nothing exists at the path, so nothing is loaded or run.
        let failure = unsafe { Facade::load(path) }.unwrap_err();
        let rendered = failure.to_string();
        match failure {
            | NinferFailure::Load {
                ref library,
                ref cause,
            } => {
                assert_eq!(library, path, "the failure names the path");
                let reason = core::error::Error::source(cause)
                    .expect("the platform loader gives a reason for a missing file");
                assert_eq!(
                    rendered,
                    format!(
                        "cannot load the ninfer facade `no-such-directory/libninfer_capi.so`: \
                         {cause}: {reason}"
                    ),
                    "the rendering names the path, the failed step and the platform's reason"
                );
            },
            | other => panic!("a missing library must fail to load, got {other:?}"),
        }
    }

    /// A library that is not the facade fails at its first entry point.
    #[test]
    fn a_library_without_the_facade_lacks_its_entry_points()
    {
        // SAFETY: the system C library is already loaded into every test
        // process, so loading it again runs no initializer, and the lookup
        // stops at the first facade name it lacks without reading any symbol.
        match unsafe { Facade::load(Path::new(SYSTEM_LIBRARY)) } {
            | Err(NinferFailure::MissingSymbol { symbol, .. }) => {
                assert_eq!(
                    symbol,
                    SymbolName("ninfer_engine_open"),
                    "the first entry point looked up is the one reported"
                );
            },
            | other => panic!("a foreign library must lack the facade, got {other:?}"),
        }
    }
}
