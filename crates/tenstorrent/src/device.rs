//! The C++ host behind safe types: build, lower, load, open, and run.

use core::num::NonZeroU32;

use infinitum_round::DraftWidth;
use infinitum_round::Maybe;
use infinitum_round::TokenCount;
use infinitum_round::TokenId;

use crate::accept::Acceptance;
use crate::accept::Bf16;
use crate::accept::DeviceBits;
use crate::accept::DraftBlock;
use crate::accept::ShapeMismatch;
use crate::accept::VerifyLogits;
use crate::accept::VocabularyLayout;
use crate::bridge::ffi;
use crate::digits::Digit;
use crate::digits::DigitFailure;
use crate::digits::IdDigits;
use crate::digits::compose;
use crate::digits::constant_planes;
use crate::digits::fill_tail;

/// Where the accepted-length prefix runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixSite
{
    /// On the device: the program returns the licensed ids and the length.
    Device,
    /// On the host: the program returns the target ids and the host scans
    /// them.
    Host,
}

/// The host operation a failure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation
{
    /// Printing the builder's module.
    Print,
    /// Building and lowering in process.
    Compile,
    /// Loading a flatbuffer.
    Load,
    /// Opening the device.
    Open,
    /// Writing the system descriptor.
    SystemDesc,
    /// Running the program.
    Run,
    /// Probing a binary's round trip.
    Probe,
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
            | Self::Print => "printing the module",
            | Self::Compile => "lowering the module",
            | Self::Load => "loading the flatbuffer",
            | Self::Open => "opening the device",
            | Self::SystemDesc => "writing the system descriptor",
            | Self::Run => "running the program",
            | Self::Probe => "probing the binary",
        });
    }
}

/// A failure in the host or of the device's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostFailure
{
    /// The host reported a failure.
    Host
    {
        /// What was being done.
        operation: Operation,
        /// The host's status, rendered.
        status: String,
        /// Its message.
        message: String,
    },
    /// A path is not Unicode, which the bridge's string cannot carry.
    Path(std::path::PathBuf),
    /// Inputs or outputs of the wrong shape.
    Shape(ShapeMismatch),
    /// A draft, or the device's answer, is not a digit of a vocabulary id.
    Digits(DigitFailure),
    /// A licensed id past the accepted length is not zero.
    NonzeroTail,
}

impl core::fmt::Display for HostFailure
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
            | Self::Host {
                operation,
                ref status,
                ref message,
            } => write!(f, "the host failed {operation} ({status}): {message}"),
            | Self::Path(ref path) => write!(f, "the path {} is not Unicode", path.display()),
            | Self::Shape(ref mismatch) => {
                write!(f, "the device's answer has the wrong shape: {mismatch}")
            },
            | Self::Digits(ref failure) => write!(f, "{failure}"),
            | Self::NonzeroTail => {
                f.write_str("the device licensed an id past its accepted length")
            },
        };
    }
}

impl core::error::Error for HostFailure
{
}

impl From<DigitFailure> for HostFailure
{
    /// Wrap a digit failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(failure: DigitFailure) -> Self
    {
        return Self::Digits(failure);
    }
}

impl From<ShapeMismatch> for HostFailure
{
    /// Wrap a shape mismatch.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(mismatch: ShapeMismatch) -> Self
    {
        return Self::Shape(mismatch);
    }
}

/// A fresh outcome for one call.
///
/// # Specification
/// trivial.
const fn outcome() -> ffi::Outcome
{
    return ffi::Outcome {
        status: ffi::Status::Completed,
        message: String::new(),
    };
}

/// Turn `outcome` into a result for `operation`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `Ok` exactly when the status is `Completed`.
/// - provides: the one mapping from the host's outcome to a failure.
/// - fails: with [`HostFailure::Host`] otherwise.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Host`]: the host reported a failure.
fn checked(
    operation: Operation,
    outcome: ffi::Outcome,
) -> Result<(), HostFailure>
{
    if outcome.status == ffi::Status::Completed {
        return Ok(());
    }
    return Err(HostFailure::Host {
        operation,
        status: format!("{:?}", outcome.status),
        message: outcome.message,
    });
}

/// The bridge's name for a path.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the path's text when it is Unicode.
/// - provides: the `&str` the bridge takes.
/// - fails: with [`HostFailure::Path`] otherwise.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Path`]: the path is not Unicode.
fn path_text(path: &std::path::Path) -> Result<&str, HostFailure>
{
    return path
        .to_str()
        .ok_or_else(|| return HostFailure::Path(path.to_path_buf()));
}

/// The shapes a program is lowered for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptGeometry
{
    /// The draft width.
    width: DraftWidth,
    /// The vocabulary layout.
    layout: VocabularyLayout,
    /// Where the prefix runs.
    prefix: PrefixSite,
}

impl AcceptGeometry
{
    /// The geometry for `width` drafts over `layout`, with the prefix at
    /// `prefix`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        width: DraftWidth,
        layout: VocabularyLayout,
        prefix: PrefixSite,
    ) -> Self
    {
        return Self {
            width,
            layout,
            prefix,
        };
    }

    /// Where the prefix runs.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn prefix(&self) -> PrefixSite
    {
        return self.prefix;
    }

    /// The wire form.
    ///
    /// # Specification
    /// trivial.
    fn wire(&self) -> ffi::AcceptGeometry
    {
        return ffi::AcceptGeometry {
            drafts: NonZeroU32::from(self.width).get(),
            vocabulary: u32::from(self.layout.size()),
            chunk: u32::from(self.layout.chunk()),
            prefix: match self.prefix {
                | PrefixSite::Device => ffi::PrefixSite::Device,
                | PrefixSite::Host => ffi::PrefixSite::Host,
            },
        };
    }
}

/// A duration in nanoseconds, as the host measured it.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Nanos(u64);

impl From<Nanos> for u64
{
    /// Unwrap the count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(nanos: Nanos) -> Self
    {
        return nanos.0;
    }
}

impl core::fmt::Display for Nanos
{
    /// Render in microseconds with one decimal.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        let whole = self.0.checked_div(1000).unwrap_or_default();
        let tenth = self
            .0
            .checked_rem(1000)
            .unwrap_or_default()
            .checked_div(100)
            .unwrap_or_default();
        return write!(f, "{whole}.{tenth} us");
    }
}

/// What an in-process lowering cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileCost
{
    /// Building and verifying the module.
    pub build: Nanos,
    /// The pipeline.
    pub pipeline: Nanos,
    /// The flatbuffer translation.
    pub translate: Nanos,
    /// Device programs enqueued per run.
    pub programs: TokenCount,
}

/// What one run cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunCost
{
    /// Submission to completion.
    pub submit: Nanos,
    /// Reading the outputs back.
    pub readback: Nanos,
}

/// A lowered Accept fragment, the geometry it was lowered for, and the
/// constant planes it reads on every call.
pub struct AcceptProgram
{
    /// The host's program.
    program: cxx::UniquePtr<ffi::Program>,
    /// Its geometry.
    geometry: AcceptGeometry,
    /// The pad, local, hi, and lo planes, as bfloat16 bits.
    planes: DeviceBits,
}

impl core::fmt::Debug for AcceptProgram
{
    /// Render the geometry.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f
            .debug_struct("AcceptProgram")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive();
    }
}

/// Print the builder's module for `geometry` as TTIR.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the text of the verified module.
/// - provides: the builder's module, to compare with the hand-written sample.
/// - fails: when the host refuses the geometry or verification fails.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Host`]: the host failed.
#[inline]
pub fn print_accept(geometry: &AcceptGeometry) -> Result<String, HostFailure>
{
    let mut text = String::new();
    let mut result = outcome();
    ffi::print_accept(&geometry.wire(), &mut text, &mut result);
    checked(Operation::Print, result)?;
    return Ok(text);
}

/// The system descriptor lowering compiles against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemDescriptor<'path>
{
    /// A descriptor written for the device the program runs on.
    Saved(&'path std::path::Path),
    /// tt-mlir's mock single-chip Blackhole: the program compiles without a
    /// device, but may not match the one it would run on.
    MockBlackhole,
}

impl AcceptProgram
{
    /// Build and lower the module for `geometry` in process against
    /// `system_desc`.
    ///
    /// # Specification
    /// - requires: a saved `system_desc` was written for the device the program
    ///   runs on.
    /// - ensures: on success the program and what each stage cost.
    /// - provides: the in-process route.
    /// - fails: when the host fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`HostFailure::Path`]: the path is not Unicode.
    /// - [`HostFailure::Host`]: the host failed.
    #[inline]
    pub fn compile(
        geometry: AcceptGeometry,
        system_desc: SystemDescriptor<'_>,
    ) -> Result<(Self, CompileCost), HostFailure>
    {
        // The host reads an empty path as the mock descriptor.
        let path = match system_desc {
            | SystemDescriptor::Saved(path) => path_text(path)?,
            | SystemDescriptor::MockBlackhole => "",
        };
        let mut record = ffi::CompileRecord::default();
        let mut result = outcome();
        let program = ffi::compile_accept(&geometry.wire(), path, &mut record, &mut result);
        checked(Operation::Compile, result)?;
        let cost = CompileCost {
            build: Nanos(record.build_ns),
            pipeline: Nanos(record.pipeline_ns),
            translate: Nanos(record.translate_ns),
            programs: TokenCount::from(record.programs),
        };
        let planes = constant_planes(geometry.layout, geometry.width)?;
        return Ok((
            Self {
                program,
                geometry,
                planes,
            },
            cost,
        ));
    }

    /// Load the flatbuffer at `path`, lowered for `geometry`.
    ///
    /// # Specification
    /// - requires: the flatbuffer was lowered for `geometry`.
    /// - ensures: on success the program.
    /// - provides: the route for a module the pipeline tools lowered.
    /// - fails: when the host fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`HostFailure::Path`]: the path is not Unicode.
    /// - [`HostFailure::Host`]: the host failed.
    #[inline]
    pub fn load(
        geometry: AcceptGeometry,
        path: &std::path::Path,
    ) -> Result<Self, HostFailure>
    {
        let text = path_text(path)?;
        let mut result = outcome();
        let program = ffi::load_program(text, &mut result);
        checked(Operation::Load, result)?;
        let planes = constant_planes(geometry.layout, geometry.width)?;
        return Ok(Self {
            program,
            geometry,
            planes,
        });
    }
}

/// Write the current system descriptor to `path`.
///
/// # Specification
/// - requires: a device is visible and none is open in this process.
/// - ensures: on success the file holds the descriptor.
/// - provides: the descriptor lowering compiles against.
/// - fails: when the host fails.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Path`]: the path is not Unicode.
/// - [`HostFailure::Host`]: the host failed.
#[inline]
pub fn save_system_desc(path: &std::path::Path) -> Result<(), HostFailure>
{
    let text = path_text(path)?;
    let mut result = outcome();
    ffi::save_system_desc(text, &mut result);
    return checked(Operation::SystemDesc, result);
}

/// One stage's per-call timings.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTimes(Vec<Nanos>);

impl StageTimes
{
    /// Wrap the host's counts, sorted.
    ///
    /// # Specification
    /// trivial.
    fn sorted(counts: Vec<u64>) -> Self
    {
        let mut times: Vec<Nanos> = counts.into_iter().map(Nanos).collect();
        times.sort_unstable();
        return Self(times);
    }

    /// The fastest call, the median call, and the slowest call.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: [`Maybe::Present`] with the three when any call was timed.
    /// - provides: the summary the probe report prints.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    #[must_use]
    pub fn summary(&self) -> Maybe<[Nanos; 3], NoCalls>
    {
        let middle = self.0.len().checked_div(2).unwrap_or_default();
        return match (self.0.first(), self.0.get(middle), self.0.last()) {
            | (Some(&least), Some(&median), Some(&most)) => Maybe::Present([least, median, most]),
            | _ => Maybe::Absent(NoCalls),
        };
    }
}

/// Why a stage has no summary: no call was timed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoCalls;

/// A binary's round trip, stage by stage, over zero inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeReport
{
    /// Moving the inputs into the program's layout on every call.
    pub per_call_move: StageTimes,
    /// Submission to completion, inputs moved on every call.
    pub per_call_submit: StageTimes,
    /// Reading back, inputs moved on every call.
    pub per_call_readback: StageTimes,
    /// Submission to completion, inputs moved once.
    pub staged_submit: StageTimes,
    /// Reading back, inputs moved once.
    pub staged_readback: StageTimes,
    /// Output bytes read back over every call of both modes.
    pub bytes_read: u64,
}

/// Load the flatbuffer at `path`, open the device for its runtime, and time
/// `calls` round trips over zero inputs in each input mode.
///
/// # Specification
/// - requires: a device is visible and none is open in this process.
/// - ensures: on success every stage holds `calls` timings.
/// - provides: the dispatch floor of a binary for either runtime.
/// - fails: when the host fails.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Path`]: the path is not Unicode.
/// - [`HostFailure::Host`]: the host failed.
#[inline]
pub fn probe_binary(
    path: &std::path::Path,
    calls: ProbeCalls,
) -> Result<ProbeReport, HostFailure>
{
    let text = path_text(path)?;
    let mut result = outcome();
    let program = ffi::load_program(text, &mut result);
    checked(Operation::Load, result)?;
    let mut result = outcome();
    let mut device = ffi::open_device(&program, &mut result);
    checked(Operation::Open, result)?;
    let mut output = ffi::ProbeOutput::default();
    let mut result = outcome();
    if let Some(pinned) = device.as_mut() {
        ffi::probe_program(pinned, &program, calls.0.get(), &mut output, &mut result);
    }
    checked(Operation::Probe, result)?;
    return Ok(ProbeReport {
        per_call_move: StageTimes::sorted(output.per_call_move),
        per_call_submit: StageTimes::sorted(output.per_call_submit),
        per_call_readback: StageTimes::sorted(output.per_call_readback),
        staged_submit: StageTimes::sorted(output.staged_submit),
        staged_readback: StageTimes::sorted(output.staged_readback),
        bytes_read: output.bytes_read,
    });
}

/// How many round trips a probe times in each mode.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeCalls(NonZeroU32);

impl From<NonZeroU32> for ProbeCalls
{
    /// Wrap a count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(calls: NonZeroU32) -> Self
    {
        return Self(calls);
    }
}

/// An open device and the buffers its calls reuse.
pub struct TenstorrentDevice
{
    /// The host's device.
    device: cxx::UniquePtr<ffi::Device>,
    /// The per-call drafts, positions, fill, and zeros.
    tail: DeviceBits,
    /// The last run's outputs and cost.
    output: ffi::RunOutput,
}

impl core::fmt::Debug for TenstorrentDevice
{
    /// Render the type name.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.debug_struct("TenstorrentDevice").finish_non_exhaustive();
    }
}

/// What the device answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceAnswer
{
    /// The whole fragment ran on the device.
    Acceptance(Acceptance),
    /// The device returned the targets; the host applied the prefix.
    Targets
    {
        /// The target ids.
        targets: Vec<TokenId>,
        /// The acceptance the host derived from them.
        acceptance: Acceptance,
    },
}

impl DeviceAnswer
{
    /// The acceptance, wherever the prefix ran.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn acceptance(&self) -> &Acceptance
    {
        return match *self {
            | Self::Acceptance(ref acceptance) | Self::Targets { ref acceptance, .. } => acceptance,
        };
    }
}

/// The device's digits for one column, read from the three output planes.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the digits at `column` of each plane, when all three exist.
/// - provides: one target's digits.
/// - fails: with [`HostFailure::Shape`] when a plane is short, and with
///   [`HostFailure::Digits`] when a value is not a digit.
/// - panics: none.
///
/// # Errors
/// - [`HostFailure::Shape`]: an output plane is short.
/// - [`HostFailure::Digits`]: a value is not a digit.
fn column_digits(
    planes: [&[u16]; 3],
    column: usize,
) -> Result<IdDigits, HostFailure>
{
    let [hi, lo, local] = planes.map(|plane| {
        let bits = plane
            .get(column)
            .copied()
            .ok_or(HostFailure::Shape(ShapeMismatch::Logits))?;
        return Digit::try_from(Bf16::from(bits)).map_err(HostFailure::Digits);
    });
    return Ok(IdDigits {
        hi: hi?,
        lo: lo?,
        local: local?,
    });
}

impl TenstorrentDevice
{
    /// Open the visible device for `program`'s runtime.
    ///
    /// # Specification
    /// - requires: a device is visible to the process.
    /// - ensures: on success the device; dropping it closes it.
    /// - provides: the device runs submit to.
    /// - fails: when the host fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`HostFailure::Host`]: the host failed.
    #[inline]
    pub fn open(program: &AcceptProgram) -> Result<Self, HostFailure>
    {
        let mut result = outcome();
        let device = ffi::open_device(&program.program, &mut result);
        checked(Operation::Open, result)?;
        return Ok(Self {
            device,
            tail: DeviceBits::default(),
            output: ffi::RunOutput::default(),
        });
    }

    /// Run `program` once over `logits` and `drafts`.
    ///
    /// # Specification
    /// - requires: `logits` and `drafts` follow the program's geometry.
    /// - ensures: on success the device's acceptance (with the host's prefix,
    ///   for a host-prefix program) and the run's cost; each target id is
    ///   composed from the device's three digits; targets past the accepted
    ///   length, which the device zeroes, are checked and dropped.
    /// - provides: one device acceptance.
    /// - fails: when a draft is not a vocabulary id, the host fails, the
    ///   outputs have the wrong length, a value is not a digit, or a zeroed
    ///   target is not zero.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`HostFailure::Host`]: the host failed.
    /// - [`HostFailure::Shape`]: the outputs or drafts have the wrong length.
    /// - [`HostFailure::Digits`]: a draft or an output is not a digit of a
    ///   vocabulary id.
    /// - [`HostFailure::NonzeroTail`]: a target past the accepted length is not
    ///   zero.
    #[inline]
    pub fn run(
        &mut self,
        program: &AcceptProgram,
        logits: &VerifyLogits,
        drafts: &DraftBlock,
    ) -> Result<(DeviceAnswer, RunCost), HostFailure>
    {
        let layout = program.geometry.layout;
        let on_device = program.geometry.prefix == PrefixSite::Device;
        if on_device {
            fill_tail(layout, drafts, &mut self.tail)?;
        }
        else {
            self.tail.clear();
        }
        let mut result = outcome();
        let Some(device) = self.device.as_mut()
        else {
            return Err(HostFailure::Host {
                operation: Operation::Run,
                status: String::from("Closed"),
                message: String::from("the device is closed"),
            });
        };
        ffi::run_accept(
            device,
            &program.program,
            logits.bits().as_ref(),
            program.planes.as_ref(),
            self.tail.as_ref(),
            &mut self.output,
            &mut result,
        );
        checked(Operation::Run, result)?;
        let cost = RunCost {
            submit: Nanos(self.output.submit_ns),
            readback: Nanos(self.output.readback_ns),
        };
        let columns = drafts.ids().len().saturating_add(1);
        let expected = columns
            .saturating_mul(3)
            .saturating_add(usize::from(on_device));
        if self.output.values.len() != expected {
            return Err(HostFailure::Shape(ShapeMismatch::Logits));
        }
        let (hi, rest) = self.output.values.split_at(columns);
        let (lo, rest) = rest.split_at(columns);
        let (local, rest) = rest.split_at(columns);
        let planes = [hi, lo, local];
        let answer = if on_device {
            let length = rest
                .first()
                .copied()
                .ok_or(HostFailure::Shape(ShapeMismatch::Logits))?;
            let accepted = Digit::try_from(Bf16::from(length))?;
            let accepted = usize::from(u16::from(accepted));
            if accepted >= columns {
                return Err(HostFailure::Shape(ShapeMismatch::Drafts));
            }
            let mut licensed = Vec::with_capacity(accepted.saturating_add(1));
            for column in 0 .. columns {
                let digits = column_digits(planes, column)?;
                if column <= accepted {
                    licensed.push(compose(layout, digits)?);
                }
                else if digits != IdDigits::ZERO {
                    // The fragment zeroes the targets past the accepted length.
                    return Err(HostFailure::NonzeroTail);
                }
            }
            DeviceAnswer::Acceptance(Acceptance::new(licensed)?)
        }
        else {
            let mut targets = Vec::with_capacity(columns);
            for column in 0 .. columns {
                targets.push(compose(layout, column_digits(planes, column)?)?);
            }
            let acceptance = Acceptance::from_targets(&targets, drafts)?;
            DeviceAnswer::Targets {
                targets,
                acceptance,
            }
        };
        return Ok((answer, cost));
    }
}
