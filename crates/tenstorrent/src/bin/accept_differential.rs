//! The Accept differential: the device's answer against the host reference,
//! sample by sample, with the wall time of every call. `round` reaches the
//! device from infinitum's planner and hands each answer to its preview.

use core::num::NonZeroU32;
use core::num::NonZeroU64;
use std::io::Write as _;

use infinitum_round::Backend as _;
use infinitum_round::BuildFailure;
use infinitum_round::CountOverflow;
use infinitum_round::DraftWidth;
use infinitum_round::Preview;
use infinitum_round::RoundKind;
use infinitum_round::RoundOffer;
use infinitum_round::RoundVerdict;
use infinitum_round::TokenCount;
use infinitum_round::TokenId;
use infinitum_tenstorrent::AcceptGeometry;
use infinitum_tenstorrent::AcceptProgram;
use infinitum_tenstorrent::CompileCost;
use infinitum_tenstorrent::DeviceAnswer;
use infinitum_tenstorrent::HostFailure;
use infinitum_tenstorrent::Lowering;
use infinitum_tenstorrent::PrefixSite;
use infinitum_tenstorrent::Sample;
use infinitum_tenstorrent::SampleCase;
use infinitum_tenstorrent::Seed;
use infinitum_tenstorrent::ShapeMismatch;
use infinitum_tenstorrent::SystemDescriptor;
use infinitum_tenstorrent::Tenstorrent;
use infinitum_tenstorrent::TenstorrentDevice;
use infinitum_tenstorrent::VocabularyLayout;
use infinitum_tenstorrent::reference_accept;

/// How many times something repeats.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Repeats(NonZeroU32);

impl core::str::FromStr for Repeats
{
    type Err = core::num::ParseIntError;

    /// Parse a positive count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the count is the parsed value.
    /// - provides: the command line's counts.
    /// - fails: with the integer parser's error on zero or non-numbers.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<NonZeroU32>().map(Self);
    }
}

impl core::fmt::Display for Repeats
{
    /// Render the count.
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

/// Where the prefix runs, as the command line spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Prefix
{
    /// In the device program.
    Device,
    /// On the host, from the device's target ids.
    Host,
}

/// The harness's commands.
#[derive(Debug, clap::Parser)]
#[command(about = "Compare the Accept fragment on a Tenstorrent device with the host reference")]
enum Command
{
    /// Print the builder's module as TTIR.
    EmitTtir
    {
        /// The draft width `K`.
        #[arg(long, default_value = "15")]
        drafts: DraftWidth,
        /// Where the prefix runs.
        #[arg(long, value_enum, default_value = "device")]
        prefix: Prefix,
    },
    /// Build and lower the module in process, without a device, and report
    /// what each stage cost.
    Compile
    {
        /// The draft width `K`.
        #[arg(long, default_value = "15")]
        drafts: DraftWidth,
        /// Where the prefix runs.
        #[arg(long, value_enum, default_value = "device")]
        prefix: Prefix,
        /// The system descriptor to lower against; without it, the mock
        /// Blackhole descriptor.
        #[arg(long)]
        system_desc: Option<std::path::PathBuf>,
    },
    /// Write the visible device's system descriptor.
    SystemDesc
    {
        /// Where to write it.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Run the differential.
    Run
    {
        /// A flatbuffer the pipeline tools lowered; without it the module is
        /// built and lowered in process.
        #[arg(long, conflicts_with = "system_desc")]
        flatbuffer: Option<std::path::PathBuf>,
        /// The system descriptor to lower against in process.
        #[arg(long, required_unless_present = "flatbuffer")]
        system_desc: Option<std::path::PathBuf>,
        /// The draft width `K` the program was lowered for.
        #[arg(long, default_value = "15")]
        drafts: DraftWidth,
        /// Where the prefix runs.
        #[arg(long, value_enum, default_value = "device")]
        prefix: Prefix,
        /// Seeds per acceptance case.
        #[arg(long, default_value = "4")]
        seeds: Repeats,
        /// Warm calls timed on one sample after the differential.
        #[arg(long, default_value = "20")]
        warm: Repeats,
    },
    /// Plan the canonical DFlash2 round with the Tenstorrent planner, lower
    /// its Accept fragment, and run a request's rounds of acceptance on the
    /// device, each answer reviewed by infinitum's preview.
    Round
    {
        /// The system descriptor to lower against in process.
        #[arg(long)]
        system_desc: std::path::PathBuf,
        /// The round's draft width `K`.
        #[arg(long, default_value = "15")]
        drafts: DraftWidth,
        /// Seeds per acceptance case; each sample is one round.
        #[arg(long, default_value = "4")]
        seeds: Repeats,
        /// The request's output budget in tokens.
        #[arg(long, default_value = "70")]
        budget: Repeats,
    },
}

/// Why the harness stopped.
#[derive(Debug)]
enum HarnessFailure
{
    /// The host or the device failed.
    Host(HostFailure),
    /// A sample could not be built.
    Sample(ShapeMismatch),
    /// Writing the report failed.
    Report(std::io::Error),
    /// At least one sample disagreed.
    Disagreement,
    /// Neither a flatbuffer nor a system descriptor was given.
    NoProgram,
    /// The canonical round could not be composed.
    Composition(BuildFailure),
    /// No fragment of the round lowered to the Accept program.
    NoAcceptLowering,
    /// The preview's count left its range.
    Count(CountOverflow),
    /// The preview's admitted tokens are not the reference's.
    PreviewMismatch,
}

impl core::fmt::Display for HarnessFailure
{
    /// Render the failure.
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
            | Self::Host(ref failure) => write!(f, "{failure}"),
            | Self::Sample(ref mismatch) => write!(f, "a sample could not be built: {mismatch}"),
            | Self::Report(ref error) => write!(f, "writing the report failed: {error}"),
            | Self::Disagreement => f.write_str("the device disagreed with the host reference"),
            | Self::NoProgram => f.write_str("give a flatbuffer or a system descriptor"),
            | Self::Composition(ref failure) => write!(f, "the round did not compose: {failure}"),
            | Self::NoAcceptLowering => f.write_str("no fragment lowered to the Accept program"),
            | Self::Count(ref overflow) => write!(f, "the preview's count overflowed: {overflow}"),
            | Self::PreviewMismatch => {
                f.write_str("the preview admitted tokens other than the reference's")
            },
        };
    }
}

impl From<HostFailure> for HarnessFailure
{
    /// Wrap a host failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(failure: HostFailure) -> Self
    {
        return Self::Host(failure);
    }
}

impl From<ShapeMismatch> for HarnessFailure
{
    /// Wrap a sample failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(mismatch: ShapeMismatch) -> Self
    {
        return Self::Sample(mismatch);
    }
}

impl From<std::io::Error> for HarnessFailure
{
    /// Wrap a report failure.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(error: std::io::Error) -> Self
    {
        return Self::Report(error);
    }
}

/// The geometry for `drafts` over the served vocabulary.
///
/// # Specification
/// trivial.
const fn geometry(
    drafts: DraftWidth,
    prefix: Prefix,
) -> AcceptGeometry
{
    return AcceptGeometry::new(drafts, VocabularyLayout::served(), match prefix {
        | Prefix::Device => PrefixSite::Device,
        | Prefix::Host => PrefixSite::Host,
    });
}

/// Run the differential and the warm timing, writing one line per sample.
///
/// # Specification
/// - requires: a device is visible; the program was lowered for `geometry`.
/// - ensures: every sample of every case ran once and was compared with the
///   reference; `warm` further calls on the first sample were timed; the report
///   names each sample's case, seed, both answers, and its times.
/// - provides: the rung reports' evidence.
/// - fails: when the host fails, or with [`HarnessFailure::Disagreement`] after
///   the report when any sample disagreed.
/// - panics: none.
///
/// # Errors
/// - [`HarnessFailure`]: as described.
fn differential(
    out: &mut dyn std::io::Write,
    program: &AcceptProgram,
    geometry_drafts: DraftWidth,
    seeds: Repeats,
    warm: Repeats,
) -> Result<(), HarnessFailure>
{
    let layout = VocabularyLayout::served();
    let mut device = TenstorrentDevice::open(program)?;
    let mut disagreements = 0_u32;
    let mut first: Option<Sample> = None;
    writeln!(
        out,
        "case | seed | reference | device | agree | submit | readback"
    )?;
    for case in SampleCase::ALL {
        for seed in 0 .. NonZeroU64::from(seeds.0).get() {
            let sample = Sample::generate(case, layout, geometry_drafts, Seed::from(seed))?;
            let reference = reference_accept(sample.logits(), sample.drafts())?;
            let (answer, cost) = device.run(program, sample.logits(), sample.drafts())?;
            let agree = answer.acceptance() == &reference;
            if !agree {
                disagreements = disagreements.saturating_add(1);
            }
            let device_answer = answer.acceptance();
            let site = match answer {
                | DeviceAnswer::Acceptance(_) => "device prefix",
                | DeviceAnswer::Targets { .. } => "host prefix",
            };
            writeln!(
                out,
                "{case} | {seed} | {reference} | {device_answer} ({site}) | {agree} | {} | {}",
                cost.submit, cost.readback
            )?;
            if first.is_none() {
                first = Some(sample);
            }
        }
    }
    if let Some(sample) = first {
        let mut submits = Vec::new();
        let mut readbacks = Vec::new();
        for _ in 0 .. warm.0.get() {
            let (_, cost) = device.run(program, sample.logits(), sample.drafts())?;
            submits.push(cost.submit);
            readbacks.push(cost.readback);
        }
        submits.sort_unstable();
        readbacks.sort_unstable();
        let middle = warm
            .0
            .get()
            .checked_div(2)
            .and_then(|half| return usize::try_from(half).ok())
            .unwrap_or_default();
        if let (Some(least), Some(median), Some(most), Some(read_least), Some(read_median)) = (
            submits.first(),
            submits.get(middle),
            submits.last(),
            readbacks.first(),
            readbacks.get(middle),
        ) {
            writeln!(
                out,
                "warm calls: {warm}; submit min {least} median {median} max {most}; readback min {read_least} median {read_median}"
            )?;
        }
    }
    writeln!(out, "disagreements: {disagreements}")?;
    if disagreements > 0 {
        return Err(HarnessFailure::Disagreement);
    }
    return Ok(());
}

/// Plan the canonical round, lower its Accept fragment, and run one request
/// of rounds on the device through infinitum's preview.
///
/// # Specification
/// - requires: a device is visible.
/// - ensures: the whole-round verdict and every fragment's lowering or refusal
///   are reported; every sample ran once as a round and was compared with the
///   reference; each device answer was offered to one preview, until its first
///   limit ends the request; the preview's tokens are compared with the
///   reference's licensed tokens cut at the budget.
/// - provides: the rung 3 report's evidence.
/// - fails: when the host fails, with [`HarnessFailure::Disagreement`] when a
///   sample disagreed, or with [`HarnessFailure::PreviewMismatch`] when the
///   admitted tokens differ.
/// - panics: none.
///
/// # Errors
/// - [`HarnessFailure`]: as described.
fn round(
    out: &mut dyn std::io::Write,
    system_desc: &std::path::Path,
    drafts: DraftWidth,
    seeds: Repeats,
    budget: Repeats,
) -> Result<(), HarnessFailure>
{
    let graph = infinitum_round::dflash2(drafts).map_err(HarnessFailure::Composition)?;
    writeln!(
        out,
        "round: canonical DFlash2 at K = {drafts}, {} fragments",
        graph.fragments().len()
    )?;
    match Tenstorrent.plan(&graph) {
        | Ok(plan) => writeln!(
            out,
            "whole round: planned, {} lowerings",
            plan.lowerings().len()
        )?,
        | Err(refusal) => writeln!(out, "whole round: {refusal}")?,
    }
    let mut accept = None;
    for (at, fragment) in graph.entries() {
        let kind = fragment.kind();
        match Tenstorrent.plan_fragment(&graph, at, fragment) {
            | Ok(Lowering::Accept(plan)) => {
                writeln!(
                    out,
                    "{at} {kind}: lowers to the Accept program at K = {}",
                    plan.width()
                )?;
                accept = Some(plan);
            },
            | Err(refusal) => writeln!(out, "{at} {kind}: {refusal}")?,
        }
    }
    let plan = accept.ok_or(HarnessFailure::NoAcceptLowering)?;
    let (program, cost) = AcceptProgram::compile(
        geometry(plan.width(), Prefix::Device),
        SystemDescriptor::Saved(system_desc),
    )?;
    write_cost(out, &cost)?;
    let layout = VocabularyLayout::served();
    let mut device = TenstorrentDevice::open(&program)?;
    let budget_tokens = TokenCount::from(budget.0);
    let budget_len = usize::try_from(u32::from(budget_tokens)).unwrap_or(usize::MAX);
    let mut preview = Preview::new(budget_tokens);
    let mut offering = true;
    let mut reference_tokens: Vec<TokenId> = Vec::new();
    let mut disagreements = 0_u32;
    writeln!(
        out,
        "case | seed | device | agree | verdict | call | submit | readback"
    )?;
    for case in SampleCase::ALL {
        for seed in 0 .. NonZeroU64::from(seeds.0).get() {
            let sample = Sample::generate(case, layout, plan.width(), Seed::from(seed))?;
            let reference = reference_accept(sample.logits(), sample.drafts())?;
            let start = std::time::Instant::now();
            let (answer, cost) = device.run(&program, sample.logits(), sample.drafts())?;
            let call = start.elapsed();
            let acceptance = answer.acceptance();
            let agree = acceptance == &reference;
            if !agree {
                disagreements = disagreements.saturating_add(1);
            }
            let verdict = if offering {
                reference_tokens.extend_from_slice(reference.licensed());
                let verdict = preview
                    .review(RoundOffer::new(acceptance.licensed(), RoundKind::Decode))
                    .map_err(HarnessFailure::Count)?;
                // A backend given the same budget finishes the request once
                // it is spent, by a limit or by an exact fit.
                offering = preview.tokens().len() < budget_len;
                match verdict {
                    | RoundVerdict::Limit(limit) => {
                        format!("limit {}", core::num::NonZeroU32::from(limit))
                    },
                    | RoundVerdict::Continue => "continue".to_owned(),
                }
            }
            else {
                "request finished".to_owned()
            };
            writeln!(
                out,
                "{case} | {seed} | {acceptance} | {agree} | {verdict} | {:.1} us | {} | {}",
                call.as_secs_f64() * 1.0e6_f64,
                cost.submit,
                cost.readback
            )?;
        }
    }
    reference_tokens.truncate(budget_len);
    let admitted_agree = preview.tokens() == reference_tokens.as_slice();
    writeln!(
        out,
        "preview: {} rounds reviewed, {} tokens admitted of {budget}; equal to the reference cut at the budget: {admitted_agree}",
        preview.rounds().len(),
        preview.tokens().len()
    )?;
    writeln!(out, "disagreements: {disagreements}")?;
    if disagreements > 0 {
        return Err(HarnessFailure::Disagreement);
    }
    if !admitted_agree {
        return Err(HarnessFailure::PreviewMismatch);
    }
    return Ok(());
}

/// Write what lowering in process cost.
///
/// # Specification
/// - requires: nothing.
/// - ensures: one line naming each stage's wall time and the program count.
/// - provides: the in-process route's cost in the report.
/// - fails: when writing fails.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: writing failed.
fn write_cost(
    out: &mut dyn std::io::Write,
    cost: &CompileCost,
) -> std::io::Result<()>
{
    return writeln!(
        out,
        "program: in process; build {}, pipeline {}, translate {}, {} device programs",
        cost.build, cost.pipeline, cost.translate, cost.programs
    );
}

/// Run the command.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the command ran and its report was written to standard output.
/// - provides: the harness.
/// - fails: as the command does.
/// - panics: none.
///
/// # Errors
/// - [`HarnessFailure`]: the command failed.
fn run(command: Command) -> Result<(), HarnessFailure>
{
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    return match command {
        | Command::EmitTtir { drafts, prefix } => {
            let text = infinitum_tenstorrent::print_accept(&geometry(drafts, prefix))?;
            out.write_all(text.as_bytes())?;
            Ok(())
        },
        | Command::Compile {
            drafts,
            prefix,
            system_desc,
        } => {
            let descriptor = system_desc
                .as_deref()
                .map_or(SystemDescriptor::MockBlackhole, SystemDescriptor::Saved);
            let (_, cost) = AcceptProgram::compile(geometry(drafts, prefix), descriptor)?;
            write_cost(&mut out, &cost)?;
            Ok(())
        },
        | Command::SystemDesc { out: path } => {
            infinitum_tenstorrent::save_system_desc(&path)?;
            writeln!(out, "wrote {}", path.display())?;
            Ok(())
        },
        | Command::Run {
            flatbuffer,
            system_desc,
            drafts,
            prefix,
            seeds,
            warm,
        } => {
            let geometry = geometry(drafts, prefix);
            let program = match (flatbuffer, system_desc) {
                | (Some(path), _) => {
                    writeln!(out, "program: flatbuffer {}", path.display())?;
                    AcceptProgram::load(geometry, &path)?
                },
                | (None, Some(path)) => {
                    let (program, cost) =
                        AcceptProgram::compile(geometry, SystemDescriptor::Saved(&path))?;
                    write_cost(&mut out, &cost)?;
                    program
                },
                | (None, None) => return Err(HarnessFailure::NoProgram),
            };
            differential(&mut out, &program, drafts, seeds, warm)
        },
        | Command::Round {
            system_desc,
            drafts,
            seeds,
            budget,
        } => round(&mut out, &system_desc, drafts, seeds, budget),
    };
}

/// Parse the command line and run it.
///
/// # Specification
/// - requires: nothing.
/// - ensures: success exactly when the command succeeded.
/// - provides: the harness's entry point.
/// - fails: with the failure written to standard error and a failing exit.
/// - panics: none.
fn main() -> std::process::ExitCode
{
    let command = <Command as clap::Parser>::parse();
    return match run(command) {
        | Ok(()) => std::process::ExitCode::SUCCESS,
        | Err(failure) => {
            let stderr = std::io::stderr();
            let mut err = stderr.lock();
            let _written = writeln!(err, "accept-differential: {failure}");
            std::process::ExitCode::FAILURE
        },
    };
}
