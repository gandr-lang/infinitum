//! Squash-aware changelog generation.
//!
//! `treefmt` calls this through the `changelog` task and installs its output
//! over `CHANGELOG.md` when the bytes differ. The problem it exists to solve
//! is that a pull request lands as a single squash commit whose subject is
//! `<pull request title> (#<number>)`, so the changelog a branch would
//! generate from its own commits is never the changelog the default branch
//! generates afterwards.
//!
//! On a branch with an **open** pull request this renders the history the
//! merge will produce: every commit the branch adds is skipped, and one
//! synthetic commit carrying the squash subject takes their place. On the
//! default branch, and on the queue and push runs where the real squash
//! commit is already present, it renders history plainly.

use std::io::Write as _;

/// Where the generated changelog is written, for the installing formatter to
/// compare against the tracked file.
const OUTPUT_PATH: &str = ".git-cliff-output";

/// The default branch, whose history needs no reconstruction.
const DEFAULT_BRANCH: &str = "main";

/// The remote-tracking ref the branch history is measured against.
const DEFAULT_BRANCH_REMOTE_REF: &str = "origin/main";

/// The forge's state word for a pull request that has not landed or closed.
const OPEN_STATE: &str = "OPEN";

/// A forge event name, as the runner reports it.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct EventName(String);

/// A Git ref name, as the runner reports it.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RefName(String);

/// A local branch name.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct BranchName(String);

/// The argument that names a pull request to the forge CLI: a branch name or
/// a request number, whichever the run can supply.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PullRequestReference(String);

/// A commit the squash replaces.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommitId(String);

/// The subject the squash commit will carry.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SquashSubject(String);

/// An executable this binary runs.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Program(String);

/// One command-line argument.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Argument(String);

/// A command's captured standard output, trimmed.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedOutput(String);

/// The generator's assembled command line.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CliffArguments(Vec<Argument>);

/// The open pull request a branch's changelog is rendered as.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PullRequest
{
    /// The title, which becomes the squash subject.
    title: String,
    /// The number the squash subject carries in parentheses.
    number: String,
}

/// Which history a run renders.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RenderingMode
{
    /// Render what Git holds: the real squash commit is already present.
    Plain,
    /// Render the merge's history: the branch's commits replaced by the
    /// subject its squash will carry.
    Reconstructed
    {
        /// The commits the squash replaces.
        skipped: Vec<CommitId>,
        /// The subject that replaces them.
        subject: SquashSubject,
    },
}

/// Why changelog generation could not complete.
#[derive(Debug)]
enum Failure
{
    /// A command could not be started.
    Spawn
    {
        /// The program that could not be started.
        program: Program,
        /// What the operating system reported.
        cause: std::io::Error,
    },
    /// A command ran and reported failure.
    Command
    {
        /// The program that failed.
        program: Program,
        /// What it wrote to standard error.
        message: String,
    },
    /// A command wrote output that was not UTF-8.
    Encoding
    {
        /// The program whose output could not be decoded.
        program: Program,
    },
    /// The forge reported no pull request, or one that is not open.
    NoOpenPullRequest
    {
        /// The reference that was looked up.
        reference: PullRequestReference,
        /// The state the forge reported, where it reported one.
        state: String,
    },
    /// The forge's answer did not carry the three fields that were requested.
    MalformedPayload
    {
        /// The reference that was looked up.
        reference: PullRequestReference,
    },
}

impl core::fmt::Display for Failure
{
    /// Render the failure for the operator who has to act on it.
    ///
    /// # Specification
    /// - requires: `formatter` is the sink the standard trait supplies.
    /// - ensures: each variant renders the fact and, where one exists, the
    ///   action that resolves it.
    /// - provides: the process's only diagnostic surface.
    /// - fails: only as the writer fails.
    /// - panics: none.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        match *self {
            | Self::Spawn {
                ref program,
                ref cause,
            } => {
                return write!(f, "`{}` could not be started: {cause}", program.0);
            },
            | Self::Command {
                ref program,
                ref message,
            } => {
                return write!(f, "`{}` failed: {message}", program.0);
            },
            | Self::Encoding { ref program } => {
                return write!(f, "`{}` wrote output that is not UTF-8", program.0);
            },
            | Self::NoOpenPullRequest {
                ref reference,
                ref state,
            } => {
                return write!(
                    f,
                    "no open pull request for `{}` (state: {state}): open the PR first, then \
                 regenerate",
                    reference.0
                );
            },
            | Self::MalformedPayload { ref reference } => {
                return write!(
                    f,
                    "the forge's answer for `{}` did not carry state, number and title",
                    reference.0
                );
            },
        }
    }
}

impl core::error::Error for Failure
{
}

impl PullRequest
{
    /// The subject the squash commit will carry.
    ///
    /// # Specification
    /// - requires: nothing of the caller.
    /// - ensures: the returned subject is the title followed by the number in
    ///   the parenthesised form the forge appends when it squashes.
    /// - provides: the single commit message the branch's changelog is rendered
    ///   from.
    /// - fails: never.
    /// - panics: none.
    fn squash_subject(&self) -> SquashSubject
    {
        return SquashSubject(format!("{} (#{})", self.title, self.number));
    }
}

/// Name the pull request whose changelog a run must render.
///
/// # Specification
/// - requires: `event` is the forge's event name where one is set, `head_ref`
///   the source branch a pull-request run reports, `ref_name` the ref the run
///   checked out, and `branch` the local branch name.
/// - ensures: a pull-request run resolves to its source branch where the forge
///   reports one, else to the number the forge puts in
///   `refs/pull/<number>/merge`; every other run, and a pull-request run
///   reporting neither, resolves to the branch name, which is what the forge
///   CLI accepts when a pull request is open for it.
/// - provides: the one argument the pull-request lookup needs, independent of
///   where the run happens.
/// - fails: never; an unusable reference surfaces as a lookup failure, whose
///   message names the fix.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 only — one selection over the three shapes a run supplies,
///   each asserted exactly.
/// - witness: `tests::pull_request_reference_prefers_the_head_ref`
/// - witness: `tests::pull_request_reference_takes_the_number_from_a_merge_ref`
/// - witness: `tests::pull_request_reference_takes_the_branch_elsewhere`
/// - witness: `tests::pull_request_reference_falls_back_to_the_branch`
fn pull_request_reference(
    event: &EventName,
    head_ref: &RefName,
    ref_name: &RefName,
    branch: &BranchName,
) -> PullRequestReference
{
    if event.0 == "pull_request" {
        if !head_ref.0.is_empty() {
            return PullRequestReference(head_ref.0.clone());
        }
        if let Some((number, _)) = ref_name.0.split_once('/')
            && !number.is_empty()
        {
            return PullRequestReference(String::from(number));
        }
    }
    return PullRequestReference(branch.0.clone());
}

/// Read the forge's answer, accepting an open pull request and nothing else.
///
/// # Specification
/// - requires: `payload` is the forge CLI's answer to a request for state,
///   number and title, one field per line in that order.
/// - ensures: a payload whose state is `OPEN` and whose number and title are
///   both non-empty yields that request; every other payload is a failure
///   naming what was found.
/// - provides: the guarantee the generator's contract rests on — a merged or
///   closed request is refused, so a reused branch cannot render an old title
///   over a changelog the merge then contradicts.
/// - fails: with [`Failure::NoOpenPullRequest`] when the state is not `OPEN`,
///   and with [`Failure::MalformedPayload`] when the three fields are not
///   there.
/// - panics: none.
///
/// # Errors
/// - [`Failure::NoOpenPullRequest`]: the request is merged, closed, or absent.
/// - [`Failure::MalformedPayload`]: the answer lacked a field.
///
/// # Adequacy
/// - hypothesis: L3 pointwise over the state partition — open accepted, merged
///   and closed refused — plus the malformed shape, because the partition is
///   what the rule turns on.
/// - witness: `tests::an_open_request_is_accepted`
/// - witness: `tests::a_merged_request_is_refused`
/// - witness: `tests::a_closed_request_is_refused`
/// - witness: `tests::an_incomplete_answer_is_refused`
fn accept_open_pull_request(
    reference: &PullRequestReference,
    payload: &CapturedOutput,
) -> Result<PullRequest, Failure>
{
    let mut fields = payload.0.lines();
    let state = fields.next().unwrap_or_default().trim();
    let number = fields.next().unwrap_or_default().trim();
    let title = fields.next().unwrap_or_default().trim();
    if state.is_empty() || number.is_empty() || title.is_empty() {
        return Err(Failure::MalformedPayload {
            reference: reference.clone(),
        });
    }
    if state != OPEN_STATE {
        return Err(Failure::NoOpenPullRequest {
            reference: reference.clone(),
            state: String::from(state),
        });
    }
    return Ok(PullRequest {
        title: String::from(title),
        number: String::from(number),
    });
}

/// Assemble the generator's arguments.
///
/// # Specification
/// - requires: `mode` names the history to render and `passthrough` carries
///   whatever the formatter appended.
/// - ensures: the output path leads; a reconstructed render contributes one
///   `--skip-commit` per replaced commit and exactly one `--with-commit` for
///   the squash subject; the passthrough arguments come last, so the
///   formatter's own exclusion still reaches the generator.
/// - provides: the whole command line, built once and identically for every
///   mode, so a mode differs only in what it supplies here.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 only — assembly whose observable residue is argument order
///   and multiplicity, asserted exactly for both modes.
/// - witness: `tests::plain_arguments_carry_only_the_output_and_passthrough`
/// - witness: `tests::reconstructed_arguments_skip_each_commit_and_add_one`
fn cliff_arguments(
    mode: &RenderingMode,
    passthrough: Vec<Argument>,
) -> CliffArguments
{
    let mut arguments = vec![
        Argument(String::from("--output")),
        Argument(String::from(OUTPUT_PATH)),
    ];
    if let RenderingMode::Reconstructed {
        ref skipped,
        ref subject,
    } = *mode
    {
        for commit in skipped {
            arguments.push(Argument(String::from("--skip-commit")));
            arguments.push(Argument(commit.0.clone()));
        }
        arguments.push(Argument(String::from("--with-commit")));
        arguments.push(Argument(subject.0.clone()));
    }
    arguments.extend(passthrough);
    return CliffArguments(arguments);
}

/// Run a command and capture its trimmed standard output.
///
/// # Specification
/// - requires: `program` names an executable on the path and `arguments` are
///   its arguments.
/// - ensures: on success the captured standard output is returned with
///   surrounding whitespace removed.
/// - provides: the only channel through which this binary reads Git and the
///   forge CLI.
/// - fails: when the command cannot be spawned, exits nonzero, or writes output
///   that is not UTF-8; each failure names the program, and a command failure
///   carries its standard error.
/// - panics: none.
///
/// # Errors
/// - [`Failure::Spawn`]: the command could not be started.
/// - [`Failure::Command`]: the command reported failure.
/// - [`Failure::Encoding`]: the output was not UTF-8.
///
/// # Adequacy
/// - hypothesis: L3 over the outcome partition, exercised against real commands
///   rather than a mock: success returns the trimmed output, a nonzero exit
///   carries the program's own standard error, and an absent executable is a
///   spawn failure. Encoding is unreachable from the two callers, whose outputs
///   are Git object text and the forge CLI's JSON projection, and is left to
///   the type.
/// - witness: `tests::capture_returns_trimmed_output`
/// - witness: `tests::capture_reports_a_failing_command`
/// - witness: `tests::capture_reports_a_missing_program`
fn capture(
    program: &Program,
    arguments: &[Argument],
) -> Result<CapturedOutput, Failure>
{
    let rendered: Vec<&str> = arguments
        .iter()
        .map(|argument| return argument.0.as_str())
        .collect();
    let output = std::process::Command::new(&program.0)
        .args(&rendered)
        .output()
        .map_err(|cause| {
            return Failure::Spawn {
                program: program.clone(),
                cause,
            };
        })?;
    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr);
        return Err(Failure::Command {
            program: program.clone(),
            message: String::from(message.trim()),
        });
    }
    let captured = String::from_utf8(output.stdout).map_err(|_ignored| {
        return Failure::Encoding {
            program: program.clone(),
        };
    })?;
    return Ok(CapturedOutput(String::from(captured.trim())));
}

/// Look up the open pull request for `reference`.
///
/// # Specification
/// - requires: `reference` is a branch name or pull request number, and the
///   forge CLI is available and authenticated.
/// - ensures: on success the title and number of an **open** pull request are
///   returned.
/// - provides: the squash subject a branch's changelog is rendered from.
/// - fails: when the forge CLI cannot answer, when no request exists for the
///   reference, and — the case that distinguishes this from a bare lookup —
///   when the request that exists is merged or closed. Guessing a title would
///   write a changelog the merge then contradicts, so the failure says to open
///   the request first.
/// - panics: none.
///
/// # Errors
/// - [`Failure::Spawn`], [`Failure::Command`]: the forge CLI could not answer.
/// - [`Failure::NoOpenPullRequest`]: no request, or not an open one.
/// - [`Failure::MalformedPayload`]: the answer lacked a field.
///
/// # Adequacy
/// - hypothesis: L2 for the process boundary — that the CLI is asked for state,
///   number and title in that order — since the decision this function makes is
///   [`accept_open_pull_request`], which carries the pointwise witnesses. A
///   failing lookup is mapped to the same refusal, so a repository without the
///   request behaves as a closed one.
/// - witness: `tests::an_open_request_is_accepted`
/// - witness: `tests::a_merged_request_is_refused`
fn open_pull_request(reference: &PullRequestReference) -> Result<PullRequest, Failure>
{
    let program = Program(String::from("gh"));
    let arguments: Vec<Argument> = [
        "pr",
        "view",
        reference.0.as_str(),
        "--json",
        "state,number,title",
        "--jq",
        ".state, .number, .title",
    ]
    .iter()
    .map(|piece| return Argument(String::from(*piece)))
    .collect();
    let payload = capture(&program, &arguments).map_err(|_ignored| {
        return Failure::NoOpenPullRequest {
            reference: reference.clone(),
            state: String::from("none"),
        };
    })?;
    return accept_open_pull_request(reference, &payload);
}

/// Decide which history this run renders.
///
/// # Specification
/// - requires: the working directory is inside the repository, and `git` — on a
///   branch, also `gh` — resolves on the path.
/// - ensures: a run that already holds the squash commit renders plainly; every
///   other run renders the merge's history, with the branch's commits since the
///   merge base replaced by the open request's squash subject.
/// - provides: the single decision the binary exists to make.
/// - fails: as [`open_pull_request`] and the Git lookups fail.
/// - panics: none.
///
/// # Errors
/// - [`Failure::NoOpenPullRequest`]: the branch has no open pull request.
/// - [`Failure::Spawn`], [`Failure::Command`]: a lookup failed.
///
/// # Adequacy
/// - hypothesis: L2 for the branch selection, whose arms are the two rendering
///   modes; the modes' observable output is witnessed by the argument assembly,
///   and the end-to-end equivalence between a branch render and its squash
///   render is witnessed outside the suite, in the pull request's recorded
///   comparison.
/// - witness: `tests::plain_arguments_carry_only_the_output_and_passthrough`
/// - witness: `tests::reconstructed_arguments_skip_each_commit_and_add_one`
fn rendering_mode(
    event: &EventName,
    branch: &BranchName,
) -> Result<RenderingMode, Failure>
{
    // The queue and push runs already carry the squash commit itself, so the
    // history they render is the history that lands: no reconstruction, and a
    // changelog that went stale between opening and merging bounces there.
    let already_squashed = event.0 == "merge_group" || event.0 == "push";
    if already_squashed || branch.0 == DEFAULT_BRANCH {
        return Ok(RenderingMode::Plain);
    }
    let head_ref = RefName(std::env::var("GITHUB_HEAD_REF").unwrap_or_default());
    let ref_name = RefName(std::env::var("GITHUB_REF_NAME").unwrap_or_default());
    let reference = pull_request_reference(event, &head_ref, &ref_name, branch);
    let pull_request = open_pull_request(&reference)?;
    let git = Program(String::from("git"));
    let base = capture(&git, &[
        Argument(String::from("merge-base")),
        Argument(String::from(DEFAULT_BRANCH_REMOTE_REF)),
        Argument(String::from("HEAD")),
    ])?;
    let range = Argument(format!("{}..HEAD", base.0));
    let listed = capture(&git, &[Argument(String::from("rev-list")), range])?;
    let skipped = listed
        .0
        .lines()
        .map(|line| return CommitId(String::from(line.trim())))
        .collect();
    return Ok(RenderingMode::Reconstructed {
        skipped,
        subject: pull_request.squash_subject(),
    });
}

/// Generate the changelog, reporting failure to the caller rather than to the
/// operator.
///
/// # Specification
/// - requires: the working directory is inside the repository, and `git`,
///   `git-cliff` and — on a branch — `gh` resolve on the path.
/// - ensures: the generator has written [`OUTPUT_PATH`] from history rendered
///   as the default branch will hold it.
/// - provides: a changelog on a branch that the merge cannot invalidate.
/// - fails: when a command fails, or when a branch has no open pull request.
/// - panics: none.
///
/// # Errors
/// - [`Failure::Spawn`], [`Failure::Command`]: a command failed.
/// - [`Failure::NoOpenPullRequest`]: the branch has no open pull request.
///
/// # Adequacy
/// - hypothesis: L2 for the wiring — mode decision, argument assembly, one
///   generator invocation — with the decisions themselves witnessed by
///   [`rendering_mode`] and [`cliff_arguments`]. The end-to-end claim, that a
///   branch render equals its post-squash render byte for byte, is witnessed by
///   the comparison recorded on the pull request, which no in-process test can
///   make honestly: it needs a second history.
/// - witness: `tests::reconstructed_arguments_skip_each_commit_and_add_one`
fn run() -> Result<(), Failure>
{
    let passthrough: Vec<Argument> = std::env::args()
        .skip(1)
        .map(|argument| return Argument(argument))
        .collect();
    let event = EventName(std::env::var("GITHUB_EVENT_NAME").unwrap_or_default());
    let git = Program(String::from("git"));
    let branch = capture(&git, &[
        Argument(String::from("rev-parse")),
        Argument(String::from("--abbrev-ref")),
        Argument(String::from("HEAD")),
    ])?;
    let mode = rendering_mode(&event, &BranchName(branch.0))?;
    let arguments = cliff_arguments(&mode, passthrough);
    let rendered: Vec<&str> = arguments
        .0
        .iter()
        .map(|argument| return argument.0.as_str())
        .collect();
    let generator = Program(String::from("git-cliff"));
    let status = std::process::Command::new(&generator.0)
        .args(&rendered)
        .status()
        .map_err(|cause| {
            return Failure::Spawn {
                program: generator.clone(),
                cause,
            };
        })?;
    if !status.success() {
        return Err(Failure::Command {
            program: generator,
            message: format!("exited with {status}"),
        });
    }
    return Ok(());
}

/// Report a failure the way its reader needs it, and set the exit status.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a failure has reached the standard error stream through
///   [`Failure`]'s [`Display`](core::fmt::Display) implementation, carrying the
///   remediation that implementation spells out, and the process exits
///   unsuccessfully.
/// - provides: an operator reading a failure rather than a data structure.
/// - fails: never; a stream that refuses the write still yields the status.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 over the reporting partition — success is silent and
///   successful, failure is printed through `Display` and unsuccessful. The
///   distinction that matters is invisible to a return-type test, because
///   `Result`'s own [`Termination`](std::process::Termination) prints `Debug`;
///   the witness therefore runs the built binary and reads what it wrote.
/// - witness: `tests/process_boundary.
///   rs::a_failure_is_reported_through_display`
fn main() -> std::process::ExitCode
{
    if let Err(failure) = run() {
        let mut stderr = std::io::stderr().lock();
        let written = writeln!(stderr, "error: {failure}");
        drop(written);
        return std::process::ExitCode::FAILURE;
    }
    return std::process::ExitCode::SUCCESS;
}

/// Tests for reference selection, request acceptance, command capture and
/// argument assembly.
#[cfg(test)]
mod tests
{
    use super::Argument;
    use super::BranchName;
    use super::CapturedOutput;
    use super::CommitId;
    use super::EventName;
    use super::Failure;
    use super::Program;
    use super::PullRequest;
    use super::PullRequestReference;
    use super::RefName;
    use super::RenderingMode;
    use super::SquashSubject;
    use super::accept_open_pull_request;
    use super::capture;
    use super::cliff_arguments;
    use super::pull_request_reference;

    /// The reference used throughout the request tests.
    fn reference() -> PullRequestReference
    {
        return PullRequestReference(String::from("changelog-squash"));
    }

    /// A pull-request run names the request by the source branch the forge
    /// reports, which is what the forge CLI resolves without a lookup table.
    #[test]
    fn pull_request_reference_prefers_the_head_ref()
    {
        let reference = pull_request_reference(
            &EventName(String::from("pull_request")),
            &RefName(String::from("changelog-squash")),
            &RefName(String::from("2/merge")),
            &BranchName(String::from("HEAD")),
        );
        assert_eq!(
            reference.0, "changelog-squash",
            "the head ref names the request"
        );
    }

    /// A pull-request run names the request by the number in its merge ref,
    /// since no branch of that name exists in that checkout.
    #[test]
    fn pull_request_reference_takes_the_number_from_a_merge_ref()
    {
        let reference = pull_request_reference(
            &EventName(String::from("pull_request")),
            &RefName(String::new()),
            &RefName(String::from("2/merge")),
            &BranchName(String::from("HEAD")),
        );
        assert_eq!(
            reference.0, "2",
            "the merge ref names the request by number"
        );
    }

    /// Every other run names the request by the branch it is on.
    #[test]
    fn pull_request_reference_takes_the_branch_elsewhere()
    {
        let reference = pull_request_reference(
            &EventName(String::new()),
            &RefName(String::new()),
            &RefName(String::new()),
            &BranchName(String::from("changelog-squash")),
        );
        assert_eq!(
            reference.0, "changelog-squash",
            "a local run names the branch"
        );
    }

    /// A pull-request run whose refs the runner never set falls back to the
    /// branch, rather than asking the forge about an empty reference.
    #[test]
    fn pull_request_reference_falls_back_to_the_branch()
    {
        let reference = pull_request_reference(
            &EventName(String::from("pull_request")),
            &RefName(String::new()),
            &RefName(String::new()),
            &BranchName(String::from("changelog-squash")),
        );
        assert_eq!(
            reference.0, "changelog-squash",
            "an empty ref falls back to the branch"
        );
    }

    /// An open request supplies the subject the squash will carry.
    #[test]
    fn an_open_request_is_accepted()
    {
        let payload = CapturedOutput(String::from("OPEN\n2\nfix(repo): tidy"));
        let accepted = accept_open_pull_request(&reference(), &payload);
        let request = accepted.expect("an open request is accepted");
        assert_eq!(
            request.squash_subject(),
            SquashSubject(String::from("fix(repo): tidy (#2)")),
            "the subject is the title with the number appended"
        );
    }

    /// A merged request is refused: a reused branch would otherwise render an
    /// old title over a changelog the next merge contradicts.
    #[test]
    fn a_merged_request_is_refused()
    {
        let payload = CapturedOutput(String::from("MERGED\n1\nfeat(repo): scaffold"));
        let refused = accept_open_pull_request(&reference(), &payload);
        match refused {
            | Err(Failure::NoOpenPullRequest { state, .. }) => {
                assert_eq!(state, "MERGED", "the refusal names the state it found");
            },
            | other => panic!("a merged request must be refused, got {other:?}"),
        }
    }

    /// A closed request is refused on the same terms.
    #[test]
    fn a_closed_request_is_refused()
    {
        let payload = CapturedOutput(String::from("CLOSED\n7\nfix(repo): abandoned"));
        let refused = accept_open_pull_request(&reference(), &payload);
        match refused {
            | Err(Failure::NoOpenPullRequest { state, .. }) => {
                assert_eq!(state, "CLOSED", "the refusal names the state it found");
            },
            | other => panic!("a closed request must be refused, got {other:?}"),
        }
    }

    /// The refusal reads as the remediation it asks for: the process boundary
    /// prints this text, so this is the text the author acts on.
    #[test]
    fn the_refusal_reads_as_its_remediation()
    {
        let payload = CapturedOutput(String::from("MERGED\n1\nfeat(repo): scaffold"));
        let refused = accept_open_pull_request(&reference(), &payload)
            .expect_err("a merged request is refused");
        assert_eq!(
            refused.to_string(),
            "no open pull request for `changelog-squash` (state: MERGED): open the PR first, \
             then regenerate",
            "the refusal names the branch, the state and the remedy"
        );
    }

    /// An answer missing a field is a malformed payload, not a silent render.
    #[test]
    fn an_incomplete_answer_is_refused()
    {
        let payload = CapturedOutput(String::from("OPEN\n2"));
        let refused = accept_open_pull_request(&reference(), &payload);
        assert!(
            matches!(refused, Err(Failure::MalformedPayload { .. })),
            "a short answer is refused"
        );
    }

    /// Capture returns what the command wrote, without its trailing newline.
    #[test]
    fn capture_returns_trimmed_output()
    {
        let captured = capture(&Program(String::from("echo")), &[Argument(String::from(
            "hello",
        ))]);
        assert_eq!(
            captured.expect("echo succeeds"),
            CapturedOutput(String::from("hello")),
            "the captured output is trimmed"
        );
    }

    /// A command that exits nonzero is a failure carrying its own message.
    #[test]
    fn capture_reports_a_failing_command()
    {
        let failed = capture(&Program(String::from("git")), &[
            Argument(String::from("rev-parse")),
            Argument(String::from("--verify")),
            Argument(String::from("refs/heads/this-ref-does-not-exist")),
        ]);
        assert!(
            matches!(failed, Err(Failure::Command { .. })),
            "a nonzero exit is a failure"
        );
    }

    /// A program that is not installed is a spawn failure, distinct from a
    /// command that ran and refused.
    #[test]
    fn capture_reports_a_missing_program()
    {
        let failed = capture(&Program(String::from("this-program-does-not-exist")), &[]);
        assert!(
            matches!(failed, Err(Failure::Spawn { .. })),
            "an absent program cannot spawn"
        );
    }

    /// Plain generation adds nothing to the output path and the formatter's
    /// own arguments.
    #[test]
    fn plain_arguments_carry_only_the_output_and_passthrough()
    {
        let arguments = cliff_arguments(&RenderingMode::Plain, vec![
            Argument(String::from("--exclude-path")),
            Argument(String::from("CHANGELOG.md")),
        ]);
        let rendered: Vec<String> = arguments
            .0
            .iter()
            .map(|argument| return argument.0.clone())
            .collect();
        assert_eq!(
            rendered,
            vec![
                "--output",
                ".git-cliff-output",
                "--exclude-path",
                "CHANGELOG.md"
            ],
            "plain generation passes the formatter's arguments through"
        );
    }

    /// Reconstruction skips every branch commit and adds exactly one commit,
    /// whose subject is the squash subject.
    #[test]
    fn reconstructed_arguments_skip_each_commit_and_add_one()
    {
        let request = PullRequest {
            title: String::from("fix(repo): tidy"),
            number: String::from("7"),
        };
        let mode = RenderingMode::Reconstructed {
            skipped: vec![CommitId(String::from("aaa")), CommitId(String::from("bbb"))],
            subject: request.squash_subject(),
        };
        let arguments = cliff_arguments(&mode, Vec::new());
        let rendered: Vec<String> = arguments
            .0
            .iter()
            .map(|argument| return argument.0.clone())
            .collect();
        assert_eq!(
            rendered,
            vec![
                "--output",
                ".git-cliff-output",
                "--skip-commit",
                "aaa",
                "--skip-commit",
                "bbb",
                "--with-commit",
                "fix(repo): tidy (#7)",
            ],
            "each branch commit is skipped and the squash subject replaces them"
        );
    }
}
