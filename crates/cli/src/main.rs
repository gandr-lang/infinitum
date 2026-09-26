//! Driver binary for the infinitum engine.
//!
//! Installs as `infinitum`. The driver owns the argument surface and the
//! process boundary: it parses an invocation, renders the outcome, and leaves.
//! A bare invocation renders the driver's own help. `infinitum generate` runs
//! one prompt through infinitum's DFlash2 round on ninfer's Engine and prints
//! the prompt's ids, the greedy continuation's ids, its text, and the round
//! tallies: the reference an implementation of the same model is compared
//! against.

mod generate;

use std::io::Write as _;

/// The infinitum command line.
#[derive(Debug, clap::Parser)]
#[command(name = "infinitum", version, about)]
struct Cli
{
    /// The command to run; absent, the driver renders its help.
    #[command(subcommand)]
    command: Option<Command>,
}

/// The driver's commands.
#[derive(Debug, clap::Subcommand)]
enum Command
{
    /// Run one prompt through infinitum's DFlash2 round on ninfer: plan,
    /// tokenize, generate greedily, and print the ids, the text, and the round
    /// tallies.
    Generate(generate::Request),
}

/// A failure of an invocation.
#[derive(Debug)]
enum DriverFailure
{
    /// The request failed.
    Request(generate::RequestFailure),
    /// Writing the help or flushing the output failed.
    Output(std::io::Error),
}

impl From<generate::RequestFailure> for DriverFailure
{
    /// Wrap a request failure.
    ///
    /// # Specification
    /// trivial.
    fn from(failure: generate::RequestFailure) -> Self
    {
        return Self::Request(failure);
    }
}

impl From<std::io::Error> for DriverFailure
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

impl core::fmt::Display for DriverFailure
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
            | Self::Request(ref failure) => core::fmt::Display::fmt(failure, f),
            | Self::Output(ref failure) => write!(f, "cannot write the output: {failure}"),
        };
    }
}

/// Build the command definition the driver renders its help from.
///
/// # Specification
/// - requires: nothing of the caller.
/// - ensures: the returned command carries the driver's own name, its version
///   from the package metadata, and the flags and subcommands clap derives for
///   [`Cli`].
/// - provides: the single source of the help text, so the rendered text and the
///   parsed grammar cannot describe different commands.
/// - fails: never; the definition is derived at compile time.
/// - panics: none. Clap's command-definition assertions run under
///   `debug_assertions` and are not reached by this construction.
///
/// # Adequacy
/// - hypothesis: L3 only — the definition is one derived value, and the
///   properties worth distinguishing are the name and the version it reports,
///   each asserted exactly.
/// - witness: `tests::command_carries_the_driver_name`
/// - witness: `tests::command_reports_the_package_version`
#[quenchant::spec(ensures: |ref command| !command.get_name().is_empty())]
fn command() -> clap::Command
{
    return <Cli as clap::CommandFactory>::command();
}

/// Render the driver's long help text to `out`.
///
/// # Specification
/// - requires: `out` accepts the bytes clap writes; a closed or full writer is
///   a failure, not a precondition violation.
/// - ensures: on success the whole long-help rendering of [`command`] has been
///   written to `out`, and nothing else.
/// - provides: the driver's output for a bare invocation.
/// - fails: returns the writer's error unchanged, so a closed or full
///   destination surfaces at the call site rather than aborting the process.
/// - panics: none. The rendering goes through the fallible writer rather than
///   `println!`, whose internal write-failure path aborts.
///
/// # Errors
/// - [`std::io::Error`]: the underlying failure from writing the help text to
///   `out`.
///
/// # Adequacy
/// - hypothesis: L3 only — one write of a derived rendering, whose specified
///   residue is that the text reaching `out` is the help for this command
///   rather than an empty or foreign rendering.
/// - witness: `tests::long_help_describes_the_driver`
fn render_long_help<Writer>(out: &mut Writer) -> Result<(), std::io::Error>
where
    Writer: std::io::Write,
{
    return command().write_long_help(out);
}

/// Run a parsed invocation, writing its output to `out`.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: with no command, `out` holds the long help; with `generate`, it
///   holds the request's rendering; either way `out` is flushed on success.
/// - provides: the driver's behavior, apart from the process boundary.
/// - fails: with the request's failure, or with the writer's error.
/// - panics: none.
///
/// # Errors
/// - [`DriverFailure::Request`]: the request failed.
/// - [`DriverFailure::Output`]: writing or flushing `out` failed.
///
/// # Adequacy
/// - hypothesis: L3 over the two command shapes — none renders help, and
///   `generate` reaches the request, observed by its planning refusal.
/// - witness: `tests::a_bare_invocation_renders_help`
/// - witness: `tests::generate_reaches_the_request`
fn run<Writer>(
    cli: Cli,
    out: &mut Writer,
) -> Result<(), DriverFailure>
where
    Writer: std::io::Write,
{
    match cli.command {
        | None => render_long_help(out)?,
        | Some(Command::Generate(request)) => generate::run(&request, out)?,
    }
    out.flush()?;
    return Ok(());
}

/// Parse the driver's arguments, run them, and report a failure on standard
/// error.
///
/// # Specification
/// - requires: nothing of the caller; the arguments come from the process
///   environment.
/// - ensures: [`Cli`] parses before anything is written, so `--help`,
///   `--version`, and an argument error take clap's own exit path; otherwise
///   [`run`]'s output is on standard output, and a failure is one `error: `
///   line on standard error through its `Display` rendering.
/// - provides: the process boundary: success exits zero, failure nonzero.
/// - fails: never as a function; a failure becomes the exit status.
/// - panics: none. The handles are locked once and written through fallible
///   calls; clap leaves by process exit rather than by panic on `--help`,
///   `--version`, and argument errors.
///
/// # Adequacy
/// - hypothesis: none in the suite — the boundary is a process, and [`run`]'s
///   witnesses carry the behavior; the device smoke runs the binary.
fn main() -> std::process::ExitCode
{
    let cli = <Cli as clap::Parser>::parse();
    let mut stdout = std::io::stdout().lock();
    let outcome = run(cli, &mut stdout);
    drop(stdout);
    if let Err(failure) = outcome {
        let mut stderr = std::io::stderr().lock();
        let written = writeln!(stderr, "error: {failure}");
        drop(written);
        return std::process::ExitCode::FAILURE;
    }
    return std::process::ExitCode::SUCCESS;
}

/// Tests for the driver's command definition, its rendered help, and its
/// dispatch.
#[cfg(test)]
mod tests
{
    use infinitum_round::RefusalReason;

    use super::Cli;
    use super::DriverFailure;
    use super::command;
    use super::render_long_help;
    use super::run;
    use crate::generate::RequestFailure;

    /// The command renders under the name the binary installs as.
    #[test]
    fn command_carries_the_driver_name()
    {
        assert_eq!(
            command().get_name(),
            "infinitum",
            "the command names the installed binary"
        );
    }

    /// The command reports the package version, which is what `--version`
    /// prints.
    #[test]
    fn command_reports_the_package_version()
    {
        assert_eq!(
            command().get_version(),
            Some(env!("CARGO_PKG_VERSION")),
            "the command reports the package version"
        );
    }

    /// The rendered long help describes this driver: its usage line names the
    /// binary, and the flags and the command clap derives are listed.
    #[test]
    fn long_help_describes_the_driver()
    {
        let mut rendered = Vec::new();
        render_long_help(&mut rendered).unwrap();
        let rendered = String::from_utf8(rendered).unwrap();
        assert!(
            rendered.contains("Usage: infinitum"),
            "the help names the binary: {rendered}"
        );
        assert!(
            rendered.contains("--help"),
            "the help lists its own flag: {rendered}"
        );
        assert!(
            rendered.contains("--version"),
            "the help lists the version flag: {rendered}"
        );
        assert!(
            rendered.contains("generate"),
            "the help lists the generate command: {rendered}"
        );
    }

    /// A bare invocation writes the long help.
    #[test]
    fn a_bare_invocation_renders_help()
    {
        let cli = <Cli as clap::Parser>::try_parse_from(["infinitum"]).unwrap();
        let mut out = Vec::new();
        run(cli, &mut out).unwrap();
        let mut expected = Vec::new();
        render_long_help(&mut expected).unwrap();
        assert_eq!(out, expected, "the output is exactly the long help");
    }

    /// `generate` runs the request, observed through its planning refusal.
    #[test]
    fn generate_reaches_the_request()
    {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "infinitum",
            "generate",
            "--artifact",
            "model.ninfer",
            "--draft-width",
            "16",
            "hello",
        ])
        .unwrap();
        let mut out = Vec::new();
        let failed = run(cli, &mut out);
        assert!(
            matches!(
                failed,
                Err(DriverFailure::Request(RequestFailure::Plan(ref refusal)))
                    if matches!(refusal.reason(), RefusalReason::UnsupportedWidth(_))
            ),
            "the request's planning refusal is reported: {failed:?}"
        );
    }

    /// A zero token budget is an argument error, before anything runs.
    #[test]
    fn a_zero_budget_is_an_argument_error()
    {
        let parsed = <Cli as clap::Parser>::try_parse_from([
            "infinitum",
            "generate",
            "--artifact",
            "model.ninfer",
            "--max-new-tokens",
            "0",
            "hello",
        ]);
        let failure = parsed.unwrap_err();
        assert_eq!(
            failure.kind(),
            clap::error::ErrorKind::ValueValidation,
            "zero is refused as a value"
        );
    }

    /// The artifact and the prompt are required.
    #[test]
    fn generate_requires_its_artifact_and_prompt()
    {
        let parsed = <Cli as clap::Parser>::try_parse_from(["infinitum", "generate", "hello"]);
        let failure = parsed.unwrap_err();
        assert_eq!(
            failure.kind(),
            clap::error::ErrorKind::MissingRequiredArgument,
            "the missing artifact is reported"
        );
    }
}
