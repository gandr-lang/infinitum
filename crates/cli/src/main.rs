//! Driver binary for the infinitum engine.
//!
//! Installs as `infinitum`. The driver owns the argument surface and the
//! process boundary: it parses an invocation, renders the outcome, and leaves.
//! The engine itself arrives here as it lands; until then the one accepted
//! invocation renders the driver's own help, so the binary's observable
//! surface is exactly what `--help` and `--version` describe.

use std::io::Write as _;

/// The infinitum command line.
#[derive(Debug, clap::Parser)]
#[command(name = "infinitum", version, about)]
struct Cli;

/// Build the command definition the driver renders its help from.
///
/// # Specification
/// - requires: nothing of the caller.
/// - ensures: the returned command carries the driver's own name, its version
///   from the package metadata, and the flags clap derives for [`Cli`].
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
/// - provides: the driver's entire observable output while the engine has no
///   invocation of its own.
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

/// Parse the driver's arguments and render the outcome.
///
/// # Specification
/// - requires: nothing of the caller; the arguments come from the process
///   environment and the only accepted form is argument-free.
/// - ensures: [`Cli`] parses before anything is written, so `--help`,
///   `--version`, and an argument error take clap's own exit path, and the help
///   rendering is reached only on a bare `infinitum` invocation.
/// - provides: the long help on standard output, flushed before the process
///   returns.
/// - fails: returns the write or flush error when standard output is closed,
///   full, or otherwise unwritable; the runtime reports it and exits nonzero.
/// - panics: none. The handle is locked once and written through fallible
///   calls; clap leaves by process exit rather than by panic on `--help`,
///   `--version`, and argument errors.
///
/// # Errors
/// - [`std::io::Error`]: the underlying failure from writing the help text to
///   standard output, or from the flush that follows it.
fn main() -> Result<(), std::io::Error>
{
    let _cli = <Cli as clap::Parser>::parse();
    let mut stdout = std::io::stdout().lock();
    render_long_help(&mut stdout)?;
    stdout.flush()?;
    return Ok(());
}

/// Tests for the driver's command definition and its rendered help.
#[cfg(test)]
mod tests
{
    use super::command;
    use super::render_long_help;

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
    /// binary, and the flags clap derives are listed.
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
    }
}
