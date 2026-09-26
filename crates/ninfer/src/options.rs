//! How an Engine is opened: the artifact, the device, the context ceiling,
//! CUDA graph capture, and the chat template.

/// The logical ceiling of one request in tokens, prompt and generation
/// together. The Engine also sizes its KV cache from it, so a small ceiling
/// keeps a one-request run's device footprint small.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLimit(core::num::NonZeroU32);

impl From<ContextLimit> for core::num::NonZeroU32
{
    /// Unwrap the limit.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(limit: ContextLimit) -> Self
    {
        return limit.0;
    }
}

impl core::str::FromStr for ContextLimit
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the limit is the parsed value, at least one.
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
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

/// A CUDA device ordinal, between zero and `u16::MAX`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOrdinal(u16);

impl From<DeviceOrdinal> for i32
{
    /// Widen the ordinal to the Engine's `int`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(ordinal: DeviceOrdinal) -> Self
    {
        return Self::from(ordinal.0);
    }
}

impl core::str::FromStr for DeviceOrdinal
{
    type Err = core::num::ParseIntError;

    /// Parse a non-negative decimal device ordinal.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the ordinal is the parsed value; a negative
    ///   ordinal names no device and is refused here rather than by the Engine.
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
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<u16>().map(Self);
    }
}

/// Whether the Engine captures its decode rounds as CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraph
{
    /// Launch every kernel directly.
    Off,
    /// Capture decode rounds as CUDA graphs, as the served engine does.
    On,
}

impl core::str::FromStr for CudaGraph
{
    type Err = UnknownCudaGraph;

    /// Parse `off` or `on`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `off` is [`CudaGraph::Off`] and `on` is [`CudaGraph::On`].
    /// - provides: the command-line spelling of the choice.
    /// - fails: on any other text, case included.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`UnknownCudaGraph`]: `text` is neither spelling.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over both spellings and one refusal.
    /// - witness: `tests::cuda_graph_parses_its_two_spellings`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return match text {
            | "off" => Ok(Self::Off),
            | "on" => Ok(Self::On),
            | _ => Err(UnknownCudaGraph),
        };
    }
}

/// A CUDA graph choice that is neither `off` nor `on`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownCudaGraph;

impl core::fmt::Display for UnknownCudaGraph
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
        return f.write_str("CUDA graph capture is `off` or `on`");
    }
}

impl core::error::Error for UnknownCudaGraph
{
}

/// Which chat template renders a chat prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate
{
    /// The template the artifact embeds.
    Artifact,
    /// The Jinja template in this file.
    File(std::path::PathBuf),
}

/// Everything an Engine is opened with except the round, which the plan
/// supplies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions
{
    /// The `.ninfer` artifact entry file.
    artifact: std::path::PathBuf,
    /// The CUDA device.
    device: DeviceOrdinal,
    /// The context ceiling.
    context: ContextLimit,
    /// CUDA graph capture.
    cuda_graph: CudaGraph,
    /// The chat template.
    chat_template: ChatTemplate,
}

impl EngineOptions
{
    /// Gather the options, with the artifact's own chat template.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        artifact: std::path::PathBuf,
        device: DeviceOrdinal,
        context: ContextLimit,
        cuda_graph: CudaGraph,
    ) -> Self
    {
        return Self {
            artifact,
            device,
            context,
            cuda_graph,
            chat_template: ChatTemplate::Artifact,
        };
    }

    /// The same options with `template` rendering chat prompts.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn with_chat_template(
        self,
        template: ChatTemplate,
    ) -> Self
    {
        return Self {
            chat_template: template,
            ..self
        };
    }

    /// The chat template.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn chat_template(&self) -> &ChatTemplate
    {
        return &self.chat_template;
    }

    /// The artifact.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn artifact(&self) -> &std::path::Path
    {
        return &self.artifact;
    }

    /// The device.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn device(&self) -> DeviceOrdinal
    {
        return self.device;
    }

    /// The context ceiling.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn context(&self) -> ContextLimit
    {
        return self.context;
    }

    /// CUDA graph capture.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn cuda_graph(&self) -> CudaGraph
    {
        return self.cuda_graph;
    }
}

/// Tests for the options' parsing boundaries.
#[cfg(test)]
mod tests
{
    use core::str::FromStr as _;

    use super::ContextLimit;
    use super::CudaGraph;
    use super::DeviceOrdinal;
    use super::UnknownCudaGraph;

    /// A zero ceiling is refused and one is admitted.
    #[test]
    fn a_zero_context_is_refused()
    {
        assert!(
            ContextLimit::from_str("0").is_err(),
            "a request needs at least one token"
        );
        assert!(
            ContextLimit::from_str("1").is_ok(),
            "one token is a ceiling"
        );
    }

    /// Minus one names no device; zero does.
    #[test]
    fn a_negative_device_is_refused()
    {
        assert!(
            DeviceOrdinal::from_str("-1").is_err(),
            "no device is negative"
        );
        assert_eq!(
            DeviceOrdinal::from_str("0").map(i32::from),
            Ok(0_i32),
            "device zero"
        );
    }

    /// The two spellings parse; anything else is refused.
    #[test]
    fn cuda_graph_parses_its_two_spellings()
    {
        assert_eq!(CudaGraph::from_str("off"), Ok(CudaGraph::Off), "off");
        assert_eq!(CudaGraph::from_str("on"), Ok(CudaGraph::On), "on");
        assert_eq!(
            CudaGraph::from_str("On"),
            Err(UnknownCudaGraph),
            "case matters"
        );
    }
}
