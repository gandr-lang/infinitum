//! What a backend reports while it opens: each startup phase's beginning,
//! progress, completion or failure.

use crate::runtime::ByteSize;

/// A phase of opening a backend, in the order they begin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartupPhase
{
    /// The whole of opening; every other phase runs inside it.
    EngineStartup,
    /// Bringing up the CUDA context.
    CudaInitialize,
    /// Reading the artifact's metadata.
    ArtifactInspect,
    /// Planning the runtime for the target model.
    TargetPlan,
    /// Uploading the weights to the device.
    WeightsMaterialize,
    /// Pinning the host staging buffers.
    WeightsStagingPin,
    /// Finalizing the target model.
    TargetFinalize,
    /// Initializing the tokenizer and chat template.
    FrontendInitialize,
    /// Initializing the runtime program.
    ProgramInitialize,
    /// Pinning host memory for cached states.
    HostStatePin,
    /// Pinning host memory for cached KV.
    HostKvPin,
    /// Capturing the CUDA graphs.
    CudaGraphPrepare,
    /// Finalizing the engine.
    EngineFinalize,
}

/// Where a phase is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartupStatus
{
    /// It began.
    Begin,
    /// It advanced.
    Progress,
    /// It finished.
    Complete,
    /// It ended by failing.
    Failed,
}

/// How much of a phase's work is done, when the phase counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartupAmount
{
    /// The phase counts nothing.
    Untracked,
    /// The phase counts bytes.
    Bytes
    {
        /// Bytes done.
        done: ByteSize,
        /// Bytes in all, zero when not yet known.
        total: ByteSize,
    },
}

/// One report from a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StartupEvent
{
    /// The phase.
    pub phase: StartupPhase,
    /// Where it is.
    pub status: StartupStatus,
    /// How much of its work is done.
    pub amount: StartupAmount,
    /// The phase's duration when it completes or fails; zero otherwise.
    pub elapsed: core::time::Duration,
}

/// The consumer of a backend's startup reports.
pub trait StartupObserver
{
    /// Take one report.
    ///
    /// # Specification
    /// - requires: called on the opening thread, in the order the phases
    ///   report, and only while the backend opens.
    /// - ensures: the report was taken; nothing it does changes how opening
    ///   goes.
    /// - provides: startup diagnostics.
    /// - fails: never.
    /// - panics: none.
    fn observe(
        &mut self,
        event: StartupEvent,
    );
}

/// An observer for a caller that reports no startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unobserved;

impl StartupObserver for Unobserved
{
    /// Leave the report unread.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn observe(
        &mut self,
        _event: StartupEvent,
    )
    {
    }
}
