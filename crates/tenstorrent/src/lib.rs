//! infinitum's Tenstorrent backend, as a spike.
//!
//! The Accept fragment under sparse rejection at temperature zero is lowered
//! through tt-mlir's TTIR-to-TTMetal kernel path and run on a Blackhole
//! device. [`reference_accept`] is the host answer the device must agree
//! with; [`Sample`] generates the verify blocks the differential compares on.
//! [`Tenstorrent`] plans a round fragment by fragment; Accept is its one
//! lowering.
//! With the `device` feature, `AcceptProgram` builds and lowers the module
//! in process, or loads one the pipeline tools lowered, and
//! `TenstorrentDevice` runs it.

mod accept;
#[cfg(feature = "device")]
mod bridge;
#[cfg(feature = "device")]
mod device;
mod digits;
mod plan;
mod samples;

pub use accept::Acceptance;
pub use accept::Bf16;
pub use accept::ChunkCount;
pub use accept::ChunkWidth;
pub use accept::DeviceBits;
pub use accept::DraftBlock;
pub use accept::LayoutFailure;
pub use accept::PaddedWidth;
pub use accept::ShapeMismatch;
pub use accept::VerifyLogits;
pub use accept::VocabularyLayout;
pub use accept::VocabularySize;
pub use accept::reference_accept;
pub use accept::reference_targets;
#[cfg(feature = "device")]
pub use device::AcceptGeometry;
#[cfg(feature = "device")]
pub use device::AcceptProgram;
#[cfg(feature = "device")]
pub use device::CompileCost;
#[cfg(feature = "device")]
pub use device::DeviceAnswer;
#[cfg(feature = "device")]
pub use device::HostFailure;
#[cfg(feature = "device")]
pub use device::Nanos;
#[cfg(feature = "device")]
pub use device::Operation;
#[cfg(feature = "device")]
pub use device::PrefixSite;
#[cfg(feature = "device")]
pub use device::RunCost;
#[cfg(feature = "device")]
pub use device::SystemDescriptor;
#[cfg(feature = "device")]
pub use device::TenstorrentDevice;
#[cfg(feature = "device")]
pub use device::print_accept;
#[cfg(feature = "device")]
pub use device::save_system_desc;
pub use digits::Digit;
pub use digits::DigitFailure;
pub use digits::IdDigits;
pub use digits::compose;
pub use digits::constant_planes;
pub use digits::fill_tail;
pub use digits::split;
pub use plan::AcceptPlan;
pub use plan::Lowering;
pub use plan::RoundPlan;
pub use plan::Tenstorrent;
pub use samples::Sample;
pub use samples::SampleCase;
pub use samples::Seed;
