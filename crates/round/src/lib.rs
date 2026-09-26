//! The speculative round as infinitum describes it.
//!
//! A round is a graph of typed fragments emitted by components — a drafter, a
//! verifier, an acceptor, and a committer — and split at one host boundary
//! into an execute phase and a commit phase. infinitum owns the graph and the
//! host's decision at that boundary; a backend plans the graph onto its own
//! programs, recognizing subgraphs it has a fused implementation for, and
//! refuses a graph it cannot run, with the reason.

mod component;
mod fragment;
mod graph;
mod maybe;
mod plan;
mod preview;
mod token;

pub use component::Acceptor;
pub use component::CausalBlockVerifier;
pub use component::Committer;
pub use component::DFlash2Drafter;
pub use component::Drafter;
pub use component::GdnReplayCommitter;
pub use component::Licensed;
pub use component::Proposal;
pub use component::SparseRejectionAcceptor;
pub use component::Verification;
pub use component::Verifier;
pub use component::compose;
pub use component::dflash2;
pub use fragment::AcceptRule;
pub use fragment::DraftWidth;
pub use fragment::DrafterModel;
pub use fragment::Effect;
pub use fragment::FragmentKind;
pub use fragment::Phase;
pub use fragment::RecurrentFold;
pub use fragment::Selector;
pub use fragment::VerifyMask;
pub use graph::BuildFailure;
pub use graph::Congruence;
pub use graph::Fragment;
pub use graph::FragmentId;
pub use graph::RoundBuilder;
pub use graph::RoundGraph;
pub use maybe::Maybe;
pub use plan::Backend;
pub use plan::BackendName;
pub use plan::Refusal;
pub use plan::RefusalReason;
pub use preview::Preview;
pub use preview::RoundKind;
pub use preview::RoundLimit;
pub use preview::RoundOffer;
pub use preview::RoundRecord;
pub use preview::RoundVerdict;
pub use token::CountOverflow;
pub use token::TokenCount;
pub use token::TokenId;
