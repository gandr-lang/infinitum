//! ninfer's planner: the DFlash2 round, run whole by ninfer's fused Engine.
//!
//! ninfer implements one round as a single fusion and has no
//! fragment-by-fragment lowering, so a graph plans exactly when it is the
//! canonical DFlash2 round at a width ninfer supports. Anything else is
//! refused with the fragment where it leaves that round.

use infinitum_round::Backend;
use infinitum_round::BackendName;
use infinitum_round::DraftWidth;
use infinitum_round::DrafterModel;
use infinitum_round::FragmentKind;
use infinitum_round::Maybe;
use infinitum_round::Refusal;
use infinitum_round::RefusalReason;
use infinitum_round::RoundGraph;

/// The widest DFlash2 round ninfer runs: `K` up to 15, a verify of 16 query
/// columns.
const MAX_WIDTH: u32 = 15;

/// The ninfer backend, as a planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ninfer;

/// A graph ninfer runs: its fused DFlash2 round at one width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2Plan
{
    /// The draft width the Engine is opened with.
    width: DraftWidth,
}

impl DFlash2Plan
{
    /// The draft width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> DraftWidth
    {
        return self.width;
    }
}

impl Backend for Ninfer
{
    type Plan = DFlash2Plan;

    /// Plan `graph` as ninfer's fused DFlash2 round.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the graph is `infinitum_round::dflash2(width)` for
    ///   the width of its first DFlash2 block forward, and that width is at
    ///   most 15.
    /// - provides: the one gate between a round and ninfer's Engine.
    /// - fails: with no DFlash2 block forward in the graph, a width above 15,
    ///   or the first fragment where the graph differs from the canonical
    ///   round.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`Refusal`]: no known drafter, an unsupported width, no known fusion,
    ///   or the canonical round's own composition refused, each as its
    ///   [`RefusalReason`].
    ///
    /// # Adequacy
    /// - hypothesis: L3 — the canonical round at the widest width plans; one
    ///   past it, a round with no drafter, and a round that drops its last
    ///   commit fragment are each refused with their reason.
    /// - witness: `tests::the_canonical_round_plans`
    /// - witness: `tests::a_width_past_fifteen_is_refused`
    /// - witness: `tests::a_round_without_a_drafter_is_refused`
    /// - witness: `tests::a_different_round_is_refused_where_it_differs`
    #[inline]
    fn plan(
        &self,
        graph: &RoundGraph,
    ) -> Result<DFlash2Plan, Refusal>
    {
        let refuse = |reason| return Refusal::new(BackendName::Ninfer, reason);
        let width = graph
            .fragments()
            .iter()
            .find_map(|fragment| {
                return match fragment.kind() {
                    | FragmentKind::DraftBlockForward {
                        model: DrafterModel::DFlash2,
                        width,
                    } => Some(width),
                    | _ => None,
                };
            })
            .ok_or_else(|| return refuse(RefusalReason::NoKnownDrafter))?;
        if core::num::NonZeroU32::from(width).get() > MAX_WIDTH {
            return Err(refuse(RefusalReason::UnsupportedWidth(width)));
        }
        let canonical = infinitum_round::dflash2(width)
            .map_err(|failure| return refuse(RefusalReason::Composition(failure)))?;
        return match graph.divergence(&canonical) {
            | Maybe::Absent(_) => Ok(DFlash2Plan { width }),
            | Maybe::Present(at) => Err(refuse(RefusalReason::NoKnownFusion(at))),
        };
    }
}

/// Tests for the planner's gate.
#[cfg(test)]
mod tests
{
    use core::str::FromStr as _;

    use infinitum_round::AcceptRule;
    use infinitum_round::Backend as _;
    use infinitum_round::DraftWidth;
    use infinitum_round::FragmentKind;
    use infinitum_round::RefusalReason;
    use infinitum_round::RoundBuilder;

    use super::Ninfer;

    /// The canonical round at the widest supported width plans at that width.
    #[test]
    fn the_canonical_round_plans()
    {
        let width = DraftWidth::from_str("15").unwrap();
        let graph = infinitum_round::dflash2(width).unwrap();
        assert_eq!(
            Ninfer.plan(&graph).map(|plan| plan.width()),
            Ok(width),
            "K = 15 runs"
        );
    }

    /// One past the widest width is refused, naming the width.
    #[test]
    fn a_width_past_fifteen_is_refused()
    {
        let width = DraftWidth::from_str("16").unwrap();
        let graph = infinitum_round::dflash2(width).unwrap();
        assert_eq!(
            Ninfer.plan(&graph).map_err(|refusal| refusal.reason()),
            Err(RefusalReason::UnsupportedWidth(width)),
            "ninfer's verify is at most sixteen columns"
        );
    }

    /// A round with no DFlash2 drafter has nothing ninfer knows.
    #[test]
    fn a_round_without_a_drafter_is_refused()
    {
        let mut builder = RoundBuilder::new();
        let accept = builder
            .add(
                FragmentKind::Accept {
                    rule: AcceptRule::SparseRejection,
                },
                &[],
            )
            .unwrap();
        builder.add(FragmentKind::CommitKv, &[accept]).unwrap();
        let graph = builder.finish().unwrap();
        assert_eq!(
            Ninfer.plan(&graph).map_err(|refusal| refusal.reason()),
            Err(RefusalReason::NoKnownDrafter),
            "no drafter, no fusion"
        );
    }

    /// A DFlash2 round missing its feature publication differs from the
    /// canonical round at the publication's position.
    #[test]
    fn a_different_round_is_refused_where_it_differs()
    {
        let width = DraftWidth::from_str("7").unwrap();
        let canonical = infinitum_round::dflash2(width).unwrap();
        let mut builder = RoundBuilder::new();
        let fragments = canonical.fragments();
        let (_, kept) = fragments.split_last().unwrap();
        for fragment in kept {
            builder.add(fragment.kind(), fragment.inputs()).unwrap();
        }
        let truncated = builder.finish().unwrap();
        let Err(refusal) = Ninfer.plan(&truncated)
        else {
            panic!("a round without feature publication is not ninfer's fusion");
        };
        assert_eq!(
            refusal.to_string(),
            "ninfer refuses the round: it differs from its known fusion at fragment #8",
            "the refusal names the backend and the missing fragment"
        );
    }
}
