//! The host's decision between a round's execute and commit phases.
//!
//! A backend offers each round's licensed tokens before it commits them; the
//! preview answers with a verdict that may narrow the commit. The decision is
//! infinitum's: the backend applies it and commits the same way for the same
//! accepted length. [`Preview`] makes the output-budget decision and keeps the
//! ledger of every round it saw.

use crate::token::CountOverflow;
use crate::token::TokenCount;
use crate::token::TokenId;

/// Which kind of round an offer comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundKind
{
    /// A decode round: accepted drafts, then the correction or bonus token.
    Decode,
    /// The one token prefill finalization produced.
    PrefillFinalization,
}

/// One round's licensed tokens, offered before their commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundOffer<'round>
{
    /// The licensed tokens in order.
    licensed: &'round [TokenId],
    /// The round they come from.
    kind: RoundKind,
}

impl<'round> RoundOffer<'round>
{
    /// An offer of `licensed` from a round of `kind`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        licensed: &'round [TokenId],
        kind: RoundKind,
    ) -> Self
    {
        return Self { licensed, kind };
    }

    /// The licensed tokens.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn licensed(&self) -> &'round [TokenId]
    {
        return self.licensed;
    }

    /// The round's kind.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> RoundKind
    {
        return self.kind;
    }
}

/// The most tokens of one round a commit may keep: at least one, because a
/// committed round always keeps a token.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoundLimit(core::num::NonZeroU32);

impl From<RoundLimit> for core::num::NonZeroU32
{
    /// Unwrap the limit.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(limit: RoundLimit) -> Self
    {
        return limit.0;
    }
}

/// The preview's answer for one round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundVerdict
{
    /// Leave the round to the backend's output policy.
    Continue,
    /// Commit at most this many licensed tokens and finish the request at
    /// the limit.
    Limit(RoundLimit),
}

/// What the preview saw and decided for one round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoundRecord
{
    /// The round's kind.
    kind: RoundKind,
    /// Tokens the round licensed.
    licensed: TokenCount,
    /// Tokens the verdict admitted: all of them, or the limit.
    admitted: TokenCount,
}

impl RoundRecord
{
    /// The round's kind.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> RoundKind
    {
        return self.kind;
    }

    /// Tokens the round licensed.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn licensed(&self) -> TokenCount
    {
        return self.licensed;
    }

    /// Tokens the verdict admitted.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn admitted(&self) -> TokenCount
    {
        return self.admitted;
    }
}

/// The output-budget decision and the ledger of rounds it made it for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preview
{
    /// The most tokens the request may commit.
    budget: TokenCount,
    /// Tokens admitted so far.
    admitted: TokenCount,
    /// Every round reviewed, in order.
    rounds: Vec<RoundRecord>,
    /// Every admitted token, in order.
    tokens: Vec<TokenId>,
}

impl Preview
{
    /// A preview that admits at most `budget` tokens over the request.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(budget: TokenCount) -> Self
    {
        return Self {
            budget,
            admitted: TokenCount::ZERO,
            rounds: Vec::new(),
            tokens: Vec::new(),
        };
    }

    /// Review one round.
    ///
    /// # Specification
    /// - requires: offers arrive in round order.
    /// - ensures: the round is recorded; while budget remains, a round that
    ///   fits is admitted whole and answered [`RoundVerdict::Continue`], and a
    ///   round that overruns is admitted up to the budget and answered
    ///   [`RoundVerdict::Limit`] at the remainder. With no budget left the
    ///   answer is `Continue` and nothing is admitted: a limit of zero does not
    ///   exist, and a backend given the same budget has already finished the
    ///   request, so no such round is offered.
    /// - provides: infinitum's per-round output decision.
    /// - fails: when the round is longer than a count can hold, or the admitted
    ///   total overflows.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`CountOverflow`]: a count left its range.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the budget boundary — a round that fits exactly, one
    ///   that overruns by one, and a round after the budget is spent.
    /// - witness: `tests::a_round_that_fits_continues`
    /// - witness: `tests::an_overrunning_round_is_limited_to_the_remainder`
    /// - witness: `tests::a_spent_budget_admits_nothing`
    #[inline]
    pub fn review(
        &mut self,
        offer: RoundOffer<'_>,
    ) -> Result<RoundVerdict, CountOverflow>
    {
        let licensed = TokenCount::of(offer.licensed)?;
        let remaining = self.budget.saturating_sub(self.admitted);
        let (admitted, verdict) = if licensed <= remaining {
            (licensed, RoundVerdict::Continue)
        }
        else {
            match core::num::NonZeroU32::new(u32::from(remaining)) {
                | Some(limit) => (remaining, RoundVerdict::Limit(RoundLimit(limit))),
                | None => (TokenCount::ZERO, RoundVerdict::Continue),
            }
        };
        let kept = offer
            .licensed
            .iter()
            .zip(0_u32 .. u32::from(admitted))
            .map(|(&token, _)| token);
        self.tokens.extend(kept);
        self.admitted = self.admitted.checked_add(admitted)?;
        self.rounds.push(RoundRecord {
            kind: offer.kind,
            licensed,
            admitted,
        });
        return Ok(verdict);
    }

    /// Every round reviewed, in order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn rounds(&self) -> &[RoundRecord]
    {
        return &self.rounds;
    }

    /// Every admitted token, in order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn tokens(&self) -> &[TokenId]
    {
        return &self.tokens;
    }
}

/// Tests for the budget decision.
#[cfg(test)]
mod tests
{
    use core::num::NonZeroU32;

    use super::Preview;
    use super::RoundKind;
    use super::RoundLimit;
    use super::RoundOffer;
    use super::RoundVerdict;
    use crate::token::TokenCount;
    use crate::token::TokenId;

    /// A round that exactly fills the budget is admitted whole.
    #[test]
    fn a_round_that_fits_continues()
    {
        let mut preview = Preview::new(TokenCount::from(4_u32));
        let span: Vec<TokenId> = (0_i32 .. 4_i32).map(TokenId::from).collect();
        assert_eq!(
            preview.review(RoundOffer::new(&span, RoundKind::Decode)),
            Ok(RoundVerdict::Continue),
            "four tokens fit a budget of four"
        );
        assert_eq!(preview.tokens(), span.as_slice(), "every token is admitted");
        assert_eq!(
            preview.rounds()[0].admitted(),
            TokenCount::from(4_u32),
            "the record agrees"
        );
    }

    /// An overrun is limited to what remains, and only that prefix is
    /// admitted.
    #[test]
    fn an_overrunning_round_is_limited_to_the_remainder()
    {
        let mut preview = Preview::new(TokenCount::from(3_u32));
        let first: Vec<TokenId> = (0_i32 .. 1_i32).map(TokenId::from).collect();
        assert_eq!(
            preview.review(RoundOffer::new(&first, RoundKind::PrefillFinalization)),
            Ok(RoundVerdict::Continue),
            "the prefill token fits"
        );
        let second: Vec<TokenId> = (0_i32 .. 3_i32).map(TokenId::from).collect();
        assert_eq!(
            preview.review(RoundOffer::new(&second, RoundKind::Decode)),
            Ok(RoundVerdict::Limit(RoundLimit(NonZeroU32::new(2).unwrap()))),
            "two of three fit"
        );
        assert_eq!(
            preview.tokens(),
            [
                TokenId::from(0_i32),
                TokenId::from(0_i32),
                TokenId::from(1_i32)
            ]
            .as_slice(),
            "the admitted stream is the prefill token then the kept prefix"
        );
        assert_eq!(
            preview.rounds()[1].licensed(),
            TokenCount::from(3_u32),
            "licensed kept"
        );
    }

    /// With the budget spent, nothing more is admitted and no zero limit is
    /// invented.
    #[test]
    fn a_spent_budget_admits_nothing()
    {
        let mut preview = Preview::new(TokenCount::from(1_u32));
        let span: Vec<TokenId> = (0_i32 .. 1_i32).map(TokenId::from).collect();
        assert_eq!(
            preview.review(RoundOffer::new(&span, RoundKind::PrefillFinalization)),
            Ok(RoundVerdict::Continue),
            "the budget is spent exactly"
        );
        assert_eq!(
            preview.review(RoundOffer::new(&span, RoundKind::Decode)),
            Ok(RoundVerdict::Continue),
            "a spent budget cannot be expressed as a limit"
        );
        assert_eq!(preview.tokens().len(), 1, "the late round admitted nothing");
        assert_eq!(
            preview.rounds()[1].admitted(),
            TokenCount::ZERO,
            "and records so"
        );
    }
}
