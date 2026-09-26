//! Token ids and token counts, the two quantities a round moves.

/// A token id in the backend tokenizer's vocabulary.
///
/// The representation is the one every backend in view uses, a signed 32-bit
/// id, so a backend adapter converts without a range check.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(i32);

impl From<i32> for TokenId
{
    /// Wrap an id.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(id: i32) -> Self
    {
        return Self(id);
    }
}

impl From<TokenId> for i32
{
    /// Unwrap an id.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(id: TokenId) -> Self
    {
        return id.0;
    }
}

impl core::fmt::Display for TokenId
{
    /// Render the id as its decimal value.
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

/// A number of tokens: licensed in a round, committed, or still allowed.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenCount(u32);

impl TokenCount
{
    /// No tokens.
    pub const ZERO: Self = Self(0);

    /// The tokens in `tokens`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the count is `tokens.len()`.
    /// - provides: the size of a licensed span as a count.
    /// - fails: when the span holds more than `u32::MAX` tokens, which no round
    ///   or request in view approaches.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`CountOverflow`]: the span is longer than a count can hold.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on an empty and a non-empty span; the overflow arm
    ///   needs four billion tokens and is carried by `u32::try_from`.
    /// - witness: `tests::a_span_counts_its_tokens`
    #[inline]
    pub fn of(tokens: &[TokenId]) -> Result<Self, CountOverflow>
    {
        return u32::try_from(tokens.len())
            .map(Self)
            .map_err(|_overflow| CountOverflow);
    }

    /// The count plus `other`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the sum.
    /// - provides: accumulation of committed tokens.
    /// - fails: when the sum exceeds `u32::MAX`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`CountOverflow`]: the sum does not fit.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the overflow boundary.
    /// - witness: `tests::a_sum_past_the_maximum_overflows`
    #[inline]
    pub fn checked_add(
        self,
        other: Self,
    ) -> Result<Self, CountOverflow>
    {
        return self.0.checked_add(other.0).map(Self).ok_or(CountOverflow);
    }

    /// The count minus `other`, or zero where `other` is larger.
    ///
    /// # Specification
    /// trivial: saturating subtraction.
    #[inline]
    #[must_use]
    pub const fn saturating_sub(
        self,
        other: Self,
    ) -> Self
    {
        return Self(self.0.saturating_sub(other.0));
    }
}

impl From<u32> for TokenCount
{
    /// Wrap a count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(count: u32) -> Self
    {
        return Self(count);
    }
}

impl From<TokenCount> for u32
{
    /// Unwrap a count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(count: TokenCount) -> Self
    {
        return count.0;
    }
}

impl From<core::num::NonZeroU32> for TokenCount
{
    /// Widen a positive count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(count: core::num::NonZeroU32) -> Self
    {
        return Self(count.get());
    }
}

impl core::fmt::Display for TokenCount
{
    /// Render the count as its decimal value.
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

/// A token count left its range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountOverflow;

impl core::fmt::Display for CountOverflow
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
        return f.write_str("a token count exceeds its 32-bit range");
    }
}

impl core::error::Error for CountOverflow
{
}

/// Tests for the count's boundaries.
#[cfg(test)]
mod tests
{
    use super::CountOverflow;
    use super::TokenCount;
    use super::TokenId;

    /// A span's count is its length, empty included.
    #[test]
    fn a_span_counts_its_tokens()
    {
        assert_eq!(
            TokenCount::of(&[]),
            Ok(TokenCount::ZERO),
            "an empty span counts zero"
        );
        let span = [TokenId::from(3_i32), TokenId::from(-1_i32)];
        assert_eq!(
            TokenCount::of(&span),
            Ok(TokenCount::from(2_u32)),
            "two ids count two"
        );
    }

    /// The last representable sum succeeds and the next one overflows.
    #[test]
    fn a_sum_past_the_maximum_overflows()
    {
        let top = TokenCount::from(u32::MAX);
        assert_eq!(
            TokenCount::from(u32::MAX - 1).checked_add(TokenCount::from(1_u32)),
            Ok(top),
            "the maximum itself is reachable"
        );
        assert_eq!(
            top.checked_add(TokenCount::from(1_u32)),
            Err(CountOverflow),
            "one past the maximum overflows"
        );
    }
}
