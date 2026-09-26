//! ninfer's operational number formats, so a log line reads the same from
//! either server.

use core::fmt;

/// A count with thousands separators: `1,234,567`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Count(pub u64);

impl fmt::Display for Count
{
    /// Render the count, a comma before every group of three digits from
    /// the right.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the decimal digits of the count, with `,` before each group
    ///   of three counted from the right, as ninfer's `format_pretty_count`.
    /// - provides: every count in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of each separator boundary.
    /// - witness: `tests::formats_match_ninfer`
    #[inline]
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        let digits = self.0.to_string();
        let mut remaining = digits.len();
        for digit in digits.chars() {
            if remaining != digits.len() && remaining.is_multiple_of(3) {
                f.write_str(",")?;
            }
            write!(f, "{digit}")?;
            remaining = remaining.saturating_sub(1);
        }
        return Ok(());
    }
}

/// A duration in seconds: `850 us`, `12.5 ms`, `3.2s`, `2m 5.0s`, `1h 3m`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Duration(pub f64);

impl fmt::Display for Duration
{
    /// Render the duration in ninfer's units.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `n/a` for a negative or non-finite duration; below a
    ///   millisecond, whole microseconds; below a second, milliseconds with two
    ///   decimals under ten, one under a hundred, none above; below a minute,
    ///   seconds with one decimal and no space; below an hour, whole minutes
    ///   and one-decimal seconds; above, whole hours and minutes, as ninfer's
    ///   `format_pretty_duration`.
    /// - provides: every duration in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of each unit boundary.
    /// - witness: `tests::formats_match_ninfer`
    #[expect(
        clippy::suboptimal_flops,
        reason = "ninfer rounds each product and difference separately"
    )]
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        let seconds = self.0;
        if !seconds.is_finite() || seconds < 0.0_f64 {
            return f.write_str("n/a");
        }
        if seconds < 0.001_f64 {
            return write!(f, "{:.0} us", seconds * 1.0e6_f64);
        }
        if seconds < 1.0_f64 {
            let milliseconds = seconds * 1.0e3_f64;
            let precision = usize::from(milliseconds < 10.0_f64)
                .saturating_add(usize::from(milliseconds < 100.0_f64));
            return write!(f, "{milliseconds:.precision$} ms");
        }
        if seconds < 60.0_f64 {
            return write!(f, "{seconds:.1}s");
        }
        if seconds < 3600.0_f64 {
            let minutes = (seconds / 60.0_f64).floor();
            return write!(f, "{minutes:.0}m {:.1}s", seconds - 60.0_f64 * minutes);
        }
        let hours = (seconds / 3600.0_f64).floor();
        let minutes = ((seconds - 3600.0_f64 * hours) / 60.0_f64).floor();
        return write!(f, "{hours:.0}h {minutes:.0}m");
    }
}

/// A ratio as a percentage: `42.5%`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percent(pub f64);

impl fmt::Display for Percent
{
    /// Render the ratio as a one-decimal percentage.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `n/a` for a negative or non-finite ratio; otherwise the ratio
    ///   times a hundred with one decimal and `%`, as ninfer's
    ///   `format_pretty_percent`.
    /// - provides: every ratio in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on the refusal and one value.
    /// - witness: `tests::formats_match_ninfer`
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        if !self.0.is_finite() || self.0 < 0.0_f64 {
            return f.write_str("n/a");
        }
        return write!(f, "{:.1}%", self.0 * 100.0_f64);
    }
}

/// A token rate per second: `812.4 tok/s`, `1.25k tok/s`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenRate(pub f64);

impl fmt::Display for TokenRate
{
    /// Render the rate with a decimal prefix.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `n/a` for a negative or non-finite rate; otherwise the rate
    ///   scaled by thousands to `k`, `M`, `G` or `T`, with one decimal
    ///   unscaled, and once scaled two decimals under ten and one above, then `
    ///   tok/s`, as ninfer's `format_pretty_rate` with unit `tok`.
    /// - provides: every rate in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of the first prefix boundary and at the
    ///   scaled precision switch.
    /// - witness: `tests::formats_match_ninfer`
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        let mut rate = self.0;
        if !rate.is_finite() || rate < 0.0_f64 {
            return f.write_str("n/a");
        }
        let mut prefix = "";
        for next in ["k", "M", "G", "T"] {
            if rate < 1000.0_f64 {
                break;
            }
            rate /= 1000.0_f64;
            prefix = next;
        }
        let precision = if prefix.is_empty() || rate >= 10.0_f64 {
            1
        }
        else {
            2
        };
        return write!(f, "{rate:.precision$}{prefix} tok/s");
    }
}

/// A byte size in binary units: `512 B`, `1.50 KiB`, `36.0 GiB`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bytes(pub u64);

impl fmt::Display for Bytes
{
    /// Render the size in the largest binary unit it reaches.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: under a KiB, the exact count and ` B`; otherwise the size
    ///   divided by 1024 per unit up to EiB, two decimals under ten and one
    ///   above, as ninfer's `format_pretty_bytes`.
    /// - provides: every size in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 either side of the first unit boundary.
    /// - witness: `tests::formats_match_ninfer`
    #[inline]
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        if self.0 < 1024 {
            return write!(f, "{} B", self.0);
        }
        // `u64` to `f64` rounded once, as a C++ `static_cast<double>`: the
        // high half is exact after scaling, so the sum rounds only once.
        let high = u32::try_from(self.0 >> 32_u32).unwrap_or(u32::MAX);
        let low = u32::try_from(self.0 & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        let mut value = f64::from(high).mul_add(4_294_967_296.0_f64, f64::from(low));
        let mut unit = "B";
        for next in ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB"] {
            if value < 1024.0_f64 {
                break;
            }
            value /= 1024.0_f64;
            unit = next;
        }
        let precision = if value < 10.0_f64 { 2 } else { 1 };
        return write!(f, "{value:.precision$} {unit}");
    }
}

/// Text made safe for one log line: newlines and tabs become spaces and
/// other control bytes `?`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Text<'text>(pub &'text str);

impl fmt::Display for Text<'_>
{
    /// Render the text with its control characters replaced.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `\n`, `\r` and `\t` become a space and every other C0 control
    ///   or DEL becomes `?`, as ninfer's `format_pretty_text`.
    /// - provides: every client- or model-supplied text in a log line.
    /// - fails: when the formatter fails.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`fmt::Error`]: the formatter failed.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on each replaced class and a kept character.
    /// - witness: `tests::formats_match_ninfer`
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        for character in self.0.chars() {
            let shown = match character {
                | '\n' | '\r' | '\t' => ' ',
                | '\u{0}' ..= '\u{1f}' | '\u{7f}' => '?',
                | other => other,
            };
            write!(f, "{shown}")?;
        }
        return Ok(());
    }
}

/// Tests for the formats against ninfer's.
#[cfg(test)]
mod tests
{
    use super::Bytes;
    use super::Count;
    use super::Duration;
    use super::Percent;
    use super::Text;
    use super::TokenRate;

    #[test]
    fn formats_match_ninfer()
    {
        for (count, shown) in [
            (0_u64, "0"),
            (999, "999"),
            (1000, "1,000"),
            (1_234_567, "1,234,567"),
        ] {
            assert_eq!(Count(count).to_string(), shown, "count {count}");
        }
        for (seconds, shown) in [
            (-1.0_f64, "n/a"),
            (0.000_85_f64, "850 us"),
            (0.005_f64, "5.00 ms"),
            (0.0125_f64, "12.5 ms"),
            (0.25_f64, "250 ms"),
            (3.21_f64, "3.2s"),
            (125.0_f64, "2m 5.0s"),
            (3780.0_f64, "1h 3m"),
        ] {
            assert_eq!(Duration(seconds).to_string(), shown, "duration {seconds}");
        }
        assert_eq!(Percent(0.425_f64).to_string(), "42.5%", "percent");
        assert_eq!(Percent(f64::NAN).to_string(), "n/a", "percent of nothing");
        for (rate, shown) in [
            (812.44_f64, "812.4 tok/s"),
            (1250.0_f64, "1.25k tok/s"),
            (12_500.0_f64, "12.5k tok/s"),
        ] {
            assert_eq!(TokenRate(rate).to_string(), shown, "rate {rate}");
        }
        for (bytes, shown) in [
            (1023_u64, "1023 B"),
            (1536, "1.50 KiB"),
            (36_864_u64 << 20_u32, "36.0 GiB"),
            (5_u64 << 40_u32, "5.00 TiB"),
        ] {
            assert_eq!(Bytes(bytes).to_string(), shown, "bytes {bytes}");
        }
        assert_eq!(Text("a\tb\nc\u{1}\u{7f}é").to_string(), "a b c??é", "text");
    }
}
