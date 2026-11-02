"""The smallest mean contrast a run of corrected intervals detects, before the run is made.

A contrast's interval is Nadeau and Bengio's corrected resampled t (`bko.evaluate.criteria`): its
half-width is `t * sqrt((1/J + n2/n1) S^2)` over J holdouts. A design detects a mean contrast at a
given power when the mean clears that half-width by the power's own t quantile, so the minimum
detectable effect adds both quantiles over one scale.
"""

from __future__ import annotations

import math

from scipy.stats import t as student


def mde(
    sd: float, *, holdouts: int, ratio: float, confidence: float = 0.95, power: float = 0.8
) -> float:
    """The minimum detectable mean contrast of a corrected interval with one family member.

    `(t(1 - (1 - confidence)/2, J - 1) + t(power, J - 1)) * sqrt((1/J + ratio) * sd^2)`, J the
    holdouts. At a power of 0.5 the second quantile is 0, so it is the interval's half-width.

    Args:
        sd: The contrast's standard deviation over holdouts.
        holdouts: The holdouts, J; 2 or more.
        ratio: n2/n1, held-out over training records.
        confidence: The interval's two-sided level.
        power: The chance of reading a true effect of this size above zero.

    Returns:
        The smallest mean contrast the interval reads above zero at `power`.

    Raises:
        TypeError: If the holdouts are not an int.
        ValueError: If there are fewer than 2 holdouts, the standard deviation or the ratio is NaN
            or below 0, or the confidence or power is not in (0, 1).
    """
    if isinstance(holdouts, bool) or not isinstance(holdouts, int):
        msg = f"holdouts {holdouts!r} must be an int"
        raise TypeError(msg)
    if holdouts < 2:
        msg = f"an interval needs 2 or more holdouts; got {holdouts}"
        raise ValueError(msg)
    if not 0 < confidence < 1 or not 0 < power < 1:
        msg = f"confidence {confidence} and power {power} must lie strictly between 0 and 1"
        raise ValueError(msg)
    for name, value in (("standard deviation", sd), ("ratio", ratio)):
        if not value >= 0:
            msg = f"{name} {value} must be a number at or above 0"
            raise ValueError(msg)
    freedom = holdouts - 1
    quantiles = float(student.ppf(1 - (1 - confidence) / 2, freedom)) + float(
        student.ppf(power, freedom)
    )
    return quantiles * math.sqrt((1 / holdouts + ratio) * sd**2)
