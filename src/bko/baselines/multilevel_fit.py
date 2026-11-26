"""The per-column fit of `hb-marginals`: a hierarchical multinomial logit and its scales.

The model is a baseline-category multinomial logit over every training cell. A non-reference
category's log-odds in cell `(a, s, y)` is

    eta(a, s, y) = mu + A[a] + S[s] + Y[y] + AS[a, s] + AY[a, y] + SY[s, y] + E[a, s, y]

with independent normal priors at mean 0: the intercept and main effects at scale `MAIN_SCALE`,
the two-way terms at `tau2` and the cell effect at `tau3`, a scale of 0 dropping its terms; a cell
with no training record has no cell effect, so its prediction reads its margins alone. For each
pair of scales on `TAU_GRID` the fit finds the posterior mode by `OPTIMISER` from zeros with the
analytic gradient, then by Newton steps on the analytic Hessian until the gradient is below
`GTOL`, and the Laplace log evidence from that Hessian's Cholesky factor; the pair with the
largest evidence, rounded to `EVIDENCE_DECIMALS` decimals, wins, ties going to the smaller
`tau2`, then the smaller `tau3`. A fit with more than `MAX_PARAMETERS` free parameters, one that
stops short of convergence, or a Hessian that is not positive definite raises `BaselineError`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp

from bko.baselines.errors import BaselineError
from bko.baselines.request import Stratum


TAU_GRID: Final = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
"""The values of `tau2`, the two-way terms' prior scale, and of `tau3`, the cell effect's; at 0
the model drops those terms. ADR `research/matched-knowledge-controls`, Decision 7, says why the
grid runs below 1/8 and to 0."""

MAIN_SCALE: Final = 10.0
"""The prior scale of the intercept and of each main effect."""

OPTIMISER: Final = "L-BFGS-B"
"""The `scipy.optimize.minimize` method that finds the posterior mode."""

GTOL: Final = 1e-8
"""The optimiser's projected-gradient tolerance."""

MAXITER: Final = 1000
"""The optimiser's iteration limit; a fit that reaches it is refused."""

NEWTON_STEPS: Final = 20
"""The Newton steps after the optimiser at most; a mode whose gradient is not below `GTOL` by
then is refused."""

HALVINGS: Final = 30
"""The halvings of one Newton step at most before the step is refused."""

MAX_PARAMETERS: Final = 4096
"""The most free parameters a fit may hold: its dense Hessian holds their square in float64,
128 MiB at this ceiling. Diabetes 130's largest column holds 585."""

EVIDENCE_DECIMALS: Final = 6
"""The decimals the Laplace log evidence is rounded to before scale pairs are compared."""

type Key = tuple[Hashable, ...]
"""A model term's name: `("mu",)`, `("A", band)`, ..., `("E", band, sex, label)`."""

MAINS: Final = frozenset({"mu", "A", "S", "Y"})
"""The terms at `MAIN_SCALE`."""

PAIRS: Final = frozenset({"AS", "AY", "SY"})
"""The two-way terms, at `tau2`; the cell effect `E` is at `tau3`."""


@dataclass(frozen=True, slots=True, eq=False)
class Fit:
    """One column's fitted model at its chosen scales.

    Attributes:
        tau2: The two-way terms' prior scale.
        tau3: The cell effect's prior scale.
        evidence: The Laplace log evidence at those scales, rounded to `EVIDENCE_DECIMALS`.
        iterations: The optimiser's iterations at those scales, before the Newton steps.
        categories: The column's categories, in the order `predict` gives their probabilities.
        reference: The category whose log-odds are fixed at 0.
        index: Each term's row in `coefficients`.
        coefficients: The posterior mode, one row per term and one column per category other
            than the reference, in `categories`' order.
        gradient: The largest absolute entry of the negative log posterior's gradient at the
            mode, below `GTOL`.
    """

    tau2: float
    tau3: float
    evidence: float
    iterations: int
    categories: tuple[Hashable, ...]
    reference: Hashable
    index: Mapping[Key, int]
    coefficients: np.ndarray
    gradient: float

    def __post_init__(self) -> None:
        """Make `coefficients` read-only, so no caller changes a fit it shares."""
        self.coefficients.flags.writeable = False

    def predict(self, cell: Stratum) -> tuple[float, ...]:
        """Each category's probability in `cell`; a term the fit never met contributes 0."""
        rows = [self.index[key] for key in _keys(cell, observed=True) if key in self.index]
        others = self.coefficients[rows].sum(axis=0)
        eta = np.insert(others, self.categories.index(self.reference), 0.0)
        return tuple(float(p) for p in np.exp(eta - logsumexp(eta)))


def fit_column[K: Hashable](
    counts: Mapping[Stratum, Mapping[K, int]], *, reference: K | None = None
) -> Fit:
    """Fit one column's model on each cell's category counts, its scales chosen on `TAU_GRID`.

    Args:
        counts: As `fit_pair` takes them.
        reference: As `fit_pair` takes it.

    Returns:
        The `fit_pair` fit with the largest rounded evidence, `max` keeping the first of tied
        pairs in grid order: the smaller `tau2`, then the smaller `tau3`. A one-category column
        has no free parameter and evidence 0 at every pair, so it takes the grid's first pair.

    Raises:
        BaselineError: As `fit_pair` raises it.
        ValueError: As `fit_pair` raises it.
    """
    pairs = [(tau2, tau3) for tau2 in TAU_GRID for tau3 in TAU_GRID]
    if len({category for cell in counts.values() for category, n in cell.items() if n > 0}) == 1:
        pairs = pairs[:1]
    fits = [fit_pair(counts, tau2, tau3, reference=reference) for tau2, tau3 in pairs]
    return max(fits, key=lambda fit: fit.evidence)


def fit_pair[K: Hashable](
    counts: Mapping[Stratum, Mapping[K, int]],
    tau2: float,
    tau3: float,
    *,
    reference: K | None = None,
) -> Fit:
    """Fit one column's model at the scales `tau2` and `tau3`; a scale of 0 drops its terms.

    Args:
        counts: Per cell `(band, sex, label)`, its training records' count per category; a cell
            with no record has no cell effect.
        tau2: The two-way terms' prior scale; at 0 the model has no two-way term.
        tau3: The cell effect's prior scale; at 0 the model has no cell effect.
        reference: The reference category; by default the modal one over every cell, ties to the
            first in category order.

    Returns:
        The posterior mode with its rounded Laplace log evidence.

    Raises:
        BaselineError: If no cell holds a record, the fit holds more than `MAX_PARAMETERS` free
            parameters, or as `_mode` raises it.
        ValueError: If `reference` is not a counted category.
    """
    totals: dict[K, int] = {}
    for cell in counts.values():
        for category, count in cell.items():
            if count > 0:
                totals[category] = totals.get(category, 0) + count
    if not totals:
        msg = "no cell holds a record to fit"
        raise BaselineError(msg)
    categories = tuple(sorted(totals, key=category_order))
    if reference is None:
        reference = max(categories, key=lambda category: totals[category])
    elif reference not in totals:
        msg = f"the reference {reference!r} is not a counted category"
        raise ValueError(msg)
    observed = [cell for cell, held in counts.items() if sum(held.values()) > 0]
    seen = frozenset(observed)
    index: dict[Key, int] = {}
    for cell in counts:
        for key in _keys(cell, observed=cell in seen):
            if _scale(key, tau2, tau3) > 0:
                index.setdefault(key, len(index))
    _check_size(len(index) * (len(categories) - 1))
    design = np.zeros((len(observed), len(index)))
    for row, cell in enumerate(observed):
        held = [index[key] for key in _keys(cell, observed=True) if key in index]
        design[row, held] = 1.0
    table = np.array(
        [[counts[cell].get(category, 0) for category in categories] for cell in observed],
        dtype="float64",
    )
    scales = np.array([_scale(key, tau2, tau3) for key in index])
    coefficients, evidence, iterations, gradient = _mode(
        design, table, categories.index(reference), scales
    )
    return Fit(
        tau2=tau2,
        tau3=tau3,
        evidence=evidence,
        iterations=iterations,
        categories=categories,
        reference=reference,
        index=index,
        coefficients=coefficients,
        gradient=gradient,
    )


def _check_size(parameters: int) -> None:
    """Refuse a fit of more than `MAX_PARAMETERS` free parameters, before its Hessian is built.

    Raises:
        BaselineError: If `parameters` exceeds `MAX_PARAMETERS`.
    """
    if parameters > MAX_PARAMETERS:
        msg = f"the multilevel fit has {parameters} parameters, above MAX_PARAMETERS "
        raise BaselineError(msg + str(MAX_PARAMETERS))


def _keys(cell: Stratum, *, observed: bool) -> list[Key]:
    """The terms whose sum is `cell`'s log-odds; the cell effect only for an observed cell."""
    band, sex, label = cell
    keys: list[Key] = [
        ("mu",),
        ("A", band),
        ("S", sex),
        ("Y", label),
        ("AS", band, sex),
        ("AY", band, label),
        ("SY", sex, label),
    ]
    if observed:
        keys.append(("E", band, sex, label))
    return keys


def _scale(key: Key, tau2: float, tau3: float) -> float:
    """The prior scale of the term `key` names."""
    if key[0] in MAINS:
        return MAIN_SCALE
    return tau2 if key[0] in PAIRS else tau3


def category_order(category: Hashable) -> tuple[int, float, str]:
    """A total order over categories: numbers ascending, then names such as `OTHER`."""
    if isinstance(category, str):
        return (1, 0.0, category)
    return (0, float(category), "")  # ty: ignore[invalid-argument-type]


def _mode(
    design: np.ndarray, table: np.ndarray, reference: int, scales: np.ndarray
) -> tuple[np.ndarray, float, int, float]:
    """The posterior mode at `scales`, its rounded Laplace log evidence, iterations and gradient.

    `OPTIMISER` runs from zeros; Newton steps on the analytic Hessian, each halved until it lowers
    the objective or the largest gradient entry (`_newton`), then take the mode until its gradient
    is below `GTOL`. The optimiser's own stop is refused only at `MAXITER` iterations or a
    non-finite point; any other stop leaves the Newton stage and the `GTOL` check to decide. The
    objective is strictly convex, so the mode is unique and both stages reach the same point from
    any start.

    Args:
        design: One row per observed cell, a 1 at each term the cell's log-odds sums.
        table: One row per observed cell, its count per category.
        reference: The reference category's column of `table`.
        scales: Each term's prior scale.

    Returns:
        The mode, one row per term; its evidence; the optimiser's iterations; and the largest
        absolute gradient entry at the mode.

    Raises:
        BaselineError: If the optimiser reaches `MAXITER` or a non-finite point, the Newton steps
            do not bring the gradient below `GTOL`, or a Hessian is not positive definite.
    """
    others = [k for k in range(table.shape[1]) if k != reference]
    shape = (design.shape[1], len(others))
    if not others:
        return np.zeros(shape), 0.0, 0, 0.0
    precision = np.repeat(scales**-2.0, len(others))
    records = table.sum(axis=1)

    def log_probabilities(theta: np.ndarray) -> np.ndarray:
        eta = np.zeros_like(table)
        eta[:, others] = design @ theta.reshape(shape)
        return eta - logsumexp(eta, axis=1, keepdims=True)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logp = log_probabilities(theta)
        residual = records[:, None] * np.exp(logp[:, others]) - table[:, others]
        gradient = (design.T @ residual).ravel() + precision * theta
        value = -float((table * logp).sum()) + 0.5 * float((precision * theta**2).sum())
        return value, gradient

    def factor(theta: np.ndarray) -> np.ndarray:
        shares = np.exp(log_probabilities(theta)[:, others])
        weights = records[:, None, None] * (
            np.einsum("ck,kl->ckl", shares, np.eye(len(others)))
            - np.einsum("ck,cl->ckl", shares, shares)
        )
        hessian = np.einsum("ci,cj,ckl->ikjl", design, design, weights, optimize=True)
        hessian = hessian.reshape(theta.size, -1)
        hessian[np.diag_indices_from(hessian)] += precision
        try:
            return np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError as error:
            msg = "the multilevel fit's Hessian is not positive definite"
            raise BaselineError(msg) from error

    result = minimize(
        objective,
        np.zeros(precision.size),
        jac=True,
        method=OPTIMISER,
        options={"gtol": GTOL, "maxiter": MAXITER},
    )
    if result.nit >= MAXITER or not np.isfinite(result.x).all():
        msg = f"the multilevel fit did not converge in {result.nit} iterations: {result.message}"
        raise BaselineError(msg)
    theta, (value, gradient) = result.x, objective(result.x)
    for _ in range(NEWTON_STEPS):
        if np.abs(gradient).max() < GTOL:
            break
        theta, value, gradient = _newton(objective, factor(theta), theta, value, gradient)
    largest = float(np.abs(gradient).max())
    if largest >= GTOL:
        msg = f"the multilevel fit's gradient is {largest:.3g} after {NEWTON_STEPS} Newton steps"
        raise BaselineError(msg)
    evidence = (
        -value + 0.5 * float(np.log(precision).sum()) - float(np.log(np.diag(factor(theta))).sum())
    )
    return theta.reshape(shape), round(evidence, EVIDENCE_DECIMALS), int(result.nit), largest


def _newton(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    factor: np.ndarray,
    theta: np.ndarray,
    value: float,
    gradient: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """One Newton step from `theta` on the Hessian's Cholesky `factor`, halved until it helps.

    A step is taken once it lowers the objective or the largest gradient entry; rounding can hold
    the objective level near the mode while the gradient still falls.

    Raises:
        BaselineError: If `HALVINGS` halvings leave the step unhelpful.
    """
    step = cho_solve((factor, True), gradient)
    for halving in range(HALVINGS + 1):
        candidate = theta - step / 2**halving
        moved, slope = objective(candidate)
        if moved <= value or np.abs(slope).max() < np.abs(gradient).max():
            return candidate, moved, slope
    msg = f"a Newton step of the multilevel fit helps at no length to 1/2**{HALVINGS}"
    raise BaselineError(msg)
