"""Deterministic numerical primitives with explicit behavioral contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from typing import Callable, Iterable, Sequence


class NumericalContractError(ValueError):
    """Raised when an input violates a documented numerical contract."""


class ConvergenceError(ArithmeticError):
    """Raised when an iterative method cannot satisfy its tolerance contract."""


@dataclass(frozen=True)
class NumericalContract:
    """Controls precision, convergence, and resource limits.

    Decimal conversion always uses ``str(value)`` for floats so results do not
    depend on the platform-specific binary expansion of a float.
    """

    precision: int = 50
    tolerance: Decimal = Decimal("1e-30")
    max_iterations: int = 256
    max_fraction_terms: int = 10_000
    rounding: str = ROUND_HALF_EVEN

    def __post_init__(self) -> None:
        if self.precision < 16:
            raise NumericalContractError("precision must be at least 16 digits")
        if self.tolerance <= 0:
            raise NumericalContractError("tolerance must be positive")
        if self.max_iterations < 1:
            raise NumericalContractError("max_iterations must be positive")
        if self.max_fraction_terms < 1:
            raise NumericalContractError("max_fraction_terms must be positive")


class DeterministicPrecisionKernel:
    """Validated exact and high-precision deterministic operations."""

    def __init__(self, contract: NumericalContract | None = None) -> None:
        self.contract = contract or NumericalContract()

    def decimal(self, value: Decimal | int | float | str) -> Decimal:
        if isinstance(value, bool):
            raise NumericalContractError("boolean values are not numeric inputs")
        try:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise NumericalContractError(f"invalid decimal value: {value!r}") from exc
        if not result.is_finite():
            raise NumericalContractError("NaN and infinity are not supported")
        return result

    def sqrt(self, value: Decimal | int | float | str) -> Decimal:
        """Return the non-negative square root using validated Newton iteration."""

        n = self.decimal(value)
        if n < 0:
            raise NumericalContractError("square root requires value >= 0")
        if n == 0:
            return Decimal(0)
        with localcontext() as ctx:
            ctx.prec = self.contract.precision + 8
            ctx.rounding = self.contract.rounding
            x = n if n >= 1 else Decimal(1)
            for _ in range(self.contract.max_iterations):
                nxt = (x + n / x) / 2
                if abs(nxt - x) <= self.contract.tolerance:
                    ctx.prec = self.contract.precision
                    return +nxt
                x = nxt
        raise ConvergenceError("sqrt did not converge within max_iterations")

    def root(
        self,
        function: Callable[[Decimal], Decimal | int | float | str],
        lower: Decimal | int | float | str,
        upper: Decimal | int | float | str,
        *,
        target: Decimal | int | float | str = 0,
    ) -> Decimal:
        """Find a bracketed root with deterministic bisection.

        Contract: ``lower < upper`` and the target-adjusted endpoint values must
        have opposite signs (or one endpoint must be the root). The returned
        point has residual or bracket width at most ``tolerance``.
        """

        a, b, goal = self.decimal(lower), self.decimal(upper), self.decimal(target)
        if not a < b:
            raise NumericalContractError("root requires lower < upper")

        def evaluate(x: Decimal) -> Decimal:
            return self.decimal(function(x)) - goal

        with localcontext() as ctx:
            ctx.prec = self.contract.precision + 8
            ctx.rounding = self.contract.rounding
            fa, fb = evaluate(a), evaluate(b)
            if fa == 0:
                return +a
            if fb == 0:
                return +b
            if fa * fb > 0:
                raise NumericalContractError(
                    "root interval must bracket the target with opposite signs"
                )
            for _ in range(self.contract.max_iterations):
                midpoint = (a + b) / 2
                fm = evaluate(midpoint)
                if abs(fm) <= self.contract.tolerance or (b - a) / 2 <= self.contract.tolerance:
                    ctx.prec = self.contract.precision
                    return +midpoint
                if fa * fm < 0:
                    b, fb = midpoint, fm
                else:
                    a, fa = midpoint, fm
        raise ConvergenceError("root did not converge within max_iterations")

    def egyptian_fraction(self, numerator: int, denominator: int) -> tuple[int, ...]:
        """Return the exact greedy unit-fraction decomposition of a positive rational."""

        if isinstance(numerator, bool) or isinstance(denominator, bool):
            raise NumericalContractError("boolean values are not valid integers")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise NumericalContractError("numerator and denominator must be integers")
        if numerator <= 0 or denominator <= 0:
            raise NumericalContractError("numerator and denominator must be positive")
        value = Fraction(numerator, denominator)
        if value >= 1:
            raise NumericalContractError("unit-fraction form requires 0 < numerator < denominator")
        terms: list[int] = []
        while value:
            if len(terms) >= self.contract.max_fraction_terms:
                raise ConvergenceError("fraction term limit exceeded")
            unit_denominator = (value.denominator + value.numerator - 1) // value.numerator
            terms.append(unit_denominator)
            value -= Fraction(1, unit_denominator)
        return tuple(terms)

    def weighted_mean(
        self,
        values: Sequence[Decimal | int | float | str],
        weights: Sequence[Decimal | int | float | str],
    ) -> Decimal:
        """Compute a reproducible weighted mean with non-negative weights."""

        if not values or len(values) != len(weights):
            raise NumericalContractError("values and weights must have equal non-zero length")
        vals = [self.decimal(value) for value in values]
        wts = [self.decimal(weight) for weight in weights]
        if any(weight < 0 for weight in wts):
            raise NumericalContractError("weights must be non-negative")
        total_weight = sum(wts, Decimal(0))
        if total_weight == 0:
            raise NumericalContractError("at least one weight must be positive")
        with localcontext() as ctx:
            ctx.prec = self.contract.precision
            ctx.rounding = self.contract.rounding
            return +(sum((v * w for v, w in zip(vals, wts)), Decimal(0)) / total_weight)

    @staticmethod
    def reconstruct_unit_fractions(denominators: Iterable[int]) -> Fraction:
        result = Fraction(0, 1)
        for denominator in denominators:
            if not isinstance(denominator, int) or denominator <= 0:
                raise NumericalContractError("unit-fraction denominators must be positive integers")
            result += Fraction(1, denominator)
        return result

