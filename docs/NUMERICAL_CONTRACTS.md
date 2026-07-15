# Numerical contracts

`DeterministicPrecisionKernel` uses local `decimal` contexts so a caller does not
silently alter global precision or rounding.

## Conversion

- Integers, strings, finite `Decimal` values and finite floats are accepted.
- Floats convert through their string representation.
- Booleans, NaN and infinity are rejected.

## Square root

- Domain: `value >= 0`.
- Zero returns exactly zero without division.
- Newton/Babylonian iteration stops when successive estimates differ by at most
  the contract tolerance.
- Exhausting `max_iterations` raises `ConvergenceError`.

## Root

- Domain: `lower < upper`.
- The target-adjusted endpoint values must have opposite signs unless an endpoint
  is already the root.
- Deterministic bisection stops on residual or half-width tolerance.
- An interval that does not bracket a root raises `NumericalContractError`.
- This is a root finder, not an optimizer.

## Egyptian fractions

- Domain: positive integers with `0 < numerator < denominator`.
- Arithmetic uses `fractions.Fraction`; reconstruction is exact.
- A configurable term limit prevents unbounded resource consumption.

## Weighted mean

- Values and weights have equal non-zero length.
- Weights are non-negative and at least one must be positive.
- Evaluation uses the configured decimal precision and rounding mode.

