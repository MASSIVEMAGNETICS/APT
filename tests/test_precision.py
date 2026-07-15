from decimal import Decimal
from fractions import Fraction
import unittest

from apt.precision import DeterministicPrecisionKernel, NumericalContractError


class PrecisionKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = DeterministicPrecisionKernel()

    def test_sqrt_is_high_precision_and_deterministic(self) -> None:
        first = self.kernel.sqrt(2)
        second = self.kernel.sqrt("2")
        self.assertEqual(first, second)
        self.assertLess(abs(first * first - Decimal(2)), Decimal("1e-29"))

    def test_bracketed_root(self) -> None:
        root = self.kernel.root(lambda x: x * x, 1, 2, target=2)
        self.assertLess(abs(root * root - Decimal(2)), Decimal("1e-29"))

    def test_root_rejects_unbracketed_interval(self) -> None:
        with self.assertRaises(NumericalContractError):
            self.kernel.root(lambda x: x * x + 1, -1, 1)

    def test_egyptian_fraction_reconstructs_exact_value(self) -> None:
        terms = self.kernel.egyptian_fraction(8, 13)
        self.assertEqual(self.kernel.reconstruct_unit_fractions(terms), Fraction(8, 13))

    def test_weighted_mean_contracts(self) -> None:
        self.assertEqual(self.kernel.weighted_mean([1, 3], [1, 1]), Decimal(2))
        with self.assertRaises(NumericalContractError):
            self.kernel.weighted_mean([1], [0])


if __name__ == "__main__":
    unittest.main()

