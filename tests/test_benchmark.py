import unittest

from benchmarks.compare_models import compare


class BenchmarkTests(unittest.TestCase):
    def test_comparison_reports_all_models_and_metrics(self) -> None:
        result = compare((b"abc state memory branch " * 16), epochs=1, max_bytes=256, seed=1)
        self.assertEqual(len(result["results"]), 3)
        for model in result["results"]:
            self.assertGreater(model["parameters"], 0)
            self.assertGreater(model["initial_validation_bits_per_byte"], 0)
            self.assertGreater(model["final_validation_bits_per_byte"], 0)
            self.assertGreater(model["training_bytes_per_second"], 0)


if __name__ == "__main__":
    unittest.main()

