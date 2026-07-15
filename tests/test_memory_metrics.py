from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apt.memory import ByteNGramEncoder, ContentAddressedMemory
from apt.metrics import CognitiveMetrics


class MemoryAndMetricsTests(unittest.TestCase):
    def test_content_addressing_and_similarity_search(self) -> None:
        with TemporaryDirectory() as directory, ContentAddressedMemory(
            Path(directory) / "memory.sqlite3"
        ) as memory:
            first = memory.remember_episode("magnetic field geometry", metadata={"source": "lab"})
            duplicate = memory.remember_episode("magnetic field geometry", metadata={"source": "lab"})
            memory.remember_semantic("music distribution revenue")
            self.assertEqual(first.hash, duplicate.hash)
            self.assertEqual(memory.count(), 2)
            self.assertEqual(memory.occurrence_count(), 3)
            results = memory.search("magnetic field geometry", limit=2)
            self.assertEqual(results[0].record.hash, first.hash)
            self.assertAlmostEqual(results[0].similarity, 1.0, places=12)

    def test_novelty_and_coherence_are_content_derived(self) -> None:
        metrics = CognitiveMetrics(ByteNGramEncoder(dimensions=128))
        text = "Persistent memory preserves prior observations."
        self.assertAlmostEqual(metrics.novelty(text, [text]), 0.0, places=12)
        repeated = metrics.measure(text, [text])
        unrelated = metrics.measure("zxqv 9173 isolated bytes", [text])
        self.assertGreater(repeated.coherence, unrelated.coherence)
        self.assertGreater(unrelated.novelty, repeated.novelty)
        self.assertEqual(metrics.repetition("abc"), 0.0)
        self.assertGreater(metrics.repetition("abcabcabcabc"), 0.0)


if __name__ == "__main__":
    unittest.main()
