from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apt.simulator import Hypothesis, SimulatorBank
from apt.system import CognitiveOrganism


class SimulatorAndSystemTests(unittest.TestCase):
    def test_simulator_ranks_evidence_and_risk(self) -> None:
        bank = SimulatorBank()
        ranked = bank.evaluate(
            "Choose a testable memory design",
            [
                Hypothesis("Use append-only SQLite records", 0.9, identifier="measured", risk=0.1),
                Hypothesis("Assume memory works magically", 0.1, identifier="magic", risk=0.9),
            ],
            context=["SQLite transactions preserve committed records"],
        )
        self.assertEqual(ranked[0].hypothesis.identifier, "measured")
        self.assertAlmostEqual(sum(item.normalized_probability for item in ranked), 1.0)

    def test_integrated_observe_branch_rewind_recall(self) -> None:
        with TemporaryDirectory() as directory, CognitiveOrganism(Path(directory) / "state") as system:
            node, record, report = system.observe("Timeline nodes preserve observable state.")
            self.assertEqual(record.timeline_node_id, node.id)
            self.assertGreaterEqual(report.novelty, 0.0)
            system.branch("experiment")
            system.observe("Experiment branch state.", branch="experiment", semantic=True)
            self.assertEqual(len(system.replay("main")), 2)
            self.assertEqual(len(system.replay("experiment")), 3)
            self.assertTrue(system.recall("observable timeline"))
            status = system.status()
            self.assertEqual(status["memory_count"], 2)
            self.assertIn("experiment", status["branches"])


if __name__ == "__main__":
    unittest.main()

