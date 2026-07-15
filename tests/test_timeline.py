from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apt.timeline import TimelineDAG, TimelineError


class TimelineTests(unittest.TestCase):
    def test_branch_rewind_replay_and_persistence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.sqlite3"
            with TimelineDAG(path) as timeline:
                genesis = timeline.genesis({"value": 0})
                one = timeline.commit({"value": 1})
                two = timeline.commit({"value": 2})
                timeline.fork("main", "experiment", at_node_id=one.id)
                experiment = timeline.commit({"value": 99}, "experiment")
                self.assertEqual(timeline.head("main").id, two.id)
                self.assertEqual(timeline.head("experiment").id, experiment.id)
                self.assertEqual([node.state["value"] for node in timeline.replay(branch="experiment")], [0, 1, 99])
                rewound = timeline.rewind("main", 1)
                self.assertEqual(rewound.id, one.id)
                alternate = timeline.commit({"value": 3})
                self.assertEqual([node.state["value"] for node in timeline.replay(branch="main")], [0, 1, 3])
                self.assertEqual({child.id for child in timeline.children(one.id)}, {two.id, experiment.id, alternate.id})
            with TimelineDAG(path) as reopened:
                self.assertEqual(reopened.head("main").state["value"], 3)
                self.assertEqual(reopened.head("experiment").state["value"], 99)

    def test_unknown_branch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory, TimelineDAG(Path(directory) / "t.db") as timeline:
            with self.assertRaises(TimelineError):
                timeline.head("missing")


if __name__ == "__main__":
    unittest.main()

