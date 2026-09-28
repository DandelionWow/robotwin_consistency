import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import collect_adjust_bottle_pairs as collector_module


class BoundarySearchTest(unittest.TestCase):
    def test_search_returns_adjacent_success_and_failure(self):
        collector = object.__new__(collector_module.Collector)
        collector.cli = SimpleNamespace(
            max_magnitude=0.12,
            coarse_step=0.01,
            boundary_tolerance=0.001,
        )

        def replay_candidate(**kwargs):
            magnitude = float(kwargs["magnitude"])
            return {
                "direction": kwargs["direction"],
                "magnitude": magnitude,
                "offset": np.asarray(kwargs["offset"]).tolist(),
                "check_success": magnitude < 0.053,
            }

        collector.replay_candidate = replay_candidate
        result = collector.find_boundary({}, 0, "+x", np.asarray([1.0, 0.0]))

        self.assertTrue(result["success"]["check_success"])
        self.assertFalse(result["failure"]["check_success"])
        self.assertLessEqual(result["boundary_gap"], 0.001)
        self.assertLess(result["success"]["magnitude"], result["failure"]["magnitude"])


class ManifestTest(unittest.TestCase):
    def test_record_sequence_must_be_contiguous(self):
        collector_module.validate_record_sequence([{"pair_id": 0}, {"pair_id": 1}])
        with self.assertRaises(ValueError):
            collector_module.validate_record_sequence([{"pair_id": 0}, {"pair_id": 2}])

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            collector_module.atomic_write_json(path, {"offset": np.asarray([0.01, 0.0])})
            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "offset": [\n    0.01,\n    0.0\n  ]\n}')
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
