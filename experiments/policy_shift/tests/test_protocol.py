from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from protocol import aggregate_records, build_windows, load_and_validate_reference_stat


class ProtocolTest(unittest.TestCase):
    def test_windows_are_contiguous_and_independent(self):
        windows = build_windows(17, history_frames=5, future_frames=4, stride=4)
        self.assertEqual([window.window_start for window in windows], [0, 4, 8])
        self.assertEqual(windows[1].history_indices, (4, 5, 6, 7, 8))
        self.assertEqual(windows[1].future_indices, (9, 10, 11, 12))

    def test_reference_stat_must_be_14d(self):
        payload = {"state_pose": {"p01": [0.0] * 14, "p99": [1.0] * 14}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stat.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, provenance = load_and_validate_reference_stat(path)
        self.assertEqual(provenance["dimensions"], 14)
        self.assertEqual(provenance["lower_key"], "p01")

    def test_gap_error_has_one_direction(self):
        records = [
            {
                "task": "t",
                "episode_id": "e1",
                "distribution": "expert",
                "policy_id": "expert",
                "success": True,
                "metrics": {"psnr": 30.0, "ssim": 0.9, "jepa": None, "trajectory_accuracy": None},
            },
            {
                "task": "t",
                "episode_id": "e2",
                "distribution": "policy",
                "policy_id": "p",
                "success": False,
                "metrics": {"psnr": 20.0, "ssim": 0.7, "jepa": None, "trajectory_accuracy": None},
            },
        ]
        summary = aggregate_records(records, bootstrap_samples=10, seed=7)
        self.assertEqual(summary["expert_vs_policy"]["psnr"]["raw_gap_policy_minus_expert"], -10.0)
        self.assertEqual(summary["expert_vs_policy"]["psnr"]["gap_error"], 10.0)
        self.assertAlmostEqual(summary["expert_vs_policy"]["ssim"]["gap_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
