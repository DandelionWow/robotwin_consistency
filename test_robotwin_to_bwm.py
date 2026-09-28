import unittest
from pathlib import Path

import numpy as np

import robotwin_to_bwm as converter


class BwmActionConversionTest(unittest.TestCase):
    def test_state_to_action_uses_delta_motion_and_absolute_grippers(self):
        Rotation = converter.import_rotation()
        state = np.zeros((3, converter.STATE_WIDTH), dtype=np.float32)
        state[:, 0] = [0.0, 0.2, 0.5]
        state[:, 7] = [0.0, 0.1, 0.4]
        state[:, 6] = [1.0, 0.4, 0.0]
        state[:, 19] = [0.0, 0.6, 1.0]

        current = Rotation.from_euler("xyz", [0.2, -0.1, 0.3])
        delta = Rotation.from_euler("xyz", [0.04, -0.03, 0.02])
        state[0, 10:13] = current.as_euler("xyz")
        state[1, 10:13] = (current * delta).as_euler("xyz")
        state[2, 10:13] = state[1, 10:13]

        action = converter.state_to_action(state)

        self.assertAlmostEqual(float(action[0, 0]), 0.2)
        self.assertAlmostEqual(float(action[1, 7]), 0.3)
        np.testing.assert_allclose(action[0, 10:13], [0.04, -0.03, 0.02], atol=1e-6)
        np.testing.assert_allclose(action[:, 6], [0.4, 0.0, 0.0])
        np.testing.assert_allclose(action[:, 19], [0.6, 1.0, 1.0])
        continuous = [index for index in range(converter.STATE_WIDTH) if index not in (6, 19)]
        np.testing.assert_allclose(action[-1, continuous], 0.0)

    def test_action_conversion_matches_bwm_reference_parquet(self):
        _, pq = converter.import_pyarrow()
        reference = (
            Path(__file__).resolve().parent
            / "third_party"
            / "boundless-world-model"
            / "demo"
            / "adjust_bottle"
            / "data"
            / "chunk-000"
            / "episode_000040.parquet"
        )
        table = pq.read_table(reference).to_pydict()
        state = np.asarray(table["observation.state"], dtype=np.float32)
        expected = np.asarray(table["action"], dtype=np.float32)
        np.testing.assert_allclose(converter.state_to_action(state), expected, atol=1e-6)

    def test_all_statistics_have_bwm_action_width(self):
        state = np.zeros((2, converter.STATE_WIDTH), dtype=np.float32)
        action = converter.state_to_action(state)
        stats = converter.build_stat([state], [action])
        self.assertEqual(set(stats), {"state_joint", "state_pose", "action_joint", "action_pose"})
        for value in stats.values():
            self.assertEqual(len(value["min"]), 14)
            self.assertEqual(len(value["max"]), 14)


if __name__ == "__main__":
    unittest.main()
