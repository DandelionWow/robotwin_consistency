from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from bwm_conversion import (
    build_bwm_trajectory,
    realized_eef16_from_fields,
    realized_eef16_from_mapping,
    state_pose14_from_realized_eef16,
)
from collect_matched_pair import SamplingSceneProxy
from protocol import (
    aggregate_paired_metric,
    aggregate_records,
    artifact_directory,
    balanced_jepa_sets,
    build_windows,
    cadence_record,
    checkpoint_identity,
    decode_robotwin_rgb_jpeg,
    generation_seed,
    legacy_episode_formal_eligible,
    load_and_validate_reference_stat,
    metric_value_with_normalization,
    pair_id,
    pair_length_eligibility,
    require_matching_cadence,
    require_matching_initial_state,
    require_uniform_timestamp_grid,
    select_protocol_windows,
    state_fingerprint,
    summarize_tracker_results,
    validate_formal_seed_plan,
    validate_reproducible_code_provenance,
    verified_failure_label,
    worldarena_sample_id,
    worldarena_summary_row,
)


class ProtocolTest(unittest.TestCase):
    def test_robotwin_jpeg_decode_preserves_rgb_channel_positions(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is unavailable")

        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        rgb[..., 0] = 240
        rgb[..., 1] = 80
        rgb[..., 2] = 10
        success, encoded = cv2.imencode(".jpg", rgb)
        self.assertTrue(success)

        decoded = decode_robotwin_rgb_jpeg(encoded.tobytes(), source="test")
        channel_means = decoded.mean(axis=(0, 1))
        self.assertGreater(channel_means[0], channel_means[1])
        self.assertGreater(channel_means[1], channel_means[2])

    def test_sampling_proxy_never_appends_off_grid_terminal_frame(self):
        class Scene:
            def step(self):
                return None

        class Task:
            def __init__(self):
                self.pictures = 0

            def _take_picture(self):
                self.pictures += 1

        task = Task()
        proxy = SamplingSceneProxy(Scene(), task, physics_timestep=0.004, steps_per_sample=25)
        for _ in range(27):
            proxy.step()
        np.testing.assert_allclose(proxy.timestamps, [0.0, 0.1])
        self.assertEqual(task.pictures, 1)
        self.assertEqual(proxy.physics_steps, 27)

    def test_orbax_checkpoint_identity_covers_params_and_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "params/d").mkdir(parents=True)
            (root / "assets/task").mkdir(parents=True)
            (root / "params/d/part").write_bytes(b"weights")
            (root / "assets/task/norm_stats.json").write_text("{}", encoding="utf-8")
            first = checkpoint_identity(root)
            self.assertEqual(first["kind"], "orbax_jax")
            self.assertEqual(first["file_count"], 2)
            (root / "assets/task/norm_stats.json").write_text('{"changed":true}', encoding="utf-8")
            second = checkpoint_identity(root)
            self.assertNotEqual(first["sha256"], second["sha256"])

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

    def test_unconfirmed_old_seed_mapping_is_not_formal(self):
        record = {
            "seed_mapping_status": "AMBIGUOUS",
            "failure_label_status": "CONFIRMED",
            "policy_checkpoint_status": "CONFIRMED",
            "task_config_status": "CONFIRMED",
        }
        self.assertFalse(legacy_episode_formal_eligible(record))

    def test_filename_failure_is_not_verified_failure(self):
        self.assertEqual(
            verified_failure_label("filename", False),
            "FILENAME_LABELLED_FAILURE",
        )
        self.assertNotEqual(verified_failure_label("filename", False), "VERIFIED_FAILURE")

    def test_dirty_robotwin_patch_hash_is_required(self):
        with self.assertRaisesRegex(ValueError, "patch"):
            validate_reproducible_code_provenance(
                {"robotwin_commit": "abc", "robotwin_dirty": True}
            )
        validate_reproducible_code_provenance(
            {
                "robotwin_commit": "abc",
                "robotwin_dirty": True,
                "robotwin_patch_sha256": "def",
            }
        )

    def test_initial_state_fingerprint_is_deterministic(self):
        first = {
            "actors": [
                {"name": "z", "pose": [1.0000004]},
                {"name": "a", "pose": [2.0000002]},
            ]
        }
        second = {"actors": [{"pose": [2.0], "name": "a"}, {"pose": [1.0], "name": "z"}]}
        self.assertEqual(state_fingerprint(first), state_fingerprint(second))

    def test_initial_state_mismatch_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "STATE_MISMATCH"):
            require_matching_initial_state({"robot": [1.0]}, {"robot": [1.1]})

    def test_two_source_adapters_share_identical_16d_to_14d_conversion(self):
        left = np.asarray([[1, 2, 3, 1, 0, 0, 0]], dtype=np.float32)
        right = np.asarray([[4, 5, 6, 1, 0, 0, 0]], dtype=np.float32)
        left_grip = np.asarray([0.25], dtype=np.float32)
        right_grip = np.asarray([0.75], dtype=np.float32)
        hdf5_style = realized_eef16_from_fields(left, left_grip, right, right_grip)
        collector_style = realized_eef16_from_mapping(
            {
                "left_endpose": left,
                "left_gripper": left_grip,
                "right_endpose": right,
                "right_gripper": right_grip,
            }
        )
        np.testing.assert_array_equal(
            state_pose14_from_realized_eef16(hdf5_style),
            state_pose14_from_realized_eef16(collector_style),
        )

    def test_cadence_mismatch_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "CADENCE_MISMATCH"):
            require_matching_cadence(cadence_record(30, 1), cadence_record(10, 1))

    def test_uniform_timestamp_grid_accepts_exact_samples(self):
        result = require_uniform_timestamp_grid([0.0, 0.1, 0.2], 0.1)
        self.assertEqual(result["status"], "EXACT_GRID")
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["minimum_dt"], 0.1)

    def test_uniform_timestamp_grid_rejects_off_grid_terminal_sample(self):
        with self.assertRaisesRegex(
            ValueError, r"interval 1->2 has dt=0\.07.*expected 0\.1"
        ):
            require_uniform_timestamp_grid([0.0, 0.1, 0.17], 0.1)

    def test_uniform_timestamp_grid_rejects_nonfinite_and_nonmonotonic(self):
        with self.assertRaisesRegex(ValueError, "non-finite timestamp"):
            require_uniform_timestamp_grid([0.0, np.nan], 0.1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            require_uniform_timestamp_grid([0.0, 0.1, 0.1], 0.1)

    def test_bwm_trajectory_rejects_declared_fps_with_off_grid_timestamp(self):
        realized = np.zeros((3, 16), dtype=np.float32)
        realized[:, 3] = 1.0
        realized[:, 11] = 1.0
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_GRID_INVALID"):
            build_bwm_trajectory(
                head_rgb=np.zeros((3, 2, 2, 3), dtype=np.uint8),
                realized_eef_wxyz16=realized,
                frame_timestamps=[0.0, 0.1, 0.17],
                task="task",
                pair_id="task__seed1",
                side="expert",
                success=True,
                env_seed=1,
                raw_fps=10.0,
            )

    def test_short_side_excludes_whole_pair(self):
        result = pair_length_eligibility(80, 100)
        self.assertFalse(result["eligible"])
        self.assertTrue(result["pairwise_excluded"])
        self.assertTrue(result["expert_short"])
        self.assertFalse(result["policy_short"])

    def test_early_middle_late_formula_and_deduplication(self):
        windows = select_protocol_windows(101)
        self.assertEqual([(item.slot, item.window_start) for item in windows], [
            ("early", 0),
            ("middle", 10),
            ("late", 20),
        ])
        self.assertEqual([(item.slot, item.window_start) for item in select_protocol_windows(81)], [
            ("early", 0)
        ])

    def test_env_seed_and_generation_seed_are_separate_fields(self):
        identifier = pair_id("task", 123)
        manifest = {"env_seed": 123, "generation_seed": generation_seed(identifier, "middle", 0)}
        self.assertEqual(manifest["env_seed"], 123)
        self.assertNotEqual(manifest["generation_seed"], manifest["env_seed"])

    def test_formal_seed_plan_is_exactly_three_by_two(self):
        plan = {
            "schema_version": 1,
            "status": "FORMAL",
            "tasks": [
                {"task": "a", "seeds": [1, 2]},
                {"task": "b", "seeds": [3, 4]},
                {"task": "c", "seeds": [5, 6]},
            ],
        }
        self.assertEqual(validate_formal_seed_plan(plan)["b"], (3, 4))

    def test_seed_plan_rejects_duplicate_seed_within_task(self):
        plan = {
            "schema_version": 1,
            "status": "FORMAL",
            "tasks": [
                {"task": "a", "seeds": [1, 1]},
                {"task": "b", "seeds": [3, 4]},
                {"task": "c", "seeds": [5, 6]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            validate_formal_seed_plan(plan)

    def test_generation_seed_is_stable_sha256_value(self):
        value = generation_seed("task__seed123", "middle", 0)
        self.assertEqual(value, generation_seed("task__seed123", "middle", 0))
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 2**31)

    def test_matched_sides_receive_same_generation_seed(self):
        identifier = pair_id("task", 3)
        expert = generation_seed(identifier, "late", 2)
        policy = generation_seed(identifier, "late", 2)
        self.assertEqual(expert, policy)

    def test_repeat_window_pair_aggregation_hierarchy(self):
        records = []
        for side, values in (("expert", [10.0, 14.0]), ("pi05", [6.0, 10.0])):
            for repeat_id, value in enumerate(values):
                records.append(
                    {
                        "pair_id": "task__seed1",
                        "side": side,
                        "window_slot": "middle",
                        "repeat_id": repeat_id,
                        "metrics": {"psnr": value},
                    }
                )
        summary = aggregate_paired_metric(records, "psnr", bootstrap_samples=10)
        self.assertEqual(summary["n_pairs"], 1)
        self.assertEqual(summary["per_pair_gap"][0]["expert_score"], 12.0)
        self.assertEqual(summary["per_pair_gap"][0]["policy_score"], 8.0)
        self.assertEqual(summary["mean_gap"], 4.0)

    def test_unmatched_sides_cannot_compute_pair_gap(self):
        record = {
            "pair_id": "task__seed1",
            "side": "expert",
            "window_slot": "middle",
            "repeat_id": 0,
            "metrics": {"psnr": 10.0},
        }
        with self.assertRaisesRegex(ValueError, "Unmatched sides"):
            aggregate_paired_metric([record], "psnr")

    def test_worldarena_sample_id_is_globally_distinct_by_side_and_repeat(self):
        first = worldarena_sample_id("task__seed1", "expert", "middle", 0)
        second = worldarena_sample_id("task__seed1", "pi05", "middle", 0)
        third = worldarena_sample_id("task__seed1", "expert", "middle", 1)
        self.assertEqual(len({first, second, third}), 3)

    def test_worldarena_summary_schema_is_exact(self):
        sample = worldarena_sample_id("task__seed1", "expert", "middle", 0)
        row = worldarena_summary_row(sample, Path("gt.mp4"), Path("pred.mp4"))
        self.assertEqual(set(row), {"sample_id", "gt_path", "generated_video"})
        self.assertTrue(Path(row["gt_path"]).is_absolute())

    def test_jepa_cannot_enter_pair_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "set-level"):
            aggregate_paired_metric([], "jepa")

    def test_jepa_sets_must_be_balanced(self):
        sample = worldarena_sample_id("task__seed1", "expert", "middle", 0)
        with self.assertRaisesRegex(ValueError, "not balanced"):
            balanced_jepa_sets(
                [{
                    "pair_id": "task__seed1",
                    "side": "expert",
                    "window_slot": "middle",
                    "repeat_id": 0,
                    "sample_id": sample,
                }]
            )

    def test_raw_and_normalized_metric_are_separate(self):
        result = metric_value_with_normalization(12.0, 0.0, 10.0)
        self.assertEqual(result["raw_value"], 12.0)
        self.assertEqual(result["normalized_value"], 1.0)
        self.assertTrue(result["clipped_flag"])
        self.assertEqual(result["normalization_bounds"], [0.0, 10.0])

    def test_tracker_failures_are_counted_not_filtered(self):
        summary = summarize_tracker_results(
            [
                {"side": "expert", "tracker_success": True},
                {"side": "pi05", "tracker_success": False, "failure_reason": "missing_detection"},
            ]
        )
        self.assertEqual(summary["n_records"], 2)
        self.assertEqual(summary["trajectory_tracker_success_rate"], 0.5)
        self.assertEqual(summary["failure_reason_counts"], {"missing_detection": 1})

    def test_artifact_path_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = artifact_directory(
                Path(tmp), "task__seed1", "expert", "middle", 0
            )
            path.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                artifact_directory(
                    Path(tmp),
                    "task__seed1",
                    "expert",
                    "middle",
                    0,
                    require_absent=True,
                )


if __name__ == "__main__":
    unittest.main()
