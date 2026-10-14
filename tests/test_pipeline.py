import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from robot_data_lab.data import audit_arrays, split_episodes, sample_video
from robot_data_lab.baseline import normalize_fit, train_one
from robot_data_lab.io import safe_path, write_json, read_json, fingerprint

RULES = {"min_frames": 2, "max_state_step": 35, "idle_state_step": .05, "max_idle_fraction": .65}


def trajectory():
    return pd.DataFrame({"timestamp": [0., 1/30, 2/30], "frame_index": [0, 1, 2],
        "observation.state": [np.array([x, x]) for x in [0., 1., 2.]],
        "action": [np.array([x, x]) for x in [1., 2., 3.]]})


class QualityTests(unittest.TestCase):
    def test_valid_episode(self):
        result = audit_arrays(trajectory(), 30, RULES, 2, 2)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])

    def test_nonfinite_action_rejected(self):
        df = trajectory()
        df.at[1, "action"] = np.array([np.nan, 0])
        self.assertIn("nonfinite_state_or_action", audit_arrays(df, 30, RULES)["errors"])

    def test_reversed_timestamp_not_hidden_by_sort(self):
        df = trajectory().iloc[[0, 2, 1]]
        result = audit_arrays(df, 30, RULES)
        self.assertIn("nonmonotonic_timestamp", result["errors"])
        self.assertIn("noncontiguous_frame_index", result["errors"])

    def test_jump_is_warning(self):
        df = trajectory()
        df.at[1, "observation.state"] = np.array([100., 0])
        result = audit_arrays(df, 30, RULES)
        self.assertIn("state_jump", result["warnings"])
        self.assertEqual(result["errors"], [])

    def test_wrong_dimensions(self):
        result = audit_arrays(trajectory(), 30, RULES, 6, 6)
        self.assertIn("action_dimension_mismatch", result["errors"])

    def test_duplicate_frames(self):
        df = trajectory()
        df.loc[2, "frame_index"] = 1
        self.assertIn("duplicate_frame_index", audit_arrays(df, 30, RULES)["errors"])


class ReproducibilityTests(unittest.TestCase):
    def test_annotation_provenance_and_coverage(self):
        from robot_data_lab.annotations import check_annotations
        rows = [{"episode_index": 0, "task": "pick cube", "source": "upstream_dataset", "review_status": "unreviewed"}]
        self.assertFalse(check_annotations(rows, {0})["errors"])
        self.assertEqual(check_annotations(rows, {0})["reviewed_rows"], 0)
        self.assertTrue(check_annotations(rows, {0,1})["errors"])
        self.assertTrue(check_annotations(rows+rows, {0})["errors"])

    def test_cloud_plan_pins_dataset_and_disables_upload(self):
        from robot_data_lab.cloud import commands
        from types import SimpleNamespace
        args = SimpleNamespace(seed=42, steps=100, batch_size=1, checkpoint=None, episodes=10)
        config = {"dataset_repo":"owner/test", "dataset_revision":"a"*40, "task_suite":"libero_spatial"}
        train, evaluate = commands(config, Path("project"), args)
        self.assertIn("--dataset.revision="+"a"*40, train)
        self.assertIn("--policy.push_to_hub=false", train)
        self.assertIn("--eval.n_episodes=10", evaluate)

    def test_episode_split_disjoint_and_deterministic(self):
        a, b = split_episodes(range(50), 42, .2)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(50)))
        self.assertEqual((a, b), split_episodes(range(50), 42, .2))

    def test_train_only_normalization(self):
        training = np.array([[0., 2], [2., 2]])
        mean, std = normalize_fit(training)
        self.assertTrue(np.allclose(mean, [1, 2]))
        self.assertTrue(np.all(std > 0))
        self.assertGreater(((np.array([999., 2]) - mean) / std)[0], 900)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_path(directory, "../outside")

    def test_json_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory)/"bad.json", {"x": float("nan")})

    def test_deterministic_training_and_checkpoint(self):
        import torch
        rng = np.random.default_rng(5)
        x = rng.normal(size=(40, 3)).astype(np.float32)
        y = (2*x[:, :2]).astype(np.float32)
        episodes = np.repeat(np.arange(4), 10)
        cfg = {"torch_threads": 1, "hidden_dim": 8, "learning_rate": .01, "steps": 10, "batch_size": 8}
        with tempfile.TemporaryDirectory() as directory:
            a = train_one(x,y,episodes,[0,1,2],[3],cfg,11,Path(directory)/"a",np.ones(2),3)
            b = train_one(x,y,episodes,[0,1,2],[3],cfg,11,Path(directory)/"b",np.ones(2),3)
            self.assertEqual(a["mae_native_units"], b["mae_native_units"])
            self.assertTrue((Path(directory)/"a/model.pt").exists())
            checkpoint = torch.load(Path(directory)/"a/model.pt", weights_only=True)
            self.assertEqual(checkpoint["input_dim"], 3)

    def test_video_timestamp_alignment_and_missing_frame(self):
        import av
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
                for index in range(10):
                    frame = av.VideoFrame.from_ndarray(np.full((16,16,3),index*20,dtype=np.uint8),format="rgb24")
                    frame.pts, frame.time_base = index, Fraction(1,10)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            images = sample_video(path, [.0,.5,.9,2.0], (8,8))
            self.assertLess(abs(float(images[1].mean()) - 100), 10)
            self.assertIsNone(images[3])


if __name__ == "__main__":
    unittest.main()
