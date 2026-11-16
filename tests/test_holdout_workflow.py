import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
DEPENDENCIES = ("torch", "numpy", "soundfile", "sklearn")
HAS_MODEL_RUNTIME = all(importlib.util.find_spec(name) for name in DEPENDENCIES)


@unittest.skipUnless(HAS_MODEL_RUNTIME, "requires the local ML runtime")
class HoldoutWorkflowTest(unittest.TestCase):
    def setUp(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 2)

            def forward(self, audio):
                return audio, self.linear(audio)

        self.model = TinyModel
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.config = self.folder / "model.json"
        self.config.write_text(json.dumps({
            "model_config": {"nb_samp": 8, "architecture": "test"}, "num_epochs": 1,
            "batch_size": 2, "optim_config": {"base_lr": 0.001, "weight_decay": 0.001},
        }))
        self.rows = []
        for split in ("train", "dev", "test"):
            for label in (0, 1):
                path = self.folder / f"{split}-{label}.wav"
                if split != "test":
                    with wave.open(str(path), "wb") as f:
                        f.setparams((1, 2, 16000, 0, "NONE", ""))
                        f.writeframes(bytes([1 + label, 0]) * 8)
                self.rows.append({"path": str(path), "split": split, "label": label, "clip_id": path.stem})
        self.manifest = self.folder / "combined.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def test_training_never_loads_test_audio_and_allows_dev_only(self):
        import train_turkish as train

        for index, manifest in enumerate((self.manifest, self.folder / "development.jsonl")):
            if not manifest.exists():
                manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows if row["split"] != "test"))
            output = self.folder / f"training-{index}"
            argv = ["train", "--manifest", str(manifest), "--config", str(self.config),
                    "--output", str(output), "--device", "cpu"]
            opened = []
            original = train.TurkishAudioDataset.__getitem__

            def record(dataset, idx):
                opened.append(dataset.records[idx]["split"])
                return original(dataset, idx)

            with patch.object(train, "build_model", side_effect=lambda *_: self.model()), \
                    patch.object(train.TurkishAudioDataset, "__getitem__", record), patch.object(sys, "argv", argv):
                self.assertEqual(train.main(), 0)
            self.assertEqual(set(opened), {"train", "dev"})
            self.assertTrue((output / "best.pth").exists())
            self.assertNotIn("test_metrics", json.loads((output / "metrics.json").read_text()))

    def test_evaluator_separates_freeze_and_heldout_test(self):
        import evaluate_turkish_channels as evaluator
        import torch

        weights = self.folder / "model.pth"
        torch.save(self.model().state_dict(), weights)
        lock = self.folder / "lock.json"
        report = self.folder / "report.json"
        base = ["evaluate", "--manifest", str(self.manifest), "--config", str(self.config),
                "--weights", str(weights)]
        calls = []

        def score(_model, rows, *_):
            calls.extend(row["split"] for row in rows)
            return [0.8 if row["label"] else 0.2 for row in rows]

        with patch.object(evaluator, "build_model", side_effect=lambda *_: self.model()), \
                patch.object(evaluator, "score", side_effect=score):
            with patch.object(sys, "argv", base + ["--freeze", "--output", str(lock)]):
                self.assertEqual(evaluator.main(), 0)
            self.assertEqual(calls, ["dev", "dev"])
            with patch.object(sys, "argv", base + ["--lock", str(lock), "--output", str(report)]):
                self.assertEqual(evaluator.main(), 0)
            self.assertEqual(calls, ["dev", "dev", "test", "test"])
            frozen = json.loads(lock.read_text())
            self.assertEqual(json.loads(report.read_text())["threshold"], frozen["threshold"])
            self.rows[-1]["clip_id"] = "dev-1"
            self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            with patch.object(sys, "argv", base + ["--lock", str(lock), "--output", str(self.folder / "bad")]):
                with self.assertRaises(SystemExit):
                    evaluator.main()
            self.assertEqual(len(calls), 4)

    def test_step_limit_stops_before_reading_extra_batches(self):
        import torch
        from train_turkish import run_epoch

        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters())
        count = 0
        def batches():
            nonlocal count
            for _ in range(3):
                count += 1
                yield torch.zeros((2, 8)), torch.tensor([0, 1])
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            run_epoch(model, batches(), torch.nn.CrossEntropyLoss(), torch.device("cpu"),
                      optimizer, max_batches=2)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(count, 2)

    def test_prepared_phone_channel_ignores_missing_test_audio(self):
        import create_phone_channel_manifest as codec

        output = self.folder / "phone.jsonl"
        argv = ["codec", "--input-manifest", str(self.manifest), "--output-manifest",
                str(output), "--output-dir", str(self.folder / "phone"),
                "--split", "train", "--split", "dev"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(codec.main(), 0)
        converted = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(converted), 4)
        self.assertEqual({row["split"] for row in converted}, {"train", "dev"})
