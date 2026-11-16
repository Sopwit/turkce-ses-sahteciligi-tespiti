import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from aasist_protocol import channel_training_rows, experiment_settings, require_clean_development, verify_initial_weights
from prepare_turkish_protocol_v2 import COMMON_VOICE, FLEURS, create_protocol
from validate_manifest import speaker_audit


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def person(identity):
    return {"id": identity, "kind": "person", "verified": True, "evidence": "fixture"}


class ProtocolTest(unittest.TestCase):
    def test_person_overlap_across_different_roles(self):
        rows = [
            {"split": "train", "scenario": "real", "speaker_roles": {"source": person("p1")}},
            {"split": "test", "scenario": "converted",
             "speaker_roles": {"source": person("p2"), "target": person("p1")}},
        ]
        audit = speaker_audit(rows)
        self.assertTrue(audit["verified_person_role_coverage"])
        self.assertEqual(audit["verified_person_ids_across_splits"], 1)
        self.assertFalse(audit["speaker_disjoint"])
        rows[1]["speaker_roles"]["target"] = person("p3")
        self.assertTrue(speaker_audit(rows)["speaker_disjoint"])

    def test_non_person_and_missing_roles_fail_closed(self):
        rows = [{"split": "train", "scenario": "real", "speaker_id": "generated-group"},
                {"split": "dev", "scenario": "converted", "speaker_roles": {"source": person("p2")}},
                {"split": "test", "scenario": "cloned",
                 "speaker_roles": {"reference": {**person("p3"), "kind": "example"}}}]
        self.assertEqual(speaker_audit(rows)["unverified_role_rows_by_split"],
                         {"train": 1, "dev": 1, "test": 1})
        with TemporaryDirectory() as temporary:
            path = write_rows(Path(temporary) / "legacy.jsonl", rows)
            completed = subprocess.run([sys.executable, str(SCRIPTS / "validate_manifest.py"),
                                        str(path), "--speaker-disjoint"],
                                       capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Speaker-disjoint criteria failed", completed.stderr)

    def test_versioned_inventory_reserves_fleurs_and_never_relabels_groups(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = []
            for split in ("train", "dev"):
                rows.extend([
                    {"path": f"{split}.wav", "split": split, "source": COMMON_VOICE, "label": 1,
                     "speaker_id": split, "clip_id": split},
                    {"path": f"{split}-tts.wav", "split": split, "source": "MLAAD", "label": 0,
                     "speaker_id": "method", "clip_id": f"{split}-tts"},
                ])
            fleur = {"path": "fleur.wav", "split": "dev", "label": 1,
                     "source": FLEURS, "speaker_id": "sentence"}
            inventory = write_rows(folder / "legacy.jsonl", rows + [fleur])
            inputs = (inventory, write_rows(folder / "p0.jsonl", rows + [{"split": "test"}]),
                      write_rows(folder / "cv.jsonl", rows[::2]),
                      write_rows(folder / "external.jsonl", [{**fleur, "split": "test", "clip_id": "F"}]))
            original = inventory.read_bytes()
            catalog, dev, report = create_protocol(*inputs, folder / "v2")
            self.assertEqual(inventory.read_bytes(), original)
            self.assertEqual(report["legacy_dev_fleurs_overlap_by_path"], 1)
            self.assertEqual(catalog[-1]["groups"], {"example_group": "sentence"})
            self.assertEqual(catalog[-1]["split"], "external")
            self.assertIsNone(dev[1]["speaker_roles"]["source"])
            self.assertFalse(report["speaker_audit_development"]["speaker_disjoint"])
            write_rows(inputs[1], rows + [fleur])
            with self.assertRaisesRegex(ValueError, "reserved"):
                create_protocol(*inputs, folder / "v2")

    def test_m0_hash_check_and_m1_m2_share_step_budget(self):
        settings = experiment_settings()
        self.assertEqual(settings["arms"]["M0"]["optimizer_steps"], 0)
        self.assertEqual(settings["arms"]["M1"]["optimizer_steps"],
                         settings["arms"]["M2"]["optimizer_steps"])
        self.assertNotEqual(settings["initial_checkpoint_sha256"],
                            settings["historical"]["P0"]["checkpoint_sha256"])
        with TemporaryDirectory() as temporary:
            weights = Path(temporary) / "not-m0.pth"
            weights.write_bytes(b"not published weights")
            with self.assertRaisesRegex(ValueError, "original"):
                verify_initial_weights(weights, settings)

    def test_m2_retains_one_matched_view_per_base_clip(self):
        clean = [{"clip_id": str(i), "split": "train", "path": f"clean-{i}.wav", "label": i % 2}
                 for i in range(12)]
        channels = ("real_g711_mulaw", "real_opus")
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = [write_rows(folder / f"{channel}.jsonl", [
                {**row, "path": f"{channel}-{row['clip_id']}.wav", "channel": channel}
                for row in clean]) for channel in channels]
            chosen = channel_training_rows(clean, paths, 2209, channels)
            self.assertEqual([r["clip_id"] for r in chosen], [r["clip_id"] for r in clean])
            self.assertEqual([r["label"] for r in chosen], [r["label"] for r in clean])
            for label in (0, 1):
                self.assertEqual({r.get("channel", "clean") for r in chosen if r["label"] == label},
                                 {"clean", *channels})
            with self.assertRaisesRegex(ValueError, "exactly one"):
                channel_training_rows(clean, paths[:1], 2209, channels)
            wrong_split = [{**row, "channel": channels[0], "split": "test"} for row in clean]
            write_rows(paths[0], wrong_split)
            with self.assertRaisesRegex(ValueError, "train row"):
                channel_training_rows(clean, paths, 2209, channels)

    def test_named_arms_require_clean_v2_without_fleurs(self):
        clean = {"schema_version": 2, "source": COMMON_VOICE}
        require_clean_development([clean])
        for bad in ({**clean, "source": FLEURS}, {**clean, "schema_version": 1},
                    {**clean, "channel": "real_opus"}):
            with self.assertRaisesRegex(ValueError, "clean V2"):
                require_clean_development([bad])
