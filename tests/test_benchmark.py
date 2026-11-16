"""Verify split hygiene without model or audio dependencies."""

from collections import defaultdict
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_turkish_manifest import rebalance_holdout_dev  # noqa: E402
from prepare_turkish_benchmark import match_real, select_synthetic  # noqa: E402


class BenchmarkTest(unittest.TestCase):
    def test_holdout_rebalance_moves_whole_speaker_group(self):
        rows = [
            {"split": "dev", "label": 1, "speaker_id": "real"}
            for _ in range(21)
        ]
        for speaker, size in (("a", 3), ("b", 4)):
            rows.extend({
                "split": "train", "label": 0, "speaker_id": speaker,
                "generator": "allowed", "path": f"{speaker}-{i}",
            } for i in range(size))
        rebalance_holdout_dev(rows, "excluded", speaker_disjoint=True)
        splits = defaultdict(set)
        for row in rows:
            splits[row["speaker_id"]].add(row["split"])
        self.assertEqual(splits["a"], {"dev"})
        self.assertEqual(splits["b"], {"train"})

    def test_shared_synthetic_content_is_kept_in_only_one_split(self):
        pools = defaultdict(list)
        pools["xtts"].append(({"generator": "xtts-v2"}, {("original", "shared")}, b"xtts"))
        pools["freya"].extend((
            ({"generator": "freya-overlap"}, {("original", "shared")}, b"overlap"),
            ({"generator": "freya-independent"}, {("text", "unique")}, b"independent"),
        ))
        picked = select_synthetic(pools, random.Random(0), 20)
        self.assertEqual(picked["test"], [{"generator": "xtts-v2"}])
        self.assertEqual(picked["train"], [{"generator": "freya-independent"}])

    def test_duration_matching_caps_a_speaker(self):
        fake = {"test": [{"duration_s": 3.2, "label": 0}] * 3}
        real = {"test": [
            {"duration_s": 3.1, "label": 1, "speaker_id": "a"},
            {"duration_s": 3.3, "label": 1, "speaker_id": "a"},
            {"duration_s": 3.4, "label": 1, "speaker_id": "b"},
            {"duration_s": 5.1, "label": 1, "speaker_id": "c"},
        ]}
        matched = match_real(fake, real, random.Random(0), speaker_cap=1)
        self.assertEqual([row["label"] for row in matched], [0, 1, 0, 1])
        self.assertEqual({row["speaker_id"] for row in matched if row["label"]}, {"a", "b"})
