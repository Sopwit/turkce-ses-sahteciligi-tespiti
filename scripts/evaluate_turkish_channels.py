#!/usr/bin/env python3
"""Measure a fixed checkpoint on the same Turkish test clips in multiple channels."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from metrics import classification_metrics
from project_paths import audio_path
from train_turkish import TurkishAudioDataset, build_model, load_manifest


def score(model, rows, samples, batch_size, device):
    loader = DataLoader(TurkishAudioDataset(rows, samples, training=False), batch_size=batch_size)
    scores = []
    with torch.inference_mode():
        for audio, _ in loader:
            _, logits = model(audio.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return scores


def paired_rows(path, clean, expected_channel):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("clip_id")
            if row.get("split") != "test" or key in rows or row.get("channel") != expected_channel:
                raise ValueError(f"Invalid or duplicated test record in {path}")
            row["path"] = str(audio_path(row["path"], path))
            rows[key] = row
    keys = [row["clip_id"] for row in clean]
    if set(keys) != rows.keys() or len(keys) != len(rows):
        raise ValueError(f"Test clip IDs differ in {path}")
    for expected in clean:
        actual = rows[expected["clip_id"]]
        for field in ("label", "source", "generator", "family", "speaker_id"):
            if actual.get(field) != expected.get(field):
                raise ValueError(f"Mismatched {field} for clip {expected['clip_id']} in {path}")
    return [rows[key] for key in keys]


def metrics(rows, scores, threshold):
    labels = [row["label"] for row in rows]
    result = classification_metrics(labels, scores, threshold)
    false_alarms = defaultdict(list)
    misses = defaultdict(list)
    for row, genuine_score in zip(rows, scores, strict=True):
        if row["label"] == 1:
            false_alarms[row["speaker_id"]].append(genuine_score < threshold)
        else:
            misses[row["family"]].append(genuine_score >= threshold)
    tn, fp, fn, tp = np.ravel(result["confusion_matrix"])
    result.update({
        "records": len(rows),
        "false_alarm_on_real": float(fn / (fn + tp)),
        "missed_synthetic": float(fp / (fp + tn)),
        "speaker_macro_false_alarm": float(np.mean([np.mean(items) for items in false_alarms.values()])),
        "synthetic_families": {
            family: {"records": len(items), "missed_synthetic": float(np.mean(items))}
            for family, items in sorted(misses.items())
        },
    })
    return result


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "g711", "opus", "weights", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--external-real", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Use a new output path and a positive batch size")
    partitions = load_manifest(args.manifest)
    test = partitions["test"]
    conditions = {
        "clean": test,
        "g711_mulaw": paired_rows(args.g711, test, "real_g711_mulaw"),
        "opus": paired_rows(args.opus, test, "real_opus"),
    }
    device = torch.device(args.device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(config, device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    samples = config["model_config"]["nb_samp"]
    dev = partitions["dev"]
    dev_scores = score(model, dev, samples, args.batch_size, device)
    fpr, tpr, thresholds = roc_curve([row["label"] for row in dev], dev_scores, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    best = max(finite, key=lambda i: (tpr[i] - fpr[i], -abs(thresholds[i] - 0.5)))
    threshold = float(thresholds[best])
    output = {
        "checkpoint_sha256": sha256(args.weights),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "channel_manifest_sha256": {"g711_mulaw": sha256(args.g711), "opus": sha256(args.opus)},
        "threshold_selected_on": "clean dev, maximum balanced accuracy, genuine score",
        "threshold": threshold,
        "dev_records": len(dev),
        "device": str(device),
        "test": {},
    }
    for name, rows in conditions.items():
        print(f"Scoring {len(rows)} {name} test clips", flush=True)
        scores = score(model, rows, samples, args.batch_size, device)
        output["test"][name] = metrics(rows, scores, threshold)
    baseline = output["test"]["clean"]
    for name, result in output["test"].items():
        if name != "clean":
            result["change_from_clean"] = {
                metric: result[metric] - baseline[metric]
                for metric in ("eer", "false_alarm_on_real", "missed_synthetic", "balanced_accuracy")
            }
    output["external_real"] = {}
    for path in args.external_real:
        external = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != "test" or row.get("label") != 1:
                    raise ValueError(f"External genuine manifest contains non-test or synthetic data: {path}")
                row["path"] = str(audio_path(row["path"], path))
                external.append(row)
        if not external or path.name in output["external_real"]:
            raise ValueError(f"External genuine manifest is empty or repeated: {path}")
        scores = score(model, external, samples, args.batch_size, device)
        by_speaker = defaultdict(list)
        for row, probability in zip(external, scores, strict=True):
            if row.get("speaker_id"):
                by_speaker[row["speaker_id"]].append(probability < threshold)
        output["external_real"][path.name] = {
            "records": len(external), "manifest_sha256": sha256(path),
            "false_alarm_on_real": float(np.mean(np.asarray(scores) < threshold)),
            "speaker_count": len(by_speaker),
            "speaker_macro_false_alarm": (
                float(np.mean([np.mean(values) for values in by_speaker.values()]))
                if len(by_speaker) and sum(map(len, by_speaker.values())) == len(external)
                else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
