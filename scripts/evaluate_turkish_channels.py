#!/usr/bin/env python3
"""Measure a fixed checkpoint on the same Turkish test clips in multiple channels."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from math import isfinite

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from aasist_protocol import config_fingerprint, experiment_settings, require_clean_development, sha256, verify_initial_weights
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
        for field in ("label", "source", "generator", "family", "speaker_id", "speaker_roles", "groups", "scenario"):
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
            source = row.get("speaker_roles", {}).get("source") or {}
            speaker = source.get("id") if source.get("verified") is True else row.get("speaker_id")
            if speaker:
                false_alarms[speaker].append(genuine_score < threshold)
        else:
            misses[row.get("family") or row.get("generator") or "unknown"].append(genuine_score >= threshold)
    tn, fp, fn, tp = np.ravel(result["confusion_matrix"])
    result.update({
        "records": len(rows),
        "false_alarm_on_real": float(fn / (fn + tp)),
        "missed_synthetic": float(fp / (fp + tn)),
        "speaker_macro_false_alarm": (
            float(np.mean([np.mean(items) for items in false_alarms.values()]))
            if false_alarms and sum(map(len, false_alarms.values())) == fn + tp else None
        ),
        "synthetic_families": {
            family: {"records": len(items), "missed_synthetic": float(np.mean(items))}
            for family, items in sorted(misses.items())
        },
    })
    return result


LOCK_VERSION = "genuine-softmax-threshold-v1"


def partition_keys(rows):
    keys = set()
    for row in rows:
        keys.add("path:" + str(Path(row["path"]).resolve()))
        if row.get("clip_id"):
            keys.add("clip:" + row["clip_id"])
    return {hashlib.sha256(key.encode()).hexdigest() for key in keys}


def choose_threshold(labels, scores):
    if set(labels) != {0, 1} or not all(isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Threshold selection needs both classes and finite genuine softmax scores")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    best = max(finite, key=lambda i: (tpr[i] - fpr[i], -abs(thresholds[i] - 0.5)))
    return float(thresholds[best])


def load_lock(path, fingerprint):
    lock = json.loads(path.read_text(encoding="utf-8"))
    if any(lock.get(key) != value for key, value in fingerprint.items()):
        raise ValueError("Threshold lock does not match the model, config or arm")
    threshold = lock.get("threshold")
    if type(threshold) not in (int, float) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid genuine-score threshold lock")
    return lock


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "weights", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Score dev only and write a threshold lock")
    mode.add_argument("--lock", type=Path, help="Read a frozen threshold and score test only")
    parser.add_argument("--arm", choices=("M0", "M1", "M2", "P0"))
    parser.add_argument("--g711", type=Path)
    parser.add_argument("--opus", type=Path)
    parser.add_argument("--external-real", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Use a new output path and a positive batch size")
    if args.freeze and (args.g711 or args.opus or args.external_real):
        parser.error("Freeze mode uses clean dev only")
    if args.arm:
        protocol = experiment_settings()
        if config_fingerprint(args.config) != protocol["base_config_fingerprint"]:
            parser.error("The named arm requires the original AASIST-L config")
        if args.arm == "M0":
            verify_initial_weights(args.weights, protocol)
        if args.arm in ("M1", "M2"):
            run = json.loads((args.weights.parent / "metrics.json").read_text(encoding="utf-8"))
            arm = protocol["arms"][args.arm]
            if (run.get("arm") != args.arm or run.get("experiment") != protocol["version"]
                    or run.get("checkpoint_sha256") != sha256(args.weights)
                    or run.get("checkpoint_optimizer_steps") != arm["optimizer_steps"]
                    or run.get("initial_checkpoint_sha256") != protocol["initial_checkpoint_sha256"]
                    or (args.freeze and run.get("manifest_sha256") != sha256(args.manifest))):
                parser.error("Checkpoint and recorded training protocol do not match the named arm")
        if args.arm == "P0" and sha256(args.weights) != protocol["historical"]["P0"]["checkpoint_sha256"]:
            parser.error("P0 requires the historical one-epoch checkpoint")
    fingerprint = {
        "version": LOCK_VERSION,
        "checkpoint_sha256": sha256(args.weights),
        "config_sha256": sha256(args.config),
        "arm": args.arm,
    }
    split = "dev" if args.freeze else "test"
    lock = None if args.freeze else load_lock(args.lock, fingerprint)
    records = load_manifest(args.manifest, (split,))[split]
    if args.freeze and args.arm in ("M0", "M1", "M2"):
        require_clean_development(records)
    if {row["label"] for row in records} != {0, 1}:
        parser.error(f"{split} requires both classes")
    keys = partition_keys(records)
    if lock and keys.intersection(lock["dev_membership_hashes"]):
        parser.error("Test overlaps threshold-selection dev by path or clip_id")
    conditions = {"clean": records}
    for channel, path in (("g711_mulaw", args.g711), ("opus", args.opus)):
        if path:
            variants = paired_rows(path, records, f"real_{channel}")
            if lock and partition_keys(variants).intersection(lock["dev_membership_hashes"]):
                parser.error("Channel test overlaps threshold-selection dev by path or clip_id")
            conditions[channel] = variants
    device = torch.device(args.device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(config, device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    samples = config["model_config"]["nb_samp"]
    if args.freeze:
        scores = score(model, records, samples, args.batch_size, device)
        write_result(args.output, {
            **fingerprint,
            "threshold": choose_threshold([row["label"] for row in records], scores),
            "threshold_selected_on": "clean dev, maximum balanced accuracy, genuine score",
            "dev_manifest_sha256": sha256(args.manifest),
            "dev_membership_hashes": sorted(keys),
            "dev_records": len(records),
            "device": str(device),
        })
        return 0

    threshold = lock["threshold"]
    output = {
        **fingerprint,
        "manifest_sha256": sha256(args.manifest),
        "channel_manifest_sha256": {
            name: sha256(path) for name, path in (("g711_mulaw", args.g711), ("opus", args.opus)) if path
        },
        "threshold_lock_sha256": sha256(args.lock),
        "threshold_selected_on": lock["threshold_selected_on"],
        "threshold": threshold,
        "dev_records": lock["dev_records"],
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
        if partition_keys(external).intersection(lock["dev_membership_hashes"]):
            raise ValueError(f"External control overlaps threshold-selection dev: {path}")
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
    write_result(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
