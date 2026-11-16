"""Versioned AASIST-L experiment settings and matched training views."""

import hashlib
import json
import random

from project_paths import ROOT, audio_path


EXPERIMENT_V1 = ROOT / "aasist-source/config/aasist-l-experiments-v1.json"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_fingerprint(path):
    settings = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def experiment_settings():
    return json.loads(EXPERIMENT_V1.read_text(encoding="utf-8"))


def verify_initial_weights(path, settings):
    if path is None or sha256(path) != settings["initial_checkpoint_sha256"]:
        raise ValueError("Use the original, unfine-tuned AASIST-L checkpoint in the protocol")


def require_clean_development(rows):
    if any(row.get("schema_version") != 2 or row.get("source") == "FLEURS_TR"
           or row.get("channel") not in (None, "clean") for row in rows):
        raise ValueError("Named experiments require clean V2 development without FLEURS")


def channel_training_rows(clean, manifest_paths, seed, channels):
    """Return one view per clean training clip, with balanced views per label."""
    by_id = {row.get("clip_id"): row for row in clean}
    if None in by_id or "" in by_id or len(by_id) != len(clean):
        raise ValueError("Clean training must have unique clip_id values")
    views = {}
    for path in manifest_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                channel = row.get("channel")
                clip_id = row.get("clip_id")
                key = channel, clip_id
                original = by_id.get(clip_id)
                if channel not in channels or original is None or key in views:
                    raise ValueError("Unexpected or duplicate channel training view")
                same = ("label", "source", "family", "generator", "speaker_roles", "groups", "scenario")
                if row.get("split") != "train" or any(row.get(k) != original.get(k) for k in same):
                    raise ValueError("Channel views must match a clean train row and its identity metadata")
                # Keep the verified clean record as the authority for all lineage fields.
                views[key] = {**original, "path": str(audio_path(row["path"], path)), "channel": channel}
    if len(views) != len(channels) * len(clean):
        raise ValueError("M2 requires exactly one view per configured channel and train clip")
    rng = random.Random(seed)
    selected = {}
    for label in (0, 1):
        ids = sorted(key for key, row in by_id.items() if row["label"] == label)
        rng.shuffle(ids)
        choices = ("clean", *channels)
        for index, key in enumerate(ids):
            channel = choices[index % len(choices)]
            selected[key] = by_id[key] if channel == "clean" else views[channel, key]
    return [selected[row["clip_id"]] for row in clean]
