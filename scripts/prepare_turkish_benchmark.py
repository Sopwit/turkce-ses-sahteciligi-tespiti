#!/usr/bin/env python3
"""Freeze a balanced, duration-matched Turkish anti-spoofing benchmark."""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
import wave
from collections import Counter, defaultdict, deque
from pathlib import Path

from project_paths import PATHS, audio_path, stored_audio_path

FAMILY_SPLIT = {
    "freya": "train", "bark": "train", "glow-tts": "train",
    "omnivoice": "dev", "voxcpm": "test", "xtts": "test",
}
MLAAD_FAMILIES = {
    "freyatts": "freya", "omnivoice": "omnivoice", "voxcpm2": "voxcpm",
    "tts_models_multilingual_multi-dataset_bark": "bark",
    "tts_models_multilingual_multi-dataset_xtts_v1.1": "xtts",
    "tts_models_multilingual_multi-dataset_xtts_v2": "xtts",
    "tts_models_tr_common-voice_glow-tts": "glow-tts",
}


def normalized_text(value):
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def read_metadata(root):
    metadata = {}
    for table in sorted((root / "MLAAD").glob("fake/tr/*/meta.csv")):
        family = MLAAD_FAMILIES[table.parent.name.casefold()]
        with table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="|"):
                metadata[(root / "MLAAD" / row["path"]).resolve()] = (
                    family, row["original_file"], normalized_text(row["transcript"]),
                )
    for table in sorted((root / "girenit-tts-tr-synthetic").glob("*/metadata.csv")):
        with table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["teacher_model"].casefold() != "freyavoice/freya-tts":
                    raise ValueError(f"Unknown Girenit teacher: {row['teacher_model']}")
                metadata[(table.parent / row["file_name"]).resolve()] = (
                    "freya", "", normalized_text(row["text"]),
                )
    return metadata


def audio_info(path, checksum):
    with wave.open(str(path), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
        digest = None
        if checksum:
            header = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
            digest = (header, hashlib.sha256(audio.readframes(audio.getnframes())).digest())
        return duration, digest


def select_synthetic(pools, rng, family_limit):
    picked = {}
    held_content = set()
    seen_audio = set()
    for split in ("test", "dev", "train"):
        split_content = set()
        for family, target in FAMILY_SPLIT.items():
            if target != split:
                continue
            versions = defaultdict(deque)
            candidates = pools[family]
            rng.shuffle(candidates)
            for row, content, audio in candidates:
                versions[row["generator"]].append((row, content, audio))
            count = 0
            while versions and count < family_limit:
                for version in list(versions):
                    row, content, audio = versions[version].popleft()
                    if not versions[version]:
                        del versions[version]
                    if held_content & content or audio in seen_audio:
                        continue
                    picked.setdefault(split, []).append(row)
                    seen_audio.add(audio)
                    split_content.update(content)
                    count += 1
                    if count == family_limit:
                        break
        held_content.update(split_content)
    return picked


def match_real(picked, real, rng, speaker_cap):
    selected = []
    for split, fake_rows in picked.items():
        by_duration = defaultdict(deque)
        rng.shuffle(real[split])
        for row in real[split]:
            by_duration[int(row["duration_s"] * 2)].append(row)
        speaker_counts = Counter()
        for fake in fake_rows:
            bucket = by_duration[int(fake["duration_s"] * 2)]
            while bucket:
                real_row = bucket.popleft()
                speaker = real_row["speaker_id"]
                if speaker_counts[speaker] < speaker_cap:
                    speaker_counts[speaker] += 1
                    selected.extend((fake, real_row))
                    break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", default=PATHS["data"], type=Path)
    parser.add_argument("--seed", default=2209, type=int)
    parser.add_argument("--per-family", default=400, type=int)
    parser.add_argument("--speaker-cap", default=20, type=int)
    args = parser.parse_args()
    if args.output.exists() or min(args.per_family, args.speaker_cap) < 1:
        parser.error("Use a new output path and positive limits")
    rng = random.Random(args.seed)
    root = args.data.resolve()
    metadata = read_metadata(root)
    real, fake = defaultdict(list), defaultdict(list)
    with args.inventory.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["source"] == "FLEURS_TR":
                continue
            path = audio_path(row["path"], args.inventory).resolve()
            relative = path.relative_to(root).as_posix()
            row["path"] = stored_audio_path(path, args.output)
            row["clip_id"] = hashlib.sha256(relative.encode()).hexdigest()
            duration, digest = audio_info(path, not row["label"])
            if not 2.5 <= duration < 8:
                continue
            row["duration_s"] = duration
            if row["label"] == 1:
                if row["source"] != "commonvoice_27_tr_fixed" or not row.get("speaker_id"):
                    raise ValueError("Expected an anonymized Common Voice group")
                real[row["split"]].append(row)
                continue
            family, original, transcript = metadata[path]
            row["family"] = family
            row["split"] = FAMILY_SPLIT[family]
            content = {(kind, value) for kind, value in (
                ("original", original), ("transcript", transcript),
            ) if value}
            fake[family].append((row, content, digest))
    picked = select_synthetic(fake, rng, args.per_family)
    selected = match_real(picked, real, rng, args.speaker_cap)
    totals = Counter((row["split"], row["label"]) for row in selected)
    if any(not totals[split, label] for split in ("train", "dev", "test") for label in (0, 1)):
        raise ValueError(f"A split is missing classes: {totals}")
    families = Counter(row["family"] for row in selected if row["label"] == 0)
    if set(families) != set(FAMILY_SPLIT):
        raise ValueError(f"Missing synthesis families: {set(FAMILY_SPLIT) - set(families)}")
    groups = defaultdict(set)
    for row in selected:
        if row["label"] == 1:
            groups[row["speaker_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Common Voice group overlaps between splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("split/class:", dict(totals))
    print("generator families:", dict(families))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
