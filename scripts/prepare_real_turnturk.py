#!/usr/bin/env python3
"""Create a genuine-only Turkish video-call evaluation set from Real-TurnTurk."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path

from project_paths import PATHS, stored_audio_path

SAMPLE_RATE = 16000


def seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=PATHS["raw"] / "Real-TurnTurk", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-speaker", default=20, type=int)
    parser.add_argument("--seed", default=2209, type=int)
    args = parser.parse_args()
    if args.output_dir.exists() or args.manifest.exists() or args.per_speaker < 1:
        parser.error("Use new output paths and a positive limit")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("FFmpeg is required")
    candidates = defaultdict(list)
    for table in sorted(args.source.glob("rec*/*__text.csv")):
        if table.name.startswith("."):
            continue
        channels = {
            path.stem.rsplit("_", 1)[1]: path
            for path in table.parent.glob("*.webm") if not path.name.startswith(".")
        }
        with table.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                start, end = seconds(row["start_time"]), seconds(row["end_time"])
                speaker = row["speaker"]
                if 2.5 <= end - start < 8 and row["text"].strip():
                    candidates[speaker].append((channels[speaker], start, end))
    if not candidates:
        parser.error(f"No eligible transcript segments in {args.source}")
    selected = defaultdict(list)
    rng = random.Random(args.seed)
    for speaker, rows in sorted(candidates.items()):
        rows = sorted(set(rows))
        for clip, start, end in rng.sample(rows, min(len(rows), args.per_speaker)):
            selected[clip].append((speaker, start, end))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        for source, spans in sorted(selected.items()):
            pcm = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(source), "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-f", "s16le", "-"],
                capture_output=True, check=True,
            ).stdout
            for speaker, start, end in sorted(spans):
                samples = pcm[round(start * SAMPLE_RATE) * 2:round(end * SAMPLE_RATE) * 2]
                if abs(len(samples) / (2 * SAMPLE_RATE) - (end - start)) > 0.025 or not any(samples):
                    raise ValueError(f"Invalid segment in {source} at {start:.3f} seconds")
                key = hashlib.sha256(f"{source.name}/{start:.3f}/{end:.3f}".encode()).hexdigest()
                target = args.output_dir / f"{key}.wav"
                with wave.open(str(target), "wb") as audio:
                    audio.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                    audio.writeframes(samples)
                counts[speaker] += 1
                row = {
                    "path": stored_audio_path(target, args.manifest),
                    "clip_id": key, "split": "test", "label": 1, "source": "Real-TurnTurk",
                    "speaker_id": f"Real-TurnTurk/{speaker}", "duration_s": len(samples) / (2 * SAMPLE_RATE),
                    "source_clip": source.name, "start_s": start, "end_s": end,
                }
                handle.write(json.dumps(row) + "\n")
    print(f"Wrote {counts.total()} external genuine clips from {len(counts)} speakers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
