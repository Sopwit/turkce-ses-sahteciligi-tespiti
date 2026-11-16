#!/usr/bin/env python3
"""Create a manifest using FFmpeg G.711/GSM/AMR/Opus codec round-trips."""

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from project_paths import audio_path, stored_audio_path


CODECS = {
    "g711_mulaw": (["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"], ".wav"),
    "gsm": (["-ar", "8000", "-ac", "1", "-c:a", "gsm"], ".wav"),
    "opus": (["-ar", "8000", "-ac", "1", "-c:a", "libopus", "-b:a", "12k"], ".ogg"),
    "amr_nb": (["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], ".amr"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=tuple(CODECS), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required for real codec conversion")
    rows = [json.loads(line) for line in args.input_manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.split is not None:
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("No records selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    codec_options, extension = CODECS[args.codec]
    with tempfile.TemporaryDirectory() as temp:
        def convert(item):
            index, row = item
            encoded = Path(temp) / f"{index}{extension}"
            target = args.output_dir / f"{index:08d}.wav"
            pending = target.with_name(f".{index:08d}.part.wav")
            source = audio_path(row["path"], args.input_manifest)
            commands = (
                [ffmpeg, "-v", "error", "-y", "-i", str(source), *codec_options, str(encoded)],
                [ffmpeg, "-v", "error", "-y", "-i", str(encoded), "-ar", "8000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(pending)],
            )
            try:
                for command in commands:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                pending.replace(target)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"FFmpeg {args.codec} round-trip failed for {source}: {error.stderr.strip()}"
                ) from error
            finally:
                encoded.unlink(missing_ok=True)
                pending.unlink(missing_ok=True)
            return {**row, "path": stored_audio_path(target, args.output_manifest),
                    "channel": f"real_{args.codec}"}

        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            converted = list(workers.map(convert, enumerate(rows)))

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output_manifest.parent, delete=False,
    ) as handle:
        pending_manifest = Path(handle.name)
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        pending_manifest.replace(args.output_manifest)
    finally:
        pending_manifest.unlink(missing_ok=True)
    print(f"Wrote {len(converted)} {args.codec} codec records to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
