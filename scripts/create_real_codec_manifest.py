#!/usr/bin/env python3
"""Create a manifest using FFmpeg G.711/GSM/AMR/Opus codec round-trips."""

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aasist_protocol import sha256
from project_paths import audio_path, stored_audio_path


CODEC_CONFIG = Path(__file__).resolve().parent / "config/telephone-codecs-v2.json"

ENCODER_OPTIONS = {
    "bitrate_bps": "-b:a", "vbr": "-vbr", "application": "-application",
    "frame_duration_ms": "-frame_duration", "cutoff_hz": "-cutoff",
    "packet_loss_percent": "-packet_loss", "fec": "-fec",
}


def codec_arguments(settings):
    return [
        "-ar", str(settings["sample_rate_hz"]), "-ac", str(settings["channels"]),
        "-c:a", settings["codec"],
        *(val for key, flag in ENCODER_OPTIONS.items() if key in settings
          for val in (flag, str(settings[key]))),
    ]


def main() -> int:
    config = json.loads(CODEC_CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=tuple(config["encoders"]), required=True)
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
    log_path = args.output_manifest.with_suffix(".codec-log.json")
    if args.output_manifest.exists() or log_path.exists():
        parser.error("Choose a new manifest and codec log")
    ffmpeg_version = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()[0]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = config["encoders"][args.codec]
    codec_options = codec_arguments(settings)
    decode_options = codec_arguments(config["pcm"])
    extension = settings["extension"]
    with tempfile.TemporaryDirectory() as temp:
        def convert(item):
            index, row = item
            encoded = Path(temp) / f"{index}{extension}"
            target = args.output_dir / f"{index:08d}.wav"
            pending = target.with_name(f".{index:08d}.part.wav")
            source = audio_path(row["path"], args.input_manifest)
            commands = (
                [ffmpeg, "-v", "error", "-y", "-i", str(source), *codec_options, str(encoded)],
                [ffmpeg, "-v", "error", "-y", "-i", str(encoded), *decode_options, str(pending)],
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
                    "channel": f"real_{args.codec}", "codec_protocol": config["version"]}

        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            converted = list(workers.map(convert, enumerate(rows)))

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with log_path.open("x", encoding="utf-8") as handle:
        json.dump({
            "protocol": config["version"],
            "config_sha256": sha256(CODEC_CONFIG),
            "ffmpeg_version": ffmpeg_version,
            "channel": args.codec,
            "encoder": settings,
            "decoded_pcm": config["pcm"],
            "encoder_arguments": codec_options,
            "decoder_arguments": decode_options,
            "split_filter": args.split,
            "records": len(converted),
            "input_manifest_sha256": sha256(args.input_manifest),
            "output_manifest_sha256": sha256(args.output_manifest),
        }, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(converted)} {args.codec} codec records to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
