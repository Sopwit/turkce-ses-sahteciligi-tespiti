#!/usr/bin/env python3
"""Verify or export a bounded Common Voice Turkish subset as WAV files."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from project_paths import PATHS, stored_audio_path


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.casefold() in {".wav", ".flac"}:
        samples, rate = sf.read(path, dtype="float32")
    else:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-f", "wav", "-"],
            check=True,
            capture_output=True,
        )
        samples, rate = sf.read(io.BytesIO(result.stdout), dtype="float32")
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    return samples, rate


def export_official_archive(args: argparse.Namespace) -> int:
    source = args.source_dir.resolve()
    source_name = args.output.name
    clips = source / "clips"
    if not clips.is_dir():
        raise SystemExit(f"Common Voice clips klasörü bulunamadı: {clips}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = []
    split_files = {
        "train": source / "train.tsv",
        "validation": source / "dev.tsv",
        "test": source / "test.tsv",
    }
    for split, metadata_path in split_files.items():
        if not metadata_path.is_file():
            raise SystemExit(f"Common Voice metadata dosyası bulunamadı: {metadata_path}")
        before = len(rows)
        with metadata_path.open(encoding="utf-8", newline="") as handle:
            metadata = csv.DictReader(handle, delimiter="\t")
            for index, item in enumerate(metadata):
                if not args.all and index >= args.limit:
                    break
                filename = item.get("path", "")
                speaker_id = item.get("client_id", "").strip()
                if args.require_speaker_id and not speaker_id:
                    raise SystemExit(f"client_id eksik: {metadata_path}:{index + 2}")
                source_audio = clips / filename
                if not source_audio.is_file():
                    raise SystemExit(f"Common Voice ses dosyası bulunamadı: {source_audio}")
                target = args.output / split / f"{index:08d}.wav"
                target.parent.mkdir(parents=True, exist_ok=True)
                samples, rate = read_audio(source_audio)
                sf.write(target, samples, rate, subtype="PCM_16")
                row = {
                    "path": stored_audio_path(target, args.manifest),
                    "split": split,
                    "label": 1,
                    "source": source_name,
                    "generator": source_name,
                }
                if speaker_id:
                    row["speaker_id"] = speaker_id
                rows.append(row)
                if args.provenance:
                    provenance.append({
                        "source_clip": filename,
                        "sentence_id": item.get("sentence_id", ""),
                        "split": split,
                        "path": stored_audio_path(target, args.provenance),
                    })
        print(f"{metadata_path.name}: {len(rows) - before} exported")
    if args.provenance:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        with args.provenance.open("x", encoding="utf-8") as handle:
            for row in provenance:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} Common Voice records to {args.manifest}")
    if args.require_speaker_id:
        print("Speaker metadata: client_id -> speaker_id")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PATHS["data"] / "commonvoice_17_tr_fixed")
    parser.add_argument("--manifest", type=Path, help="Inventory output path")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Official Common Voice archive containing clips/ and train/dev/test.tsv",
    )
    parser.add_argument(
        "--provenance", type=Path,
        help="New JSONL sidecar retaining original clip and sentence IDs from an official archive",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Export every record in each selected split")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--require-speaker-id",
        action="store_true",
        help="Fail if Common Voice metadata has no client_id",
    )
    args = parser.parse_args()
    if args.manifest is None:
        parent = PATHS["manifests"] if args.output.parent.resolve() == PATHS["data"] else args.output.parent
        args.manifest = parent / "commonvoice_tr_manifest.jsonl"
    if args.provenance:
        if not args.source_dir:
            parser.error("--provenance requires --source-dir")
        if args.provenance.exists() or args.provenance.resolve() == args.manifest.resolve():
            parser.error("--provenance must be a new path distinct from --manifest")
    if args.source_dir:
        return export_official_archive(args)
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise SystemExit(
            "HF kaynağı için 'datasets' paketi gerekli; resmi arşiv aktarımında "
            "--source-dir kullanın."
        ) from error
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        dataset = load_dataset("ysdede/commonvoice_17_tr_fixed", split=split, streaming=True)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for index, item in enumerate(dataset):
            if not args.all and index >= args.limit:
                break
            target = args.output / split / f"{index:08d}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            audio = item["audio"]
            if audio.get("bytes") is not None:
                samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            else:
                samples, rate = read_audio(Path(audio["path"]))
            sf.write(target, samples, rate, subtype="PCM_16")
            speaker_id = item.get("client_id")
            row = {"path": stored_audio_path(target, args.manifest), "split": split, "label": 1,
                   "source": "commonvoice_17_tr_fixed",
                   "generator": "commonvoice_17_tr_fixed"}
            if speaker_id:
                row["speaker_id"] = str(speaker_id)
            if args.require_speaker_id and not speaker_id:
                raise SystemExit(
                    "Common Voice HF kaynağı client_id içermiyor; "
                    "resmi arşiv için --source-dir kullanın"
                )
            rows.append(row)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} Common Voice records to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
