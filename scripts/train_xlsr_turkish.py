#!/usr/bin/env python3
"""Fine-tune a wav2vec2-XLS-R classifier from the Turkish JSONL manifest."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from metrics import classification_metrics
from project_paths import PATHS, audio_path


class ManifestAudio(Dataset):
    def __init__(self, records: list[dict], training: bool):
        self.records = records
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        audio, rate = sf.read(record["path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if not len(audio):
            raise ValueError(f"Empty audio file: {record['path']}")
        if rate != 16000:
            audio = resample_poly(audio, 16000, rate).astype("float32")
        if len(audio) > 64000:
            start = random.randint(0, len(audio) - 64000) if self.training else 0
            audio = audio[start:start + 64000]
        elif len(audio) < 64000:
            audio = np.pad(audio, (0, 64000 - len(audio)), mode="wrap")
        return {"audio": audio, "label": record["label"]}


def collate(batch: list[dict], extractor):
    return {
        **extractor([item["audio"] for item in batch], sampling_rate=16000,
                    padding=True, return_tensors="pt"),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def evaluate(model, loader, device, *, collect_scores=False):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    labels, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**batch)
            expected = batch["labels"]
            loss_sum += result.loss.item() * len(expected)
            correct += (result.logits.argmax(1) == expected).sum().item()
            total += len(expected)
            if collect_scores:
                labels.extend(expected.cpu().tolist())
                scores.extend(result.logits.softmax(dim=1)[:, 1].cpu().tolist())
    metrics = classification_metrics(labels, scores) if collect_scores else None
    return loss_sum / total, correct / total, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--output", type=Path, default=PATHS["runs"] / "XLSR")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["path"] = str(audio_path(row["path"], args.manifest))
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "dev", "test")}
    if any({row["label"] for row in items} != {0, 1} for items in splits.values()):
        raise SystemExit("Manifest must contain both labels in train, dev, and test")
    if args.device == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        name = args.device
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    device = torch.device(name)
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model, num_labels=2, label2id={"genuine": 1, "synthetic": 0},
        id2label={0: "synthetic", 1: "genuine"}, ignore_mismatched_sizes=True,
    ).to(device)
    loaders = {
        split: DataLoader(ManifestAudio(items, split == "train"), args.batch_size,
                          shuffle=split == "train", collate_fn=lambda b: collate(b, extractor))
        for split, items in splits.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dev_loss, accuracy, _ = evaluate(model, loaders["dev"], device)
        history.append({"epoch": epoch, "dev_loss": dev_loss, "dev_accuracy": accuracy})
        print(f"epoch={epoch:03d} dev_acc={accuracy:.4f}")
        if accuracy >= best:
            best = accuracy
            model.save_pretrained(args.output)
            extractor.save_pretrained(args.output)
    del optimizer, model, loss
    model = AutoModelForAudioClassification.from_pretrained(args.output, local_files_only=True).to(device)
    test_loss, test_accuracy, test_metrics = evaluate(model, loaders["test"], device, collect_scores=True)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    (args.output / "metrics.json").write_text(json.dumps({
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_dev_accuracy": best,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_metrics": test_metrics,
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
