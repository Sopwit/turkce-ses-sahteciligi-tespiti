#!/usr/bin/env python3
"""Train an anti-spoofing model on the Turkish JSONL manifest."""

import argparse
import importlib
import json
import random
from itertools import islice
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "aasist-source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aasist_protocol import config_fingerprint, channel_training_rows, experiment_settings, require_clean_development, sha256, verify_initial_weights
from metrics import classification_metrics  # noqa: E402
from project_paths import PATHS, audio_path  # noqa: E402


def pad_random(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == 0:
        raise ValueError("Encountered an empty audio file")
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length]
    repeats = (length + len(audio) - 1) // len(audio)
    return np.tile(audio, repeats)[:length]


class TurkishAudioDataset(Dataset):
    def __init__(self, records: list[dict], samples: int, training: bool):
        self.records = records
        self.samples = samples
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        audio, sample_rate = sf.read(record["path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {record['path']}")
        if sample_rate != 16000:
            import scipy.signal
            audio = scipy.signal.resample_poly(audio, 16000, sample_rate).astype("float32")
        if self.training:
            audio = pad_random(audio, self.samples)
        else:
            audio = audio[:self.samples]
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)), mode="wrap")
        return torch.from_numpy(audio.copy()), torch.tensor(record["label"], dtype=torch.long)


def load_manifest(path: Path, splits=("train", "dev", "test")) -> dict[str, list[dict]]:
    result = {split: [] for split in splits}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split = record.get("split")
            if split in result:
                record["path"] = str(audio_path(record["path"], path))
                result[split].append(record)
    if any(not rows for rows in result.values()):
        raise ValueError(f"Manifest must contain non-empty requested splits: {', '.join(splits)}")
    return result


def build_model(config: dict, device: torch.device) -> nn.Module:
    module = importlib.import_module(f"models.{config['model_config']['architecture']}")
    model = module.Model(config["model_config"]).to(device)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    batches = loader if max_batches is None else islice(loader, max_batches)
    for audio, labels in batches:
        audio, labels = audio.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            _, logits = model(audio)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float, dict]:
    model.eval()
    total_loss = correct = total = 0
    labels: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for audio, batch_labels in loader:
            audio, batch_labels = audio.to(device), batch_labels.to(device)
            _, logits = model(audio)
            loss = criterion(logits, batch_labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * batch_labels.size(0)
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total += batch_labels.size(0)
            labels.extend(batch_labels.cpu().tolist())
            scores.extend(probabilities.cpu().tolist())
    return total_loss / total, correct / total, classification_metrics(labels, scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--arm", choices=("M1", "M2"),
                        help="Apply the versioned AASIST-L fine-tuning protocol")
    parser.add_argument("--channel-train", type=Path, action="append", default=[])
    parser.add_argument("--selection-metric", choices=("accuracy", "eer"), default="accuracy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    protocol = experiment_settings() if args.arm else None
    arm = protocol["arms"][args.arm] if protocol else None
    if arm:
        if args.batch_size is not None or args.epochs is not None:
            parser.error("The experiment arm fixes batch size and optimizer steps")
        if args.seed is None:
            args.seed = protocol["seeds"][0]
        if args.seed not in protocol["seeds"] or config_fingerprint(args.config) != protocol["base_config_fingerprint"]:
            parser.error("The experiment requires a protocol seed and the original AASIST-L config")
        verify_initial_weights(args.weights, protocol)
        args.selection_metric = arm["selection_metric"]
    if bool(args.channel_train) != (args.arm == "M2"):
        parser.error("Pass channel training manifests for M2 only")
    if args.seed is None:
        args.seed = 1234
    if any(value is not None and value < 1 for value in (args.batch_size, args.epochs)):
        parser.error("Batch size and epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output is None:
        architecture = config["model_config"]["architecture"]
        args.output = PATHS["runs"] / architecture
    elif not args.output.is_absolute():
        args.output = ROOT / args.output
    records = load_manifest(args.manifest, ("train", "dev"))
    if arm:
        require_clean_development(records["train"])
        require_clean_development(records["dev"])
    if args.arm == "M2":
        records["train"] = channel_training_rows(
            records["train"], args.channel_train, args.seed, arm["training_views"][1:],
        )
    if any({row["label"] for row in rows} != {0, 1} for rows in records.values()):
        parser.error("Train and dev must each contain both classes")
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"output={args.output.resolve()}")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ancak kullanılabilir bir CUDA cihazı yok")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS istendi ancak kullanılabilir bir Apple GPU yok")
    device = torch.device(device_name)
    print(f"device={device}")
    model = build_model(config, device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    batch_size = arm["batch_size"] if arm else args.batch_size or config["batch_size"]
    epochs = args.epochs or config["num_epochs"]
    samples = config["model_config"]["nb_samp"]
    loaders = {
        split: DataLoader(
            TurkishAudioDataset(items, samples, split == "train"),
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=False,
            num_workers=0,
        )
        for split, items in records.items()
    }
    counts = torch.bincount(torch.tensor([r["label"] for r in records["train"]]), minlength=2)
    weights = (counts.sum() / (2 * counts.clamp_min(1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim_config"]["base_lr"],
                                  weight_decay=config["optim_config"]["weight_decay"])
    best_accuracy = -1.0
    best_score = float("-inf")
    checkpoint_steps = 0
    history = []
    step_budget = arm["optimizer_steps"] if arm else None
    steps = 0
    batches_per_epoch = len(loaders["train"])
    if step_budget is not None:
        epochs = (step_budget + batches_per_epoch - 1) // batches_per_epoch
    for epoch in range(1, epochs + 1):
        batches = batches_per_epoch if step_budget is None else min(batches_per_epoch, step_budget - steps)
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer, max_batches=batches,
        )
        steps += batches
        if args.selection_metric == "eer":
            dev_loss, dev_accuracy, dev_metrics = evaluate(model, loaders["dev"], criterion, device)
            dev_eer = dev_metrics["eer"]
            selection_score = -dev_eer
        else:
            dev_loss, dev_accuracy = run_epoch(model, loaders["dev"], criterion, device)
            dev_eer = None
            selection_score = dev_accuracy
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
              f"dev_loss={dev_loss:.4f} dev_acc={dev_accuracy:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
            "dev_eer": dev_eer,
            "optimizer_steps": steps,
        })
        best_accuracy = max(best_accuracy, dev_accuracy)
        save_checkpoint = steps == step_budget if arm else selection_score >= best_score
        if save_checkpoint:
            best_score = selection_score
            checkpoint_steps = steps
            torch.save(model.state_dict(), args.output / "best.pth")
    (args.output / "metrics.json").write_text(json.dumps({
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "device": str(device),
        "epochs": epochs,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "batch_size": batch_size,
        "best_dev_accuracy": best_accuracy,
        "optimizer_steps": steps,
        "checkpoint_optimizer_steps": checkpoint_steps,
        "checkpoint_sha256": sha256(args.output / "best.pth"),
        "experiment": protocol["version"] if protocol else None,
        "arm": args.arm,
        "initial_checkpoint_sha256": sha256(args.weights) if args.weights else None,
        "channel_train_manifest_sha256": {str(path): sha256(path) for path in args.channel_train},
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
