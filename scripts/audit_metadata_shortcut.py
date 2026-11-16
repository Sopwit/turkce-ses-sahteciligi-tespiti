#!/usr/bin/env python3
"""Small retrospective metadata controls for the previously scored P0 cohort."""

import argparse
import json
from collections import Counter
from pathlib import Path

from aasist_protocol import sha256
from project_paths import audio_path
from validate_manifest import manifest_rows


def samples(path, allowed):
    import soundfile as sf

    rows = [row for row in manifest_rows(path) if row["split"] in allowed]
    features = []
    for row in rows:
        stream = sf.info(str(audio_path(row["path"], path)))
        features.append({
            "source": row["source"], "duration_s": stream.duration,
            "sample_rate": stream.samplerate, "subtype": stream.subtype, "channels": stream.channels,
        })
    return rows, features


def evaluate(train, tests):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    train_rows, train_x = train
    suites = {
        "source_only": ("source",),
        "duration_only": ("duration_s",),
        "format_only": ("sample_rate", "subtype", "channels"),
        "acoustic_metadata": ("duration_s", "sample_rate", "subtype", "channels"),
    }
    results = {name: {} for name in tests}
    for suite, fields in suites.items():
        select = lambda items: [{field: item[field] for field in fields} for item in items]
        model = make_pipeline(DictVectorizer(sparse=False), DecisionTreeClassifier(max_depth=3, random_state=2209))
        model.fit(select(train_x), [row["label"] for row in train_rows])
        for name, (records, features) in tests.items():
            labels = [row["label"] for row in records]
            if set(labels) != {0, 1}:
                raise ValueError(f"{name} requires both classes")
            predictions = model.predict(select(features))
            results[name][suite] = float(balanced_accuracy_score(labels, predictions))
    return results


def describe(records, features):
    counts = Counter((row["source"], row["label"], f["sample_rate"], f["subtype"], f["channels"])
                     for row, f in zip(records, features, strict=True))
    return [{"source": source, "label": label, "sample_rate": hz, "subtype": subtype,
             "channels": channels, "count": n} for (source, label, hz, subtype, channels), n in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p0-manifest", "p0-g711", "p0-opus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    train = samples(args.p0_manifest, {"train"})
    tests = {
        "clean": samples(args.p0_manifest, {"test"}),
        "g711_mulaw": samples(args.p0_g711, {"test"}),
        "opus": samples(args.p0_opus, {"test"}),
    }
    def cohort(rows):
        return {(row["clip_id"], row["label"]) for row in rows}
    train_ids = {row["clip_id"] for row in train[0]}
    if (any(len(cohort(rows)) != len(rows) or cohort(rows) != cohort(tests["clean"][0])
            for rows, _ in tests.values())
            or train_ids & {row["clip_id"] for row in tests["clean"][0]}):
        parser.error("P0 channel cohorts must be paired and disjoint from train")
    result = {
        "protocol": "metadata-shortcut-v1",
        "scope": "retrospective P0 only",
        "train_count": len(train[0]),
        "test_counts": {name: len(rows) for name, (rows, _) in tests.items()},
        "inputs_sha256": {p.name: sha256(p) for p in (args.p0_manifest, args.p0_g711, args.p0_opus)},
        "balanced_accuracy": evaluate(train, tests),
        "metadata_by_condition": {name: describe(*cohort) for name, cohort in {"train": train, **tests}.items()},
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
