#!/usr/bin/env python3
"""Build versioned inventory/development manifests with truthful speaker metadata."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from aasist_protocol import sha256
from validate_manifest import ROLES, manifest_rows, speaker_audit

FLEURS = "FLEURS_TR"
COMMON_VOICE = "commonvoice_27_tr_fixed"


def absolute_audio(row, manifest):
    return Path(os.path.abspath(manifest.parent / row["path"]))


def version_row(row, path, directory, common_voice):
    result = {key: value for key, value in row.items() if key != "speaker_id"}
    result.update(path=os.path.relpath(path, directory), schema_version=2)
    result.setdefault("clip_id", hashlib.sha256(row["path"].replace(chr(92), "/").encode()).hexdigest())
    result["scenario"] = "real" if row["label"] == 1 else "tts_unverified"
    result["speaker_roles"] = dict.fromkeys(ROLES)
    result["groups"] = {}
    group = row.get("speaker_id")
    if row["source"] == COMMON_VOICE and group and common_voice.get(path) == group:
        result["speaker_roles"]["source"] = {
            "id": f"commonvoice27:{group}", "kind": "person", "verified": True,
            "evidence": "matching Common Voice participant group in the local source manifest",
        }
    elif group:
        kind = {"FLEURS_TR": "example_group", "MLAAD": "producer_group",
                "girenit-tts-tr-synthetic": "producer_group"}.get(row["source"], "legacy_group")
        result["groups"][kind] = group
    return result


def create_protocol(inventory, development, common_voice_path, external, output_dir):
    external_rows = manifest_rows(external)
    held = {absolute_audio(row, external): row for row in external_rows}
    if (not held or len(held) != len(external_rows)
            or any(row.get("source") != FLEURS or row.get("label") != 1
                   or row.get("split") != "test" or not row.get("clip_id") for row in external_rows)
            or len({row["clip_id"] for row in external_rows}) != len(held)):
        raise ValueError("Expected a unique FLEURS genuine test manifest")
    held_names = {p.name for p in held}
    common_voice = {absolute_audio(row, common_voice_path): row.get("speaker_id")
                    for row in manifest_rows(common_voice_path)}
    catalog = {}
    for row in manifest_rows(inventory):
        path = absolute_audio(row, inventory)
        if path in catalog:
            raise ValueError("Duplicate inventory path")
        if path in held and (row["source"] != FLEURS or row["label"] != 1):
            raise ValueError("Conflicting FLEURS inventory provenance")
        result = version_row(row, path, output_dir, common_voice)
        result["legacy_split"] = row["split"]
        result["split"] = "inventory"
        if row["source"] == FLEURS:
            result["split"] = "external" if path in held else "excluded"
            if path in held:
                result["clip_id"] = held[path]["clip_id"]
        catalog[path] = result
    if not held.keys() <= catalog.keys():
        raise ValueError("FLEURS test contains files outside the inventory")

    dev = []
    seen_paths = set()
    seen_ids = set()
    held_ids = {row["clip_id"] for row in external_rows}
    for row in manifest_rows(development):
        if row.get("split") not in ("train", "dev"):
            continue
        path = absolute_audio(row, development)
        source = catalog.get(path)
        if (source is None or source["source"] != row["source"] or source["label"] != row["label"]
                or (row["source"] == COMMON_VOICE and source["legacy_split"] != row["split"])
                or row["source"] == FLEURS or path in held or path.name in held_names
                or row.get("clip_id") in held_ids):
            raise ValueError("Development contains a reserved or mismatched record")
        result = version_row(row, path, output_dir, common_voice)
        if path in seen_paths or result["clip_id"] in seen_ids:
            raise ValueError("Duplicate development path or clip_id")
        seen_paths.add(path)
        seen_ids.add(result["clip_id"])
        dev.append(result)
    for split in ("train", "dev"):
        if {row["label"] for row in dev if row["split"] == split} != {0, 1}:
            raise ValueError(f"{split} requires both classes")
    audit = speaker_audit(dev)
    if audit["verified_person_ids_across_splits"]:
        raise ValueError("Verified person identity crosses train/dev")
    report = {
        "schema_version": 2,
        "inputs_sha256": {str(path): sha256(path) for path in (inventory, development, common_voice_path, external)},
        "inventory_by_status": dict(Counter(row["split"] for row in catalog.values())),
        "legacy_dev_fleurs_overlap_by_path": sum(catalog[path]["legacy_split"] == "dev" for path in held),
        "external_fleurs_records": len(held),
        "development_fleurs_overlap": {"path": 0, "wav_name": 0, "clip_id": 0},
        "development_counts": {
            split: dict(Counter(row["label"] for row in dev if row["split"] == split))
            for split in ("train", "dev")
        },
        "speaker_audit_development": audit,
        "development_heldout_included": False,
    }
    return list(catalog.values()), dev, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "development", "common-voice", "external", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    inventory, dev, report = create_protocol(
        args.inventory, args.development, args.common_voice, args.external, output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = (("turkish_inventory_v2.jsonl", inventory), ("turkish_channel_development_v2.jsonl", dev))
    for name, rows in outputs:
        path = output_dir / name
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["outputs_sha256"] = {name: sha256(output_dir / name) for name, _ in outputs}
    (output_dir / "manifest_validation_v2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
