#!/usr/bin/env python3
"""Check manifest splits and distinguish verified speaker identities from groups."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


ROLES = ("source", "target", "reference")
REQUIRED_ROLES = {
    "real": ("source",),
    "cloned": ("reference",),
    "converted": ("source", "target"),
}


def manifest_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def person_ids(row):
    roles = row.get("speaker_roles", {})
    required = REQUIRED_ROLES.get(row.get("scenario"))
    if required is None or not isinstance(roles, dict):
        return None
    ids = set()
    if any(roles.get(role) is None for role in required):
        return None
    for role, identity in roles.items():
        if role not in ROLES:
            return None
        if identity is None:
            continue
        if (not isinstance(identity, dict) or identity.get("kind") != "person"
                or identity.get("verified") is not True
                or not isinstance(identity.get("id"), str) or not identity["id"].strip()
                or not isinstance(identity.get("evidence"), str) or not identity["evidence"].strip()):
            return None
        ids.add(identity["id"])
    return ids


def speaker_audit(rows):
    splits = defaultdict(set)
    missing = Counter()
    for row in rows:
        ids = person_ids(row)
        if ids is None:
            missing[row["split"]] += 1
            continue
        for identity in ids:
            splits[identity].add(row["split"])
    overlaps = sum(len(partitions) > 1 for partitions in splits.values())
    return {
        "verified_person_role_coverage": not missing,
        "unverified_role_rows_by_split": dict(missing),
        "verified_person_ids_across_splits": overlaps,
        "speaker_disjoint": not missing and overlaps == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-speaker-id", action="store_true",
                        help="Check the legacy group field is present; does not certify speaker independence")
    parser.add_argument("--speaker-disjoint", action="store_true",
                        help="Require verified person identities for all scenario roles")
    parser.add_argument("--group-disjoint", action="store_true",
                        help="Check the legacy speaker_id group field across splits")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"),
                        default=("train", "dev", "test"))
    parser.add_argument("--exclude-source", action="append", default=[],
                        help="Reject rows from a reserved source")
    parser.add_argument("--max-class-ratio", type=float, default=20.0)
    args = parser.parse_args()
    rows = manifest_rows(args.manifest)
    if not rows or any(row.get("split") not in args.splits for row in rows):
        parser.error("Manifest is empty or contains an unexpected split")
    if any(row.get("source") in args.exclude_source for row in rows):
        parser.error("Manifest includes a reserved source")
    if args.require_speaker_id or args.group_disjoint:
        if any(not row.get("speaker_id") for row in rows):
            parser.error("Legacy group validation requires speaker_id on every record")
    if args.group_disjoint:
        groups = defaultdict(set)
        for row in rows:
            groups[row["speaker_id"]].add(row["split"])
        if any(len(parts) > 1 for parts in groups.values()):
            parser.error("Legacy group crosses splits")
    if args.speaker_disjoint:
        audit = speaker_audit(rows)
        if not audit["speaker_disjoint"]:
            parser.error(f"Speaker-disjoint criteria failed: {audit}")
    splits = Counter(row["split"] for row in rows)
    for split in args.splits:
        counts = Counter(row.get("label") for row in rows if row["split"] == split)
        if set(counts) != {0, 1}:
            parser.error(f"{split} requires both classes, got {dict(counts)}")
        if max(counts.values()) / min(counts.values()) > args.max_class_ratio:
            parser.error(f"{split} class imbalance exceeds limit")
    print("splits:", dict(splits))
    print("sources:", dict(Counter(row.get("source") for row in rows)))
    print("generators:", dict(Counter(row.get("generator") for row in rows)))
    print("manifest validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
