"""Resolve paths for the RedSprite workspace or a standalone checkout."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEPARATED = ROOT.parent.name == "01-kod"
BASE = ROOT.parent.parent if SEPARATED else ROOT
_LAYOUT = json.loads((ROOT / "project-layout.json").read_text(encoding="utf-8"))
PATHS = {
    name: BASE / directory
    for name, directory in _LAYOUT["workspace" if SEPARATED else "standalone"].items()
}


def audio_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    relative_to_manifest = manifest.parent / path
    if relative_to_manifest.is_file() or not path.is_file():
        return relative_to_manifest
    return path


def stored_audio_path(path: Path, manifest: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), manifest.parent.resolve())).as_posix()
    except ValueError:
        # Windows cannot express a relative path between two drives.
        return str(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", choices=PATHS)
    print(PATHS[parser.parse_args().path])
