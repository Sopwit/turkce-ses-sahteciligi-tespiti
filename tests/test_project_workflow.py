"""Run the portable CLI workflow with synthetic paths and no model downloads."""

from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectWorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.repo = self.workspace / "01-kod" / "detector"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(ROOT / "project-layout.json", self.repo / "project-layout.json")
        for name in ("project_paths", "run_turkish_pipeline", "create_turkish_manifest"):
            shutil.copyfile(ROOT / "scripts" / f"{name}.py", scripts / f"{name}.py")

    def run_cli(self, script, *args):
        return subprocess.run(
            [sys.executable, str(self.repo / "scripts" / f"{script}.py"), *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_workspace_and_standalone_paths(self):
        paths = {
            "data": "02-veri/islenmis",
            "manifests": "02-veri/manifestler",
            "weights": "03-deneyler/model-agirliklari",
            "runs": "03-deneyler/kosular",
        }
        for key, directory in paths.items():
            result = self.run_cli("project_paths", key)
            self.assertEqual(Path(result.stdout.strip()), self.workspace / directory)
        standalone = self.workspace / "standalone"
        self.repo.rename(standalone)
        self.repo = standalone
        result = self.run_cli("project_paths", "data")
        self.assertEqual(Path(result.stdout.strip()), standalone / "dataset")

    def test_pipeline_preserves_explicit_input_manifest(self):
        path = self.workspace / "input.jsonl"
        contents = b'{"path":"sample.wav","split":"train","label":1}\n'
        path.write_bytes(contents)
        self.run_cli("run_turkish_pipeline", "--manifest", str(path), "--prepare-only")
        self.assertEqual(path.read_bytes(), contents)
        self.assertFalse((self.workspace / "03-deneyler").exists())

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "requires a maintained Python tarfile")
    def test_manifest_generation_extracts_source_and_skips_channel_outputs(self):
        data = self.workspace / "02-veri" / "islenmis"
        for source in ("commonvoice_27_tr_fixed", "MLAAD"):
            for split in ("train", "dev", "test"):
                parent = data / source / split
                parent.mkdir(parents=True)
                (parent / "sample.wav").touch()
        derived = data / "kanal-varyantlari" / "test"
        derived.mkdir(parents=True)
        (derived / "derived.wav").touch()
        # This fixture only exercises extraction and paths. Audio is read in a
        # separate model run, so there is no need for actual WAV data here.
        with tarfile.open(data / "MLAAD" / "training.tar.gz", "w:gz") as archive:
            contents = b"sample"
            member = tarfile.TarInfo("train/archived.wav")
            member.size = len(contents)
            archive.addfile(member, BytesIO(contents))
        self.run_cli("create_turkish_manifest", "--extract-archives")
        output = self.workspace / "02-veri" / "manifestler" / "turkish_manifest.jsonl"
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 7)
        self.assertTrue(all((output.parent / row["path"]).is_file() for row in rows))
        self.assertTrue(all(row["path"].startswith("../islenmis/") for row in rows))
        self.assertNotIn("derived.wav", {Path(row["path"]).name for row in rows})


if __name__ == "__main__":
    unittest.main()
