import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import wave

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from create_real_codec_manifest import CODEC_CONFIG, codec_arguments


class CodecTest(unittest.TestCase):
    def test_explicit_opus_settings(self):
        config = json.loads(CODEC_CONFIG.read_text())
        options = codec_arguments(config["encoders"]["opus"])
        values = dict(zip(options[::2], options[1::2]))
        self.assertEqual({key: values[key] for key in ("-b:a", "-vbr", "-application", "-ar", "-ac")},
                         {"-b:a": "12000", "-vbr": "on", "-application": "audio", "-ar": "8000", "-ac": "1"})

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_short_roundtrip_and_no_overwrite(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "input.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setparams((1, 2, 16000, 0, "NONE", ""))
                handle.writeframes(struct.pack("<hh", 600, -600) * 800)
            manifest = base / "input.jsonl"
            manifest.write_text(json.dumps({"path": "input.wav", "label": 1, "split": "train"}) + "\n")
            for channel in ("g711_mulaw", "opus"):
                with self.subTest(channel=channel):
                    directory = base / channel
                    output = base / f"{channel}.jsonl"
                    command = [sys.executable, str(SCRIPTS / "create_real_codec_manifest.py"),
                               "--input-manifest", str(manifest), "--output-manifest", str(output),
                               "--output-dir", str(directory), "--codec", channel]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    log = json.loads(output.with_suffix(".codec-log.json").read_text())
                    self.assertTrue(log["ffmpeg_version"].startswith("ffmpeg version"))
                    self.assertEqual(log["records"], 1)
                    self.assertEqual(log["protocol"], "telephone-codecs-v2")
                    with wave.open(str(directory / "00000000.wav"), "rb") as handle:
                        self.assertEqual((handle.getframerate(), handle.getnchannels()), (8000, 1))
                    self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
