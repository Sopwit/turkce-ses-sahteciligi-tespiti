from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from streaming_audio import HOP_SAMPLES, RemoteAudioWindows, WINDOW_SAMPLES


def pcm(sample, count):
    return struct.pack("<h", sample) * count


class StreamingTest(unittest.TestCase):
    def test_chunk_boundaries_and_one_second_hop(self):
        stream = RemoteAudioWindows()
        self.assertEqual(stream.feed_pcm16le(pcm(3000, 50000), 0), [])
        windows = stream.feed_pcm16le(pcm(3000, WINDOW_SAMPLES + HOP_SAMPLES - 50000), 50000)
        self.assertEqual([(w.start_sample, w.end_sample) for w in windows],
                         [(0, WINDOW_SAMPLES), (HOP_SAMPLES, WINDOW_SAMPLES + HOP_SAMPLES)])
        self.assertEqual(len(windows[0].pcm), WINDOW_SAMPLES)
        self.assertEqual(windows[0].pcm[0], 3000 / 32768)

    def test_silence_skips_inference_and_preserves_clock(self):
        stream = RemoteAudioWindows(vad=lambda frame: any(frame))
        self.assertEqual(stream.feed_pcm16le(pcm(0, WINDOW_SAMPLES), 0), [])
        windows = stream.feed_pcm16le(pcm(3000, HOP_SAMPLES), WINDOW_SAMPLES)
        self.assertEqual(windows[0].start_sample, HOP_SAMPLES)

    def test_discontinuity_does_not_merge_segments(self):
        stream = RemoteAudioWindows()
        self.assertEqual(stream.feed_pcm16le(pcm(1000, 40000), 0), [])
        self.assertEqual(stream.feed_pcm16le(pcm(1000, 40000), 40100), [])
        windows = stream.feed_pcm16le(pcm(1000, WINDOW_SAMPLES - 40000), 80100)
        self.assertEqual(windows[0].start_sample, 40100)
        with self.assertRaises(ValueError):
            stream.feed_pcm16le(b"x", stream.expected_sample)
