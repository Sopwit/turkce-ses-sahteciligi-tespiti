"""Window a mono 16 kHz remote PCM stream and score it with PyTorch."""

from array import array
from dataclasses import dataclass
from math import isfinite
import sys
from typing import Callable


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_600
HOP_SAMPLES = 16_000
FRAME_SAMPLES = 320


def voiced_energy(frame):
    """Simple energy gate baseline, replaceable by a validated VAD callback."""
    return sum(sample * sample for sample in frame) / len(frame) >= 1e-5


@dataclass(frozen=True)
class AudioWindow:
    start_sample: int
    end_sample: int
    pcm: tuple[float, ...]


class RemoteAudioWindows:
    """Feed only the consented far-end track, decoded and resampled upstream."""

    def __init__(self, vad: Callable = voiced_energy, minimum_voiced=0.2):
        if not 0 < minimum_voiced <= 1:
            raise ValueError("minimum_voiced must be in (0, 1]")
        self.vad = vad
        self.minimum_voiced = minimum_voiced
        self.reset()

    def reset(self, start_sample=0):
        self.buffer = []
        self.start_sample = start_sample
        self.expected_sample = start_sample

    def feed_pcm16le(self, pcm, first_sample):
        """A gap resets the window; it never joins unrelated audio segments."""
        if len(pcm) % 2 or first_sample < 0:
            raise ValueError("Expected aligned signed 16-bit samples and a nonnegative offset")
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if first_sample != self.expected_sample:
            self.reset(first_sample)
        self.expected_sample += len(samples)
        self.buffer.extend(value / 32768.0 for value in samples)
        windows = []
        while len(self.buffer) >= WINDOW_SAMPLES:
            end = self.start_sample + WINDOW_SAMPLES
            candidate = self.buffer[:WINDOW_SAMPLES]
            voiced = 0
            for i in range(0, WINDOW_SAMPLES, FRAME_SAMPLES):
                frame = candidate[i:i + FRAME_SAMPLES]
                if self.vad(frame):
                    voiced += len(frame)
            if voiced >= self.minimum_voiced * WINDOW_SAMPLES:
                windows.append(AudioWindow(self.start_sample, end, tuple(candidate)))
            del self.buffer[:HOP_SAMPLES]
            self.start_sample += HOP_SAMPLES
        return windows


class TorchWindowScorer:
    """AASIST-L PyTorch scoring with an already-selected genuine-score threshold."""

    def __init__(self, model, device, threshold):
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Expected a frozen genuine-score threshold in [0, 1]")
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold

    def score(self, window):
        import torch

        if len(window.pcm) != WINDOW_SAMPLES:
            raise ValueError("AASIST-L requires exactly 64600 samples")
        with torch.inference_mode():
            audio = torch.tensor(window.pcm, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, logits = self.model(audio)
            genuine = torch.softmax(logits, dim=1)[0, 1].item()
        if not isfinite(genuine):
            raise ValueError("AASIST-L produced a non-finite genuine score")
        return {"start_sample": window.start_sample, "end_sample": window.end_sample,
                "genuine_score": genuine, "flagged": genuine < self.threshold}
