"""
Generates small synthetic WAV files under data/sample_audio/ so the
repository is immediately runnable end-to-end without requiring the
user to source real recordings first.

These are SYNTHETIC placeholder tones (sine/noise mixes), not real
environmental recordings — they exist purely so `make demo` / the
Streamlit "Upload" tab has something to try immediately. Users are
expected to replace them with real recordings for meaningful YAMNet
classification results. Uses only the Python standard library (`wave`,
`struct`, `random`) so it runs even before installing project dependencies.
"""
from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_audio"
SAMPLE_RATE = 16000


def _write_wav(path: Path, samples: list[float]) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        frames = b"".join(struct.pack("<h", max(-32767, min(32767, int(s * 32767)))) for s in samples)
        wf.writeframes(frames)


def tone(freq: float, duration: float, amplitude: float = 0.3) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    return [amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n)]


def noise(duration: float, amplitude: float = 0.3) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    return [amplitude * (random.random() * 2 - 1) for i in range(n)]


def mix(*signals: list[float]) -> list[float]:
    n = min(len(s) for s in signals)
    return [sum(s[i] for s in signals) / len(signals) for i in range(n)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_low_hum.wav": tone(80, 3.0, amplitude=0.15),
        "sample_mid_tone.wav": tone(440, 2.5, amplitude=0.25),
        "sample_broadband_noise.wav": noise(3.0, amplitude=0.35),
        "sample_mixed_tone_noise.wav": mix(tone(150, 3.0, amplitude=0.3), noise(3.0, amplitude=0.2)),
        "sample_quiet_ambient.wav": noise(3.0, amplitude=0.02),
    }

    for filename, sig in samples.items():
        _write_wav(OUT_DIR / filename, sig)
        print(f"Generated {OUT_DIR / filename} ({len(sig) / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
