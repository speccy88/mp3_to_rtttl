"""Strict RTTTL parsing plus small helpers for visualization and audition.

The converter and GUI share this module so that displayed timing, quality
metrics, and synthesized previews all interpret ringtone tokens identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

NOTE_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
NOTE_TO_PC = {name: index for index, name in enumerate(NOTE_NAMES)}


@dataclass(frozen=True)
class RtttlToken:
    """One parsed RTTTL note or rest with both wall-clock and grid duration."""

    midi: Optional[int]
    seconds: float
    steps: int


def midi_to_frequency(midi: int) -> float:
    """Convert a MIDI note number to equal-tempered frequency in hertz."""

    return 440.0 * 2.0 ** ((midi - 69) / 12)


def parse_rtttl_tokens(rtttl: str, grid: int = 16) -> list[RtttlToken]:
    """Parse a complete RTTTL string into validated, timed tokens.

    ``grid`` is the number of cells in a whole note.  A value of 16 therefore
    represents sixteenth-note cells, matching the converter's default.
    """
    if not isinstance(rtttl, str):
        raise TypeError("RTTTL input must be a string.")
    if not isinstance(grid, int) or isinstance(grid, bool) or grid <= 0:
        raise ValueError("RTTTL grid must be a positive integer.")

    parts = rtttl.strip().split(":", 2)
    if len(parts) != 3:
        raise ValueError("RTTTL must have name:defaults:notes")
    _title, defaults, melody = parts
    default_duration = 4
    default_octave = 5
    bpm = 63.0

    for item in defaults.split(","):
        if not item.strip():
            continue
        key, separator, value = item.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(f"Invalid RTTTL default: {item}")
        if key not in {"d", "o", "b"}:
            raise ValueError(f"Unsupported RTTTL default: {key}")
        try:
            if key == "d":
                default_duration = int(value)
                if default_duration <= 0:
                    raise ValueError
            elif key == "o":
                default_octave = int(value)
                if not 0 <= default_octave <= 9:
                    raise ValueError
            elif key == "b":
                bpm = float(value)
                if not math.isfinite(bpm) or bpm <= 0.0:
                    raise ValueError
        except ValueError as exc:
            raise ValueError(f"Invalid RTTTL default: {item}") from exc

    whole_note = 240.0 / bpm
    tokens: list[RtttlToken] = []

    for raw_token in melody.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue

        position = 0
        while position < len(token) and token[position].isdigit():
            position += 1
        duration = int(token[:position] or default_duration)
        if duration <= 0:
            raise ValueError(f"RTTTL duration must be positive: {raw_token}")
        if position >= len(token):
            raise ValueError(f"RTTTL token has no note: {raw_token}")

        note = token[position : position + 1]
        position += 1
        if position < len(token) and token[position] == "#":
            note += "#"
            position += 1

        dotted = False
        if position < len(token) and token[position] == ".":
            dotted = True
            position += 1

        octave = default_octave
        if position < len(token) and token[position].isdigit():
            octave = int(token[position])
            position += 1
        if not 0 <= octave <= 9:
            raise ValueError(f"RTTTL octave must be between 0 and 9: {raw_token}")
        if position < len(token) and token[position] == ".":
            dotted = True
            position += 1
        if position != len(token):
            raise ValueError(f"Unexpected RTTTL token syntax: {raw_token}")

        beat_count = 4.0 / duration
        if dotted:
            beat_count *= 1.5
        seconds = whole_note / duration
        if dotted:
            seconds *= 1.5

        if note == "p":
            midi = None
        elif note in NOTE_TO_PC:
            midi = 12 * (octave + 1) + NOTE_TO_PC[note]
        else:
            raise ValueError(f"Unsupported RTTTL note: {raw_token}")

        tokens.append(
            RtttlToken(
                midi=midi,
                seconds=seconds,
                steps=max(1, round(beat_count * grid / 4)),
            )
        )

    if not tokens:
        raise ValueError("RTTTL contains no notes")
    return tokens


def rtttl_frames(rtttl: str, grid: int = 16) -> tuple[list[Optional[int]], set[int]]:
    """Expand RTTTL into grid-aligned pitches and non-initial onset indexes."""

    frames: list[Optional[int]] = []
    onsets: set[int] = set()
    for token in parse_rtttl_tokens(rtttl, grid=grid):
        if frames:
            onsets.add(len(frames))
        frames.extend([token.midi] * token.steps)
    return frames, onsets


def synthesize_rtttl(rtttl: str, sample_rate: int = 22_050) -> np.ndarray:
    """Render RTTTL as a mono float32 square-wave preview."""
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    chunks: list[np.ndarray] = []
    for token in parse_rtttl_tokens(rtttl):
        length = max(1, int(token.seconds * sample_rate))
        if token.midi is None:
            chunks.append(np.zeros(length, dtype=np.float32))
            continue

        t = np.linspace(0.0, token.seconds, length, endpoint=False)
        envelope = np.ones(length, dtype=np.float32)
        attack = min(length, max(1, int(sample_rate * 0.006)))
        release = min(length, max(1, int(sample_rate * 0.018)))
        envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        wave = np.sign(np.sin(2 * math.pi * midi_to_frequency(token.midi) * t)).astype(np.float32)
        chunks.append(0.45 * wave * envelope)

    return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
