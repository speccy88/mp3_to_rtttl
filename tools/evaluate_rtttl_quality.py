#!/usr/bin/env python3
"""Run hermetic synthetic RTTTL checks and optional user-audio reports.

The default invocation creates all audio fixtures in a temporary directory, so
it does not depend on repository media or files in a user's home directory.
Real-audio metrics and audition exports are opt-in through repeatable
``--real-excerpt NAME PATH START DURATION`` arguments.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mp3_to_rtttl  # noqa: E402
from rtttl_utils import parse_rtttl_tokens, rtttl_frames, synthesize_rtttl  # noqa: E402

SAMPLE_RATE = 22_050


@dataclass(frozen=True)
class Fixture:
    """Definition of a generated melody used by the synthetic quality gate."""

    name: str
    bpm: int
    notes: tuple[tuple[int | None, float], ...]
    accompaniment: bool = False


@dataclass(frozen=True)
class Setting:
    """Converter parameters compared by the regression report."""

    name: str
    min_confidence: float
    onset_threshold: float
    frame_threshold: float
    minimum_note_ms: float
    top_voice_bias: float
    jump_penalty: float
    onset_change_bonus: float
    onset_change_relief: float
    onset_same_pitch_bonus: float
    sustain_bonus: float
    repeated_note_min_steps: int
    strict_rtttl_bpm: bool
    octave_smoothing: bool
    max_melody_span: int
    decoder_mode: str


@dataclass(frozen=True)
class RealExcerpt:
    """A caller-supplied segment of an audio file used for optional reporting."""

    name: str
    path: Path
    start: float
    duration: float


@dataclass(frozen=True)
class RealMetrics:
    """Listening-oriented measurements for one generated RTTTL melody."""

    chroma_score: float
    token_count: int
    span: int
    big_jumps: int
    average_jump: float


FIXTURES = (
    Fixture(
        "repeated_intervals",
        120,
        (
            (60, 1),
            (60, 1),
            (67, 1),
            (67, 1),
            (69, 1),
            (69, 1),
            (67, 2),
            (65, 1),
            (65, 1),
            (64, 1),
            (64, 1),
            (62, 1),
            (62, 1),
            (60, 2),
        ),
        accompaniment=True,
    ),
    Fixture(
        "scale",
        100,
        (
            (60, 0.5),
            (62, 0.5),
            (64, 0.5),
            (65, 0.5),
            (67, 0.5),
            (69, 0.5),
            (71, 0.5),
            (72, 1),
            (71, 0.5),
            (69, 0.5),
            (67, 0.5),
            (65, 0.5),
            (64, 0.5),
            (62, 0.5),
            (60, 1),
        ),
    ),
    Fixture(
        "leaps",
        132,
        (
            (64, 1),
            (76, 1),
            (74, 0.5),
            (72, 0.5),
            (71, 1),
            (67, 1),
            (69, 0.5),
            (71, 0.5),
            (72, 2),
            (72, 0.5),
            (72, 0.5),
            (74, 1),
            (76, 2),
        ),
        accompaniment=True,
    ),
    Fixture(
        "short_notes_and_rests",
        90,
        (
            (None, 0.5),
            (60, 1),
            (62, 0.25),
            (64, 0.25),
            (65, 0.5),
            (None, 0.25),
            (65, 0.5),
            (67, 0.25),
            (69, 0.5),
            (67, 1),
            (None, 0.5),
        ),
    ),
    Fixture(
        "short_passing_notes",
        90,
        (
            (60, 1),
            (62, 0.25),
            (60, 1),
            (64, 0.25),
            (60, 0.5),
            (65, 0.25),
            (64, 0.5),
            (62, 0.25),
            (60, 1),
        ),
    ),
    Fixture(
        "gated_repeated_notes",
        90,
        (
            (60, 1),
            (None, 0.25),
            (60, 1),
            (None, 0.25),
            (62, 0.5),
            (None, 0.25),
            (62, 0.5),
            (None, 0.25),
            (64, 1),
        ),
    ),
)


SETTINGS = (
    Setting(
        "legacy",
        min_confidence=0.25,
        onset_threshold=0.55,
        frame_threshold=0.30,
        minimum_note_ms=80.0,
        top_voice_bias=0.01,
        jump_penalty=0.035,
        onset_change_bonus=0.0,
        onset_change_relief=0.0,
        onset_same_pitch_bonus=0.40,
        sustain_bonus=0.40,
        repeated_note_min_steps=1,
        strict_rtttl_bpm=True,
        octave_smoothing=False,
        max_melody_span=mp3_to_rtttl.DEFAULT_MAX_MELODY_SPAN,
        decoder_mode="events",
    ),
    Setting(
        "current",
        min_confidence=mp3_to_rtttl.DEFAULT_MIN_CONFIDENCE,
        onset_threshold=mp3_to_rtttl.DEFAULT_ONSET_THRESHOLD,
        frame_threshold=mp3_to_rtttl.DEFAULT_FRAME_THRESHOLD,
        minimum_note_ms=mp3_to_rtttl.DEFAULT_MINIMUM_NOTE_MS,
        top_voice_bias=mp3_to_rtttl.DEFAULT_TOP_VOICE_BIAS,
        jump_penalty=mp3_to_rtttl.DEFAULT_JUMP_PENALTY,
        onset_change_bonus=mp3_to_rtttl.DEFAULT_ONSET_CHANGE_BONUS,
        onset_change_relief=mp3_to_rtttl.DEFAULT_ONSET_CHANGE_RELIEF,
        onset_same_pitch_bonus=mp3_to_rtttl.DEFAULT_ONSET_SAME_PITCH_BONUS,
        sustain_bonus=mp3_to_rtttl.DEFAULT_SUSTAIN_BONUS,
        repeated_note_min_steps=mp3_to_rtttl.DEFAULT_REPEATED_NOTE_MIN_STEPS,
        strict_rtttl_bpm=False,
        octave_smoothing=mp3_to_rtttl.DEFAULT_OCTAVE_SMOOTHING,
        max_melody_span=mp3_to_rtttl.DEFAULT_MAX_MELODY_SPAN,
        decoder_mode="events",
    ),
)


def midi_to_hz(midi: int) -> float:
    """Return the equal-tempered frequency for a MIDI note number."""

    return 440.0 * 2.0 ** ((midi - 69) / 12)


def synthesize_fixture(fixture: Fixture, path: Path) -> None:
    """Render one deterministic synthetic fixture to ``path`` as a WAV file."""

    chunks: list[np.ndarray] = []
    seconds_per_beat = 60.0 / fixture.bpm

    for index, (midi, beats) in enumerate(fixture.notes):
        length = max(1, int(beats * seconds_per_beat * SAMPLE_RATE))
        t = np.arange(length) / SAMPLE_RATE
        if midi is None:
            chunks.append(np.zeros(length, dtype=np.float32))
            continue

        frequency = midi_to_hz(midi)
        audio = (
            0.52 * np.sin(2 * np.pi * frequency * t)
            + 0.18 * np.sin(2 * np.pi * 2 * frequency * t)
            + 0.07 * np.sin(2 * np.pi * 3 * frequency * t)
        )
        if fixture.accompaniment:
            accompaniment_midi = midi - 24 if index % 2 == 0 else midi - 19
            audio += 0.11 * np.sin(2 * np.pi * midi_to_hz(accompaniment_midi) * t)

        envelope = np.ones(length)
        attack = min(length, int(0.015 * SAMPLE_RATE))
        release = min(length, int(0.055 * SAMPLE_RATE))
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        chunks.append((audio * envelope).astype(np.float32))

    output = np.concatenate(chunks)
    output = output / (np.max(np.abs(output)) or 1.0) * 0.8
    sf.write(path, output.astype(np.float32), SAMPLE_RATE)


def expected_frames(fixture: Fixture, grid: int) -> tuple[list[int | None], set[int]]:
    """Expand a fixture into expected pitch frames and onset positions."""

    frames: list[int | None] = []
    onsets: set[int] = set()
    cursor = 0
    for midi, beats in fixture.notes:
        if cursor:
            onsets.add(cursor)
        frame_count = max(1, round(beats * grid / 4))
        frames.extend([midi] * frame_count)
        cursor += frame_count
    return frames, onsets


def pitch_class_accuracy(expected: list[int | None], actual: list[int | None]) -> float:
    """Measure frame-aligned pitch-class agreement, including rests."""

    length = max(len(expected), len(actual))
    if not length:
        return 1.0
    matches = 0
    for index in range(length):
        expected_value = (
            expected[index] % 12
            if index < len(expected) and expected[index] is not None
            else expected[index]
            if index < len(expected)
            else object()
        )
        actual_value = (
            actual[index] % 12
            if index < len(actual) and actual[index] is not None
            else actual[index]
            if index < len(actual)
            else object()
        )
        if expected_value == actual_value:
            matches += 1
    return matches / length


def octave_adjusted_midi_accuracy(expected: list[int | None], actual: list[int | None]) -> float:
    """Return the best exact-frame score after whole-octave transposition."""

    length = max(len(expected), len(actual))
    if not length:
        return 1.0

    best = 0
    for shift in range(-48, 49, 12):
        matches = 0
        for index in range(length):
            expected_value = expected[index] if index < len(expected) else object()
            actual_value = (
                actual[index] + shift
                if index < len(actual) and actual[index] is not None
                else actual[index]
                if index < len(actual)
                else object()
            )
            if expected_value == actual_value:
                matches += 1
        best = max(best, matches)
    return best / length


def onset_f1(expected: set[int], actual: set[int]) -> float:
    """Compute onset F1 while allowing a one-grid-step timing tolerance."""

    if not expected and not actual:
        return 1.0
    matched_actual: set[int] = set()
    true_positives = 0
    for expected_onset in sorted(expected):
        for actual_onset in (expected_onset - 1, expected_onset, expected_onset + 1):
            if actual_onset in actual and actual_onset not in matched_actual:
                matched_actual.add(actual_onset)
                true_positives += 1
                break
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall)


def convert_fixture(path: Path, fixture: Fixture, setting: Setting) -> str:
    """Convert a generated fixture without leaking converter progress output."""

    args = argparse.Namespace(
        input=str(path),
        output="quality.rtttl",
        title=fixture.name,
        start=0.0,
        duration=30.0,
        bpm=fixture.bpm,
        grid=mp3_to_rtttl.DEFAULT_GRID,
        min_confidence=setting.min_confidence,
        onset_threshold=setting.onset_threshold,
        frame_threshold=setting.frame_threshold,
        minimum_note_ms=setting.minimum_note_ms,
        top_voice_bias=setting.top_voice_bias,
        jump_penalty=setting.jump_penalty,
        onset_change_bonus=setting.onset_change_bonus,
        onset_change_relief=setting.onset_change_relief,
        onset_same_pitch_bonus=setting.onset_same_pitch_bonus,
        sustain_bonus=setting.sustain_bonus,
        repeated_note_min_steps=setting.repeated_note_min_steps,
        strict_rtttl_bpm=setting.strict_rtttl_bpm,
        octave_smoothing=setting.octave_smoothing,
        max_melody_span=setting.max_melody_span,
        decoder_mode=setting.decoder_mode,
        midi=None,
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        return mp3_to_rtttl.convert(args)


def convert_real_excerpt(excerpt: RealExcerpt, setting: Setting) -> str:
    """Convert one caller-supplied excerpt and add context to failures."""

    args = argparse.Namespace(
        input=str(excerpt.path),
        output="quality.rtttl",
        title=excerpt.name,
        start=excerpt.start,
        duration=excerpt.duration,
        bpm=None,
        grid=mp3_to_rtttl.DEFAULT_GRID,
        min_confidence=setting.min_confidence,
        onset_threshold=setting.onset_threshold,
        frame_threshold=setting.frame_threshold,
        minimum_note_ms=setting.minimum_note_ms,
        top_voice_bias=setting.top_voice_bias,
        jump_penalty=setting.jump_penalty,
        onset_change_bonus=setting.onset_change_bonus,
        onset_change_relief=setting.onset_change_relief,
        onset_same_pitch_bonus=setting.onset_same_pitch_bonus,
        sustain_bonus=setting.sustain_bonus,
        repeated_note_min_steps=setting.repeated_note_min_steps,
        strict_rtttl_bpm=setting.strict_rtttl_bpm,
        octave_smoothing=setting.octave_smoothing,
        max_melody_span=setting.max_melody_span,
        decoder_mode=setting.decoder_mode,
        midi=None,
    )
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            return mp3_to_rtttl.convert(args)
    except Exception as exc:
        raise RuntimeError(
            f"real excerpt {excerpt.name!r} could not be converted from {excerpt.path}: {exc}"
        ) from exc


def load_excerpt_audio(excerpt: RealExcerpt) -> tuple[np.ndarray, int]:
    """Load a validated real excerpt and reject empty or unreadable segments."""

    librosa, _sf, _predict = mp3_to_rtttl.load_audio_dependencies()
    try:
        samples, sample_rate = librosa.load(
            excerpt.path,
            sr=SAMPLE_RATE,
            mono=True,
            offset=excerpt.start,
            duration=excerpt.duration,
        )
    except Exception as exc:
        raise RuntimeError(
            f"real excerpt {excerpt.name!r} could not be read from {excerpt.path}: {exc}"
        ) from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"real excerpt {excerpt.name!r} is empty; check START and DURATION")
    return samples, int(sample_rate)


def excerpt_chroma(excerpt: RealExcerpt):
    """Calculate normalized harmonic chroma for one real-audio excerpt."""

    librosa, _sf, _predict = mp3_to_rtttl.load_audio_dependencies()
    y, sr = load_excerpt_audio(excerpt)
    harmonic, _percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr, hop_length=512)
    if chroma.ndim != 2 or chroma.shape[1] == 0:
        raise ValueError(f"real excerpt {excerpt.name!r} produced no chroma frames")
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
    chroma = chroma / np.maximum(np.max(chroma, axis=0, keepdims=True), 1e-9)
    return chroma, times


def real_metrics(
    rtttl: str, chroma: np.ndarray, times: np.ndarray, excerpt: RealExcerpt
) -> RealMetrics:
    """Compare generated RTTTL pitch classes with the excerpt's chroma energy."""

    if chroma.ndim != 2 or chroma.shape[0] != 12 or not chroma.shape[1] or not times.size:
        raise ValueError(f"real excerpt {excerpt.name!r} has invalid chroma data")

    tokens = parse_rtttl_tokens(rtttl, grid=mp3_to_rtttl.DEFAULT_GRID)
    frames: list[int | None] = []
    for token in tokens:
        frames.extend([token.midi] * token.steps)

    if not frames:
        return RealMetrics(0.0, 0, 0, 0, 0.0)

    scores: list[float] = []
    for index, midi in enumerate(frames):
        if midi is None:
            continue
        time = excerpt.duration * (index + 0.5) / len(frames)
        chroma_index = int(np.argmin(np.abs(times - time)))
        scores.append(float(chroma[midi % 12, chroma_index]))

    midi_tokens = [token.midi for token in tokens if token.midi is not None]
    jumps = [
        abs(current - previous)
        for previous, current in zip(midi_tokens, midi_tokens[1:], strict=False)
    ]
    span = max(midi_tokens) - min(midi_tokens) if midi_tokens else 0
    average_jump = float(np.mean(jumps)) if jumps else 0.0
    big_jumps = sum(jump >= 12 for jump in jumps)

    return RealMetrics(
        chroma_score=float(np.mean(scores)) if scores else 0.0,
        token_count=len(tokens),
        span=span,
        big_jumps=big_jumps,
        average_jump=average_jump,
    )


def safe_name(value: str) -> str:
    """Return a portable filename stem for a caller-provided excerpt name."""

    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "excerpt"


def score_threshold(value: str) -> float:
    """Parse a finite score threshold in the inclusive range from zero to one."""

    try:
        score = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return score


def parse_real_excerpts(
    raw_specs: list[list[str]], parser: argparse.ArgumentParser
) -> tuple[RealExcerpt, ...]:
    """Validate repeatable ``--real-excerpt`` values and resolve their paths."""

    excerpts: list[RealExcerpt] = []
    seen_names: set[str] = set()
    seen_stems: set[str] = set()

    for position, (raw_name, raw_path, raw_start, raw_duration) in enumerate(raw_specs, start=1):
        option = f"--real-excerpt occurrence {position}"
        name = raw_name.strip()
        if not name:
            parser.error(f"{option}: NAME must not be empty")
        if any(ord(character) < 32 for character in name):
            parser.error(f"{option}: NAME must not contain control characters")

        name_key = name.casefold()
        stem_key = safe_name(name).casefold()
        if name_key in seen_names:
            parser.error(f"{option}: duplicate NAME {name!r}")
        if stem_key in seen_stems:
            parser.error(f"{option}: NAME {name!r} collides with another audition filename")

        try:
            start = float(raw_start)
        except ValueError:
            parser.error(f"{option}: START must be a number of seconds")
        try:
            duration = float(raw_duration)
        except ValueError:
            parser.error(f"{option}: DURATION must be a number of seconds")
        if not np.isfinite(start) or start < 0.0:
            parser.error(f"{option}: START must be a finite number at least 0")
        if not np.isfinite(duration) or duration <= 0.0:
            parser.error(f"{option}: DURATION must be a finite number greater than 0")

        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            parser.error(f"{option}: PATH {raw_path!r} cannot be resolved: {exc}")
        if not path.is_file():
            parser.error(f"{option}: PATH is not a regular file: {path}")
        try:
            if path.stat().st_size == 0:
                parser.error(f"{option}: PATH is empty: {path}")
        except OSError as exc:
            parser.error(f"{option}: PATH cannot be inspected: {exc}")

        seen_names.add(name_key)
        seen_stems.add(stem_key)
        excerpts.append(RealExcerpt(name=name, path=path, start=start, duration=duration))

    return tuple(excerpts)


def write_audition_pack(output_dir: Path, excerpts: tuple[RealExcerpt, ...]) -> list[Path]:
    """Write source, RTTTL, and synthesized WAV files for supplied excerpts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for excerpt in excerpts:
        y, _sr = load_excerpt_audio(excerpt)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * y / peak

        stem = safe_name(excerpt.name)
        source_path = output_dir / f"{stem}_source.wav"
        sf.write(source_path, y, SAMPLE_RATE)
        written.append(source_path)

        for setting in SETTINGS:
            rtttl = convert_real_excerpt(excerpt, setting)
            text_path = output_dir / f"{stem}_{setting.name}.rtttl"
            wav_path = output_dir / f"{stem}_{setting.name}_rtttl.wav"
            text_path.write_text(rtttl + "\n", encoding="utf-8")
            sf.write(wav_path, synthesize_rtttl(rtttl), SAMPLE_RATE)
            written.extend([text_path, wav_path])

    return written


def evaluate_setting(paths: dict[str, Path], setting: Setting) -> tuple[float, list[str]]:
    """Score one converter setting across every generated fixture."""

    scores: list[float] = []
    rows: list[str] = []
    for fixture in FIXTURES:
        try:
            rtttl = convert_fixture(paths[fixture.name], fixture, setting)
            expected, expected_onsets = expected_frames(fixture, mp3_to_rtttl.DEFAULT_GRID)
            actual, actual_onsets = rtttl_frames(rtttl, grid=mp3_to_rtttl.DEFAULT_GRID)
            pitch = pitch_class_accuracy(expected, actual)
            octave = octave_adjusted_midi_accuracy(expected, actual)
            rhythm = onset_f1(expected_onsets, actual_onsets)
            score = 0.55 * pitch + 0.25 * octave + 0.20 * rhythm
            token_count = rtttl.rsplit(":", 1)[-1].count(",") + 1
            scores.append(score)
            rows.append(
                f"{fixture.name}: score={score:.3f} pitch={pitch:.3f} "
                f"octave={octave:.3f} onset={rhythm:.3f} tokens={token_count}"
            )
        except Exception as exc:
            scores.append(0.0)
            rows.append(f"{fixture.name}: error={exc}")
    return sum(scores) / len(scores), rows


def evaluate_real_settings(excerpts: tuple[RealExcerpt, ...]) -> list[str]:
    """Create comparison metrics for each setting and supplied excerpt."""

    rows: list[str] = []
    chroma_cache = {excerpt.name: excerpt_chroma(excerpt) for excerpt in excerpts}

    for setting in SETTINGS:
        scores: list[float] = []
        spans: list[int] = []
        big_jumps: list[int] = []
        token_counts: list[int] = []
        details: list[str] = []

        for excerpt in excerpts:
            rtttl = convert_real_excerpt(excerpt, setting)
            chroma, times = chroma_cache[excerpt.name]
            metrics = real_metrics(rtttl, chroma, times, excerpt)
            scores.append(metrics.chroma_score)
            spans.append(metrics.span)
            big_jumps.append(metrics.big_jumps)
            token_counts.append(metrics.token_count)
            details.append(
                f"{excerpt.name}: chroma={metrics.chroma_score:.3f} "
                f"tokens={metrics.token_count} span={metrics.span} "
                f"big_jumps={metrics.big_jumps} avg_jump={metrics.average_jump:.2f}"
            )

        rows.append(
            f"{setting.name}: chroma={np.mean(scores):.3f} "
            f"tokens={np.mean(token_counts):.1f} span={np.mean(spans):.1f} "
            f"big_jumps={sum(big_jumps)}"
        )
        rows.extend(f"  {detail}" for detail in details)

    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for synthetic and opt-in real checks."""

    parser = argparse.ArgumentParser(
        description=(
            "Gate RTTTL conversion defaults with generated audio and optionally "
            "report metrics for user-supplied excerpts."
        ),
        epilog=(
            "Example:\n"
            "  evaluate_rtttl_quality.py --real-excerpt hook /path/to/audio.mp3 12 6\n"
            "Repeat --real-excerpt to evaluate more than one segment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-current-score",
        type=score_threshold,
        default=0.95,
        metavar="SCORE",
        help="Minimum synthetic score required for current settings (default: 0.95).",
    )
    parser.add_argument(
        "--real-excerpt",
        action="append",
        default=[],
        nargs=4,
        metavar=("NAME", "PATH", "START", "DURATION"),
        help=(
            "Evaluate a named audio segment; START and DURATION are seconds. "
            "The option may be repeated."
        ),
    )
    parser.add_argument(
        "--write-audition-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Write source, legacy/current RTTTL, and synthesized WAV files for "
            "the supplied real excerpts."
        ),
    )
    return parser


def main() -> int:
    """Run the synthetic gate followed by any explicitly requested reports."""

    parser = build_parser()
    args = parser.parse_args()
    excerpts = parse_real_excerpts(args.real_excerpt, parser)

    audition_dir: Path | None = args.write_audition_dir
    if audition_dir is not None:
        if not excerpts:
            parser.error("--write-audition-dir requires at least one --real-excerpt")
        audition_dir = audition_dir.expanduser().resolve()
        if audition_dir.exists() and not audition_dir.is_dir():
            parser.error(f"--write-audition-dir is not a directory: {audition_dir}")

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    with tempfile.TemporaryDirectory(prefix="rtttl_quality_") as temp_dir:
        temp_path = Path(temp_dir)
        paths: dict[str, Path] = {}
        for fixture in FIXTURES:
            path = temp_path / f"{fixture.name}.wav"
            synthesize_fixture(fixture, path)
            paths[fixture.name] = path

        results: dict[str, float] = {}
        for setting in SETTINGS:
            average, rows = evaluate_setting(paths, setting)
            results[setting.name] = average
            print(f"{setting.name}: average={average:.3f}")
            for row in rows:
                print(f"  {row}")

    current = results.get("current", 0.0)
    legacy = results.get("legacy", 0.0)
    if current < args.min_current_score:
        print(
            f"current score {current:.3f} is below required {args.min_current_score:.3f}",
            file=sys.stderr,
        )
        return 1
    if current < legacy:
        print(f"current score {current:.3f} regressed below legacy {legacy:.3f}", file=sys.stderr)
        return 1

    if excerpts:
        try:
            print("real excerpts:")
            for row in evaluate_real_settings(excerpts):
                print(row)
            if audition_dir is not None:
                written = write_audition_pack(audition_dir, excerpts)
                print(f"audition files: wrote {len(written)} files to {audition_dir}")
        except Exception as exc:
            print(f"error: real-excerpt evaluation failed: {exc}", file=sys.stderr)
            return 2
    else:
        print("real excerpts: none supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
