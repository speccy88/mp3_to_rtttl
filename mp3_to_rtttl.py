#!/usr/bin/env python3
"""Analyze short audio excerpts and render a monophonic RTTTL melody.

The expensive audio decode and ONNX transcription are deliberately separated
from the inexpensive melody rendering stage.  The desktop studio can therefore
analyze an excerpt once and compare many decoder settings without rerunning the
model.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import numpy as np

from rtttl_utils import NOTE_NAMES, parse_rtttl_tokens

RTTTL_MIN_MIDI = 60  # C4
RTTTL_MAX_MIDI = 95  # B7
DEFAULT_GRID = 16
DEFAULT_MIN_CONFIDENCE = 0.32
DEFAULT_ONSET_THRESHOLD = 0.45
DEFAULT_FRAME_THRESHOLD = 0.25
DEFAULT_MINIMUM_NOTE_MS = 100.0
DEFAULT_TOP_VOICE_BIAS = 0.10
DEFAULT_JUMP_PENALTY = 0.055
DEFAULT_NOTE_CHANGE_PENALTY = 0.14
DEFAULT_ONSET_CHANGE_BONUS = 0.70
DEFAULT_ONSET_CHANGE_RELIEF = 0.24
DEFAULT_ONSET_SAME_PITCH_BONUS = 0.15
DEFAULT_SUSTAIN_BONUS = 0.42
DEFAULT_REPEATED_NOTE_MIN_STEPS = 2
DEFAULT_MINIMUM_MELODY_RUN_STEPS = 2
DEFAULT_OCTAVE_SMOOTHING = True
DEFAULT_MAX_MELODY_SPAN = 14
DEFAULT_OCTAVE_ORIGINAL_WEIGHT = 0.04
DEFAULT_OCTAVE_CENTER_WEIGHT = 0.01
DEFAULT_OCTAVE_JUMP_WEIGHT = 0.08
DEFAULT_OCTAVE_BIG_JUMP_WEIGHT = 0.35
DEFAULT_DECODER_MODE = "events"
DEFAULT_CONTOUR_CANDIDATE_FLOOR = 0.10
DEFAULT_CONTOUR_RELATIVE_FLOOR = 0.30
DEFAULT_CONTOUR_MAX_CANDIDATES = 8
DEFAULT_CONTOUR_TIME_SUPPORT = 0.22
DEFAULT_OCTAVE_JUMP_PENALTY = 0.12
DEFAULT_UPPER_RELATIVE_FLOOR = 0.40
DEFAULT_UPPER_OCTAVE_GUARD_RATIO = 1.20
DEFAULT_UPPER_DROPOUT_SEMITONES = 14
DEFAULT_UPPER_DROPOUT_MAX_STEPS = 10
DEFAULT_UPPER_ONSET_RELATIVE_FLOOR = 0.75
DEFAULT_UPPER_ONSET_MAX_SEMITONES = 8
BASIC_PITCH_MIDI_BASE = 21
DETECTION_MIN_MIDI = 36
DETECTION_MAX_MIDI = 95
RTTTL_BPMS = (
    25,
    28,
    31,
    35,
    40,
    45,
    50,
    56,
    63,
    70,
    80,
    90,
    100,
    112,
    125,
    140,
    160,
    180,
    200,
    225,
    250,
    285,
    320,
    355,
    400,
    450,
    500,
    565,
    635,
    715,
    800,
    900,
)

ProgressCallback = Callable[[str], None]
_AUDIO_DEPENDENCIES = None
_MODEL_RUNNER = None
_EVENT_EXTRACTOR = None
_DEPENDENCY_LOCK = Lock()
_TRANSCRIPTION_LOCK = Lock()


@dataclass(frozen=True)
class MelodySettings:
    """Controls used to turn polyphonic note evidence into one RTTTL voice."""

    grid: int = DEFAULT_GRID
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD
    minimum_note_ms: float = DEFAULT_MINIMUM_NOTE_MS
    top_voice_bias: float = DEFAULT_TOP_VOICE_BIAS
    jump_penalty: float = DEFAULT_JUMP_PENALTY
    note_change_penalty: float = DEFAULT_NOTE_CHANGE_PENALTY
    onset_change_bonus: float = DEFAULT_ONSET_CHANGE_BONUS
    onset_change_relief: float = DEFAULT_ONSET_CHANGE_RELIEF
    onset_same_pitch_bonus: float = DEFAULT_ONSET_SAME_PITCH_BONUS
    sustain_bonus: float = DEFAULT_SUSTAIN_BONUS
    repeated_note_min_steps: int = DEFAULT_REPEATED_NOTE_MIN_STEPS
    minimum_melody_run_steps: int = DEFAULT_MINIMUM_MELODY_RUN_STEPS
    strict_rtttl_bpm: bool = False
    octave_smoothing: bool = DEFAULT_OCTAVE_SMOOTHING
    max_melody_span: int = DEFAULT_MAX_MELODY_SPAN
    decoder_mode: str = DEFAULT_DECODER_MODE
    contour_candidate_floor: float = DEFAULT_CONTOUR_CANDIDATE_FLOOR
    contour_relative_floor: float = DEFAULT_CONTOUR_RELATIVE_FLOOR
    contour_max_candidates: int = DEFAULT_CONTOUR_MAX_CANDIDATES
    contour_time_support: float = DEFAULT_CONTOUR_TIME_SUPPORT
    octave_jump_penalty: float = DEFAULT_OCTAVE_JUMP_PENALTY
    upper_relative_floor: float = DEFAULT_UPPER_RELATIVE_FLOOR
    upper_octave_guard_ratio: float = DEFAULT_UPPER_OCTAVE_GUARD_RATIO
    upper_dropout_semitones: int = DEFAULT_UPPER_DROPOUT_SEMITONES
    upper_dropout_max_steps: int = DEFAULT_UPPER_DROPOUT_MAX_STEPS
    upper_onset_relative_floor: float = DEFAULT_UPPER_ONSET_RELATIVE_FLOOR
    upper_onset_max_semitones: int = DEFAULT_UPPER_ONSET_MAX_SEMITONES


@dataclass(frozen=True)
class TranscriptionAnalysis:
    """Cached Basic Pitch output and timing metadata for one audio excerpt."""

    input_path: Path
    start: float
    duration: float
    estimated_bpm: float
    sample_rate: int
    source_mode: str
    model_output: dict[str, np.ndarray]


def sanitize_title(value: str) -> str:
    """Return an RTTTL-safe title limited to the format's common 10 characters."""

    title = re.sub(r"[^A-Za-z0-9_-]", "", value)[:10]
    return title or "Ringtone"


def nearest_rtttl_bpm(value: float) -> int:
    """Return the historical RTTTL preset tempo closest to ``value``."""

    return min(RTTTL_BPMS, key=lambda candidate: abs(candidate - value))


def choose_rtttl_bpm(value: float, strict_presets: bool = False) -> int:
    """Normalize a tempo to either an integer BPM or a legacy RTTTL preset."""

    if strict_presets:
        return nearest_rtttl_bpm(value)
    if not np.isfinite(value):
        value = 120.0
    return int(round(max(25.0, min(900.0, value))))


def load_audio_dependencies():
    """Load and cache the optional audio stack and the Basic Pitch ONNX model."""

    global _AUDIO_DEPENDENCIES, _EVENT_EXTRACTOR, _MODEL_RUNNER

    if _AUDIO_DEPENDENCIES is not None:
        return _AUDIO_DEPENDENCIES

    with _DEPENDENCY_LOCK:
        if _AUDIO_DEPENDENCIES is not None:
            return _AUDIO_DEPENDENCIES

        _AUDIO_DEPENDENCIES = _load_audio_dependencies_uncached()
        return _AUDIO_DEPENDENCIES


def _load_audio_dependencies_uncached():
    global _EVENT_EXTRACTOR, _MODEL_RUNNER
    missing: list[str] = []

    try:
        import librosa
    except ModuleNotFoundError:
        missing.append("librosa")
        librosa = None

    try:
        import soundfile as sf
    except ModuleNotFoundError:
        missing.append("soundfile")
        sf = None

    basic_pitch_available = importlib.util.find_spec("basic_pitch") is not None
    onnx_available = importlib.util.find_spec("onnxruntime") is not None
    if not basic_pitch_available:
        missing.append("basic-pitch")
    if not onnx_available:
        missing.append("onnxruntime")

    predict = None
    if basic_pitch_available and onnx_available:
        # basic-pitch prefers Core ML on macOS, but some Core ML/Python
        # combinations can stall during import. Its bundled ONNX model is the
        # same transcription network and behaves consistently across systems.
        sentinel = object()
        previous_coreml = sys.modules.get("coremltools", sentinel)
        previous_logging_disable = logging.root.manager.disable
        sys.modules["coremltools"] = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
                logging.disable(logging.CRITICAL)
                import basic_pitch
                from basic_pitch import note_creation
                from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
                from basic_pitch.inference import Model
                from basic_pitch.inference import predict as basic_pitch_predict
                from basic_pitch.inference import run_inference as basic_pitch_run_inference

            onnx_model_path = (
                Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
            )
            loaded_model = Model(onnx_model_path)

            def predict(audio_path, **kwargs):
                with _TRANSCRIPTION_LOCK:
                    return basic_pitch_predict(
                        audio_path,
                        model_or_model_path=loaded_model,
                        **kwargs,
                    )

            def run_model(audio_path):
                with _TRANSCRIPTION_LOCK:
                    return basic_pitch_run_inference(audio_path, loaded_model)

            def extract_events(
                model_output,
                *,
                onset_threshold,
                frame_threshold,
                minimum_note_ms,
                minimum_frequency,
                maximum_frequency,
                midi_tempo,
            ):
                minimum_frames = int(
                    np.round(minimum_note_ms / 1000.0 * (AUDIO_SAMPLE_RATE / FFT_HOP))
                )
                return note_creation.model_output_to_notes(
                    model_output,
                    onset_thresh=onset_threshold,
                    frame_thresh=frame_threshold,
                    min_note_len=max(1, minimum_frames),
                    min_freq=minimum_frequency,
                    max_freq=maximum_frequency,
                    multiple_pitch_bends=False,
                    melodia_trick=True,
                    midi_tempo=midi_tempo,
                )

            _MODEL_RUNNER = run_model
            _EVENT_EXTRACTOR = extract_events
        finally:
            logging.disable(previous_logging_disable)
            if previous_coreml is sentinel:
                sys.modules.pop("coremltools", None)
            else:
                sys.modules["coremltools"] = previous_coreml

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required audio dependencies: {names}. "
            "Install them with: python3 -m pip install -r requirements.txt"
        )

    return librosa, sf, predict


def spectral_flatness_score(samples: np.ndarray) -> float:
    """Return a small, fast flatness estimate for source-mode selection."""
    values = np.asarray(samples, dtype=np.float32)
    frame_size = 2048
    hop_size = 512
    if values.size < frame_size:
        values = np.pad(values, (0, frame_size - values.size))
    windows = np.lib.stride_tricks.sliding_window_view(values, frame_size)[::hop_size]
    power = np.abs(np.fft.rfft(windows * np.hanning(frame_size), axis=1)) ** 2
    power = np.maximum(power, 1e-12)
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return float(np.mean(geometric / np.maximum(arithmetic, 1e-12)))


def analyze_audio(
    input_path: Path | str,
    *,
    start: float = 0.0,
    duration: float = 30.0,
    bpm_hint: Optional[float] = None,
    source_mode: str = "harmonic",
    progress: Optional[ProgressCallback] = None,
) -> TranscriptionAnalysis:
    """Run the expensive audio preparation and ONNX inference once."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        start_seconds = float(start)
        duration_seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Excerpt start and duration must be numbers.") from exc
    if not math.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError("Excerpt start must be a finite, non-negative number.")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("Excerpt duration must be a finite, positive number.")

    requested_mode = str(source_mode).strip().lower().replace("_", "-")
    if requested_mode in ("full-mix", "mix"):
        requested_mode = "full"
    if requested_mode not in ("auto", "full", "harmonic"):
        raise ValueError(f"Unsupported analysis source mode: {source_mode}")

    notify = progress or (lambda _message: None)
    notify("Loading audio dependencies...")
    librosa, sf, _predict = load_audio_dependencies()

    notify("Loading the selected audio...")
    y, sr = librosa.load(
        path,
        sr=22_050,
        mono=True,
        offset=start_seconds,
        duration=duration_seconds,
    )
    if y.size == 0:
        raise RuntimeError("The selected excerpt contains no audio.")

    notify("Separating harmonic content from percussion...")
    harmonic, percussive = librosa.effects.hpss(y)
    if requested_mode == "auto":
        flatness = spectral_flatness_score(y)
        selected_mode = "full" if flatness < 0.035 else "harmonic"
    else:
        selected_mode = requested_mode
    analysis_audio = y if selected_mode == "full" else harmonic
    if selected_mode == "full":
        notify("Using the full mix to preserve short chiptune notes...")
    else:
        notify("Using harmonic content (lead and accompaniment)...")
    if bpm_hint is None:
        notify("Estimating tempo...")
        estimated_bpm = estimate_bpm(percussive, sr, librosa)
    else:
        estimated_bpm = float(bpm_hint)
        if not math.isfinite(estimated_bpm) or estimated_bpm <= 0.0:
            raise ValueError("BPM must be a finite, positive number.")

    notify("Running ONNX transcription once...")
    with tempfile.TemporaryDirectory(prefix="mp3_to_rtttl_") as temp_dir:
        analysis_wav = Path(temp_dir) / "harmonic_excerpt.wav"
        sf.write(analysis_wav, analysis_audio, sr)
        if _MODEL_RUNNER is None:
            raise RuntimeError("The ONNX transcription backend did not initialize.")
        model_output = _MODEL_RUNNER(analysis_wav)

    # HPSS and the transcription network can sustain a note across actual
    # silence. Retain a small envelope of the original signal to veto only
    # near-silent grid cells later; this is not vocal/source separation.
    energy_hop = max(1, round(sr * 0.01))
    padded_audio = np.pad(y, (0, (-y.size) % energy_hop))
    model_output["source_rms"] = np.sqrt(np.mean(padded_audio.reshape(-1, energy_hop) ** 2, axis=1))
    model_output["source_rms_hop_seconds"] = np.asarray(energy_hop / sr)

    return TranscriptionAnalysis(
        input_path=path.resolve(),
        start=start_seconds,
        duration=float(y.size / sr),
        estimated_bpm=estimated_bpm,
        sample_rate=sr,
        source_mode=selected_mode,
        model_output={name: np.asarray(values) for name, values in model_output.items()},
    )


def extract_note_events(
    analysis: TranscriptionAnalysis,
    settings: MelodySettings,
    rtttl_bpm: int,
):
    """Extract filtered MIDI note events from cached model output."""

    load_audio_dependencies()
    if _EVENT_EXTRACTOR is None:
        raise RuntimeError("The Basic Pitch note extractor did not initialize.")

    minimum_frequency = 65.40639132514966  # C2
    maximum_frequency = 3951.066410048992  # B7
    return _EVENT_EXTRACTOR(
        analysis.model_output,
        onset_threshold=settings.onset_threshold,
        frame_threshold=settings.frame_threshold,
        minimum_note_ms=settings.minimum_note_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        midi_tempo=float(rtttl_bpm),
    )


def estimate_bpm(y_percussive: np.ndarray, sr: int, _librosa_module=None) -> float:
    """Estimate tempo from spectral-flux autocorrelation without a JIT warm-up."""
    samples = np.asarray(y_percussive, dtype=np.float32)
    frame_size = 1024
    hop_size = 256
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))

    windows = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop_size]
    if windows.shape[0] < 4:
        return 120.0
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(frame_size), axis=1))
    flux = np.maximum(np.diff(np.log1p(spectrum), axis=0), 0.0).sum(axis=1)
    flux = np.maximum(flux - np.median(flux), 0.0)
    flux -= np.mean(flux)
    if not np.any(flux):
        return 120.0

    fft_size = 1 << int(math.ceil(math.log2(max(2, flux.size * 2 - 1))))
    transformed = np.fft.rfft(flux, n=fft_size)
    correlation = np.fft.irfft(transformed * np.conj(transformed), n=fft_size)[: flux.size]
    minimum_lag = max(1, int(math.floor(60.0 * sr / (hop_size * 190.0))))
    maximum_lag = min(
        correlation.size - 1,
        int(math.ceil(60.0 * sr / (hop_size * 60.0))),
    )
    if maximum_lag < minimum_lag:
        return 120.0

    best_bpm = 120.0
    best_score = -float("inf")
    for lag in range(minimum_lag, maximum_lag + 1):
        bpm = 60.0 * sr / (hop_size * lag)
        normalized = correlation[lag] / max(1, flux.size - lag)
        ringtone_prior = math.exp(-0.5 * (math.log2(bpm / 110.0) / 0.65) ** 2)
        score = normalized * ringtone_prior
        if score > best_score:
            best_score = score
            best_bpm = bpm
    return best_bpm


def build_quantized_melody_grid(
    events: list[tuple[float, float, int, float]],
    bpm: int,
    grid: int,
) -> tuple[list[dict[int, float]], set[int], list[dict[int, float]]]:
    """
    Build a piano-roll-like grid.

    grid=16 means one cell is a sixteenth note.
    grid=32 means one cell is a thirty-second note.
    """
    if bpm <= 0:
        raise ValueError("BPM must be positive when building a melody grid.")
    if grid <= 0:
        raise ValueError("Grid resolution must be positive.")
    if not events:
        return [], set(), []

    seconds_per_step = (240.0 / bpm) / grid
    last_time = max(end for _, end, _, _ in events)
    # Quantize the excerpt end to the nearest cell. Ceil always added a weak
    # tail cell, which often let accompaniment replace the final melody note.
    n_steps = max(1, int(round(last_time / seconds_per_step)))
    slots: list[dict[int, float]] = [dict() for _ in range(n_steps)]
    onset_steps: set[int] = set()
    onset_pitches: list[dict[int, float]] = [dict() for _ in range(n_steps)]

    for start, end, pitch, confidence in events:
        if end <= start:
            continue

        onset = int(round(start / seconds_per_step))
        onset = max(0, min(onset, n_steps - 1))
        onset_steps.add(onset)
        onset_pitches[onset][pitch] = max(onset_pitches[onset].get(pitch, 0.0), confidence)

        # Snap the sounding span to the same grid as its attack. Using floor
        # and ceil here made almost every slight timing offset grow a note by
        # a full cell, consuming short rests and starting the next note early.
        first = onset
        last = min(n_steps, max(first + 1, int(round(end / seconds_per_step))))

        for index in range(first, last):
            cell_start = index * seconds_per_step
            cell_end = cell_start + seconds_per_step
            overlap = max(0.0, min(end, cell_end) - max(start, cell_start))
            overlap_ratio = overlap / seconds_per_step
            if overlap_ratio <= 0.0:
                # A sub-cell note can round wholly into its attack cell.
                overlap_ratio = min(1.0, (end - start) / seconds_per_step)

            # A note touching a cell only briefly must not compete as if it
            # occupied half of it. That inflated quiet tails and early attacks.
            score = confidence * overlap_ratio
            slots[index][pitch] = max(slots[index].get(pitch, 0.0), score)

    return slots, onset_steps, onset_pitches


def build_quantized_slots(
    events: list[tuple[float, float, int, float]],
    bpm: int,
    grid: int,
) -> tuple[list[dict[int, float]], set[int]]:
    """Compatibility wrapper returning overlap slots and onset step indexes."""

    slots, onset_steps, _onset_pitches = build_quantized_melody_grid(events, bpm, grid)
    return slots, onset_steps


def build_model_salience_grid(
    analysis: TranscriptionAnalysis,
    bpm: int,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate raw Basic Pitch note/onset evidence onto the RTTTL grid."""
    if bpm <= 0:
        raise ValueError("BPM must be positive when building a salience grid.")
    if grid <= 0:
        raise ValueError("Grid resolution must be positive.")
    notes = np.asarray(analysis.model_output.get("note"), dtype=np.float32)
    onsets = np.asarray(analysis.model_output.get("onset"), dtype=np.float32)
    if notes.ndim != 2 or notes.shape[0] == 0 or notes.shape[1] == 0:
        raise RuntimeError("The transcription model returned no note salience data.")
    if onsets.shape != notes.shape:
        onsets = np.zeros_like(notes)

    seconds_per_step = (240.0 / bpm) / grid
    step_count = max(1, int(round(analysis.duration / seconds_per_step)))
    note_grid = np.zeros((step_count, notes.shape[1]), dtype=np.float32)
    onset_grid = np.zeros_like(note_grid)

    for step in range(step_count):
        first = max(0, int(math.floor(step / step_count * notes.shape[0])) - 1)
        last = min(
            notes.shape[0],
            max(first + 1, int(math.ceil((step + 1) / step_count * notes.shape[0])) + 1),
        )
        note_window = notes[first:last]
        note_grid[step] = 0.60 * np.max(note_window, axis=0) + 0.40 * np.mean(note_window, axis=0)

    frame_steps = np.rint(
        np.arange(onsets.shape[0], dtype=np.float32)
        / max(1, onsets.shape[0] - 1)
        * max(0, step_count - 1)
    ).astype(np.int32)
    for frame, step in enumerate(frame_steps):
        onset_grid[step] = np.maximum(onset_grid[step], onsets[frame])
    previous = np.vstack((np.zeros((1, onset_grid.shape[1]), dtype=np.float32), onset_grid[:-1]))
    following = np.vstack((onset_grid[1:], np.zeros((1, onset_grid.shape[1]), dtype=np.float32)))
    onset_grid = np.where(
        (onset_grid >= previous) & (onset_grid > following),
        onset_grid,
        0.0,
    ).astype(np.float32)

    return note_grid, onset_grid


def select_salience_contour(
    note_grid: np.ndarray,
    onset_grid: np.ndarray,
    settings: MelodySettings,
) -> tuple[list[Optional[int]], set[int], list[dict[int, float]]]:
    """Track one time-continuous predominant contour through polyphonic salience."""
    if note_grid.ndim != 2 or note_grid.shape[0] == 0:
        return [], set(), []

    step_count, pitch_count = note_grid.shape
    low_index = max(0, DETECTION_MIN_MIDI - BASIC_PITCH_MIDI_BASE)
    high_index = min(pitch_count - 1, DETECTION_MAX_MIDI - BASIC_PITCH_MIDI_BASE)
    if high_index < low_index:
        return [], set(), []

    time_support = max(0.0, min(float(settings.contour_time_support), 0.45))
    padded = np.pad(note_grid, ((1, 1), (0, 0)), mode="edge")
    temporal = (
        time_support * padded[:-2]
        + (1.0 - 2.0 * time_support) * note_grid
        + time_support * padded[2:]
    )
    support_count = max(2, step_count // 5)
    global_support = np.mean(np.sort(note_grid, axis=0)[-support_count:], axis=0)
    onset_floor = max(0.08, settings.onset_threshold * 0.40)

    previous_scores: dict[Optional[int], float] = {None: 0.0}
    backpointers: list[dict[Optional[int], Optional[int]]] = []
    onset_steps: set[int] = set()
    onset_pitches: list[dict[int, float]] = []

    for step in range(step_count):
        salience = (1.0 - time_support) * note_grid[step] + time_support * temporal[step]
        visible = salience[low_index : high_index + 1]
        strongest = float(np.max(visible)) if visible.size else 0.0
        threshold = max(
            settings.contour_candidate_floor,
            strongest * settings.contour_relative_floor,
        )
        candidate_indices: list[int] = []
        for index in range(low_index, high_index + 1):
            value = float(salience[index])
            if value < threshold:
                continue
            left = float(salience[index - 1]) if index > low_index else -1.0
            right = float(salience[index + 1]) if index < high_index else -1.0
            if value >= left and value >= right:
                candidate_indices.append(index)

        continuation_floor = settings.contour_candidate_floor * 0.60
        for previous in previous_scores:
            if previous is None:
                continue
            index = previous - BASIC_PITCH_MIDI_BASE
            if (
                low_index <= index <= high_index
                and float(salience[index]) >= continuation_floor
                and index not in candidate_indices
            ):
                candidate_indices.append(index)

        candidate_indices.sort(
            key=lambda index: (
                float(salience[index])
                + 0.28 * float(onset_grid[step, index])
                + 0.18 * float(global_support[index])
            ),
            reverse=True,
        )
        candidate_indices = candidate_indices[: max(3, settings.contour_max_candidates)]
        states: list[Optional[int]] = [None] + [
            index + BASIC_PITCH_MIDI_BASE for index in candidate_indices
        ]
        onset_slot = {
            index + BASIC_PITCH_MIDI_BASE: float(onset_grid[step, index])
            for index in candidate_indices
            if float(onset_grid[step, index]) >= onset_floor
        }
        onset_pitches.append(onset_slot)
        if onset_slot:
            onset_steps.add(step)

        current_scores: dict[Optional[int], float] = {}
        current_back: dict[Optional[int], Optional[int]] = {}
        for state in states:
            if state is None:
                emission = 0.32 - strongest
            else:
                index = state - BASIC_PITCH_MIDI_BASE
                emission = (
                    2.8 * float(note_grid[step, index])
                    + 0.8 * float(temporal[step, index])
                    + 0.35 * float(global_support[index])
                    + settings.top_voice_bias * (state - 60)
                    + settings.onset_change_bonus * float(onset_grid[step, index])
                )

            best_score = -float("inf")
            best_previous: Optional[int] = None
            for previous, previous_score in previous_scores.items():
                if previous is None and state is None:
                    transition = 0.16
                elif previous is None or state is None:
                    transition = -0.30
                elif previous == state:
                    transition = (
                        settings.onset_same_pitch_bonus
                        if state in onset_slot
                        else settings.sustain_bonus
                    )
                else:
                    jump = abs(state - previous)
                    transition = (
                        -settings.note_change_penalty
                        - settings.jump_penalty * jump
                        - settings.octave_jump_penalty * max(0, jump - 7)
                    )
                    if state in onset_slot:
                        transition += settings.onset_change_relief * (0.5 + onset_slot[state])

                candidate_score = previous_score + transition + emission
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous

            current_scores[state] = best_score
            current_back[state] = best_previous

        previous_scores = current_scores
        backpointers.append(current_back)

    state = max(previous_scores, key=previous_scores.get)
    path: list[Optional[int]] = [state]
    for step in range(step_count - 1, 0, -1):
        state = backpointers[step][state]
        path.append(state)
    path.reverse()
    return path, onset_steps, onset_pitches


def select_monophonic_melody(
    slots: list[dict[int, float]],
    top_voice_bias: float = 0.01,
    jump_penalty: float = 0.035,
    onset_pitches: Optional[list[dict[int, float]]] = None,
    onset_change_bonus: float = DEFAULT_ONSET_CHANGE_BONUS,
    onset_change_relief: float = DEFAULT_ONSET_CHANGE_RELIEF,
    onset_same_pitch_bonus: float = DEFAULT_ONSET_SAME_PITCH_BONUS,
    sustain_bonus: float = DEFAULT_SUSTAIN_BONUS,
    note_change_penalty: float = DEFAULT_NOTE_CHANGE_PENALTY,
) -> list[Optional[int]]:
    """
    Viterbi-like decoding: one pitch (or rest) per time cell.
    Confidence favors salient notes, a small pitch bias favors the top melody,
    transition costs discourage implausible jumps, and detected note onsets help
    the line move on musical attacks instead of clinging to overlapping tones.
    """
    if not slots:
        return []

    previous_scores: dict[Optional[int], float] = {None: 0.0}
    backpointers: list[dict[Optional[int], Optional[int]]] = []

    if onset_pitches is None:
        onset_pitches = [dict() for _ in slots]

    for index, slot in enumerate(slots):
        states: list[Optional[int]] = list(slot) if slot else [None]
        current_scores: dict[Optional[int], float] = {}
        current_back: dict[Optional[int], Optional[int]] = {}
        onset_slot = onset_pitches[index] if index < len(onset_pitches) else {}
        has_onset = bool(onset_slot)

        for state in states:
            if state is None:
                emission = 0.0
            else:
                emission = 2.5 * slot[state] + top_voice_bias * (state - 60)
                if state in onset_slot:
                    emission += onset_change_bonus * (0.5 + onset_slot[state])

            best_score = -float("inf")
            best_previous: Optional[int] = None

            for previous, previous_score in previous_scores.items():
                if previous is None and state is None:
                    transition = 0.15
                elif previous is None or state is None:
                    transition = -0.20
                elif previous == state:
                    transition = onset_same_pitch_bonus if has_onset else sustain_bonus
                else:
                    transition = -note_change_penalty - jump_penalty * abs(state - previous)
                    if state in onset_slot:
                        transition += onset_change_relief

                candidate = previous_score + transition + emission
                if candidate > best_score:
                    best_score = candidate
                    best_previous = previous

            current_scores[state] = best_score
            current_back[state] = best_previous

        previous_scores = current_scores
        backpointers.append(current_back)

    state = max(previous_scores, key=previous_scores.get)
    path: list[Optional[int]] = [state]

    for time_index in range(len(slots) - 1, 0, -1):
        state = backpointers[time_index][state]
        path.append(state)

    path.reverse()
    return path


def _filter_upper_event_harmonics(
    source: dict[int, float],
    previous: Optional[int],
    octave_guard_ratio: float,
) -> dict[int, float]:
    filtered: dict[int, float] = {}
    for pitch, strength in source.items():
        lower_strength = source.get(pitch - 12)
        continues_selected_octave = previous is not None and pitch - 12 == previous
        if (
            lower_strength is not None
            and lower_strength >= strength * octave_guard_ratio
            and not continues_selected_octave
        ):
            continue
        filtered[pitch] = strength
    return filtered


def _suppress_upper_voice_dropouts(
    path: list[Optional[int]],
    drop_semitones: int,
    max_gap_steps: int,
) -> list[Optional[int]]:
    """Turn short drops into lower accompaniment into rests when the lead returns."""
    if not path:
        return []
    runs: list[tuple[int, int, Optional[int]]] = []
    start = 0
    for index in range(1, len(path) + 1):
        if index == len(path) or path[index] != path[start]:
            runs.append((start, index, path[start]))
            start = index

    cleaned = path.copy()
    recent: list[int] = []
    for run_index, (start, end, pitch) in enumerate(runs):
        if pitch is None:
            continue
        center = float(np.median(recent[-8:])) if recent else float(pitch)
        if recent and pitch <= center - drop_semitones:
            recovery = None
            for next_start, _next_end, next_pitch in runs[run_index + 1 :]:
                if next_start - start > max_gap_steps:
                    break
                if next_pitch is not None and next_pitch >= center - 7:
                    recovery = next_start
                    break
            if recovery is not None:
                cleaned[start:recovery] = [None] * (recovery - start)
                continue
        recent.extend([pitch] * (end - start))
    return cleaned


def select_upper_event_melody(
    events: list[tuple[float, float, int, float]],
    bpm: int,
    grid: int,
    settings: MelodySettings,
) -> tuple[list[Optional[int]], set[int], list[dict[int, float]]]:
    """Follow the highest credible event voice while rejecting octave harmonics."""
    slots, onset_steps, onset_pitches = build_quantized_melody_grid(events, bpm, grid)
    path: list[Optional[int]] = []
    previous: Optional[int] = None

    for step, slot in enumerate(slots):
        filtered = _filter_upper_event_harmonics(
            slot,
            previous,
            settings.upper_octave_guard_ratio,
        )
        if not filtered:
            path.append(None)
            previous = None
            continue
        strongest = max(filtered.values())
        credible = [
            pitch
            for pitch, strength in filtered.items()
            if strength >= strongest * settings.upper_relative_floor
        ]
        selected = max(credible) if credible else None

        onset_slot = onset_pitches[step] if step < len(onset_pitches) else {}
        if selected == previous and previous is not None and onset_slot:
            attacks = _filter_upper_event_harmonics(
                onset_slot,
                previous,
                settings.upper_octave_guard_ratio,
            )
            if attacks:
                attack_peak = max(attacks.values())
                interrupting_attacks = [
                    pitch
                    for pitch, strength in attacks.items()
                    if pitch != previous
                    and abs(pitch - previous) <= settings.upper_onset_max_semitones
                    and strength >= attack_peak * settings.upper_onset_relative_floor
                ]
                if interrupting_attacks:
                    selected = max(interrupting_attacks)

        path.append(selected)
        previous = selected

    path = _suppress_upper_voice_dropouts(
        path,
        drop_semitones=max(8, settings.upper_dropout_semitones),
        max_gap_steps=max(2, settings.upper_dropout_max_steps),
    )
    return path, onset_steps, onset_pitches


def remove_one_step_blips(
    path: list[Optional[int]],
    onset_pitches: Optional[list[dict[int, float]]] = None,
) -> list[Optional[int]]:
    """Replace unsupported one-cell pitch spikes with their shared neighbor."""

    if len(path) < 3:
        return path
    cleaned = path.copy()
    for index in range(1, len(cleaned) - 1):
        pitch = cleaned[index]
        if pitch is None:
            # A sixteenth-note rest is part of the rhythm, not a pitch glitch.
            continue
        protected_onset = (
            pitch is not None
            and onset_pitches is not None
            and index < len(onset_pitches)
            and pitch in onset_pitches[index]
        )
        if not protected_onset and cleaned[index - 1] == cleaned[index + 1] != pitch:
            cleaned[index] = cleaned[index - 1]
    return cleaned


def merge_short_note_runs(
    path: list[Optional[int]],
    minimum_steps: int = DEFAULT_MINIMUM_MELODY_RUN_STEPS,
    onset_pitches: Optional[list[dict[int, float]]] = None,
) -> list[Optional[int]]:
    """Merge unsupported short pitch runs while preserving attacks and rests."""
    if len(path) < 2 or minimum_steps <= 1:
        return path

    merged = path.copy()
    for _pass in range(len(merged)):
        runs: list[tuple[int, int, Optional[int]]] = []
        run_start = 0
        for index in range(1, len(merged) + 1):
            if index == len(merged) or merged[index] != merged[run_start]:
                runs.append((run_start, index, merged[run_start]))
                run_start = index

        changed = False
        for run_index, (start, end, pitch) in enumerate(runs):
            if pitch is None or end - start >= minimum_steps:
                continue
            if onset_pitches is not None and any(
                pitch in onset_pitches[index]
                for index in range(start, min(end, len(onset_pitches)))
            ):
                continue

            previous_run = runs[run_index - 1] if run_index else None
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            previous_pitch = previous_run[2] if previous_run else None
            next_pitch = next_run[2] if next_run else None

            neighbors = []
            if previous_run and previous_pitch is not None:
                neighbors.append((previous_pitch, previous_run[1] - previous_run[0]))
            if next_run and next_pitch is not None:
                neighbors.append((next_pitch, next_run[1] - next_run[0]))
            if not neighbors:
                continue
            replacement, _length = min(
                neighbors,
                key=lambda item: (abs(item[0] - pitch), -item[1]),
            )

            merged[start:end] = [replacement] * (end - start)
            changed = True
            break

        if not changed:
            break

    return merged


def fit_to_rtttl_octaves(path: list[Optional[int]]) -> list[Optional[int]]:
    """Shift a melody into RTTTL's practical C4-to-B7 pitch range."""

    pitches = [pitch for pitch in path if pitch is not None]
    if not pitches:
        return path

    median_pitch = float(np.median(pitches))
    candidates = range(-48, 49, 12)

    def shift_score(shift: int) -> tuple[int, float]:
        inside = sum(RTTTL_MIN_MIDI <= pitch + shift <= RTTTL_MAX_MIDI for pitch in pitches)
        center_penalty = abs((median_pitch + shift) - 72.0)
        return inside, -center_penalty

    best_shift = max(candidates, key=shift_score)
    fitted: list[Optional[int]] = []

    for pitch in path:
        if pitch is None:
            fitted.append(None)
            continue

        value = pitch + best_shift
        while value < RTTTL_MIN_MIDI:
            value += 12
        while value > RTTTL_MAX_MIDI:
            value -= 12
        fitted.append(value)

    return fitted


def preserve_source_silence(
    path: list[Optional[int]], analysis: TranscriptionAnalysis, bpm: int, grid: int
) -> list[Optional[int]]:
    """Veto model sustain in near-silent source cells, leaving quiet notes alone."""
    rms = np.asarray(analysis.model_output.get("source_rms", []), dtype=np.float32)
    if rms.ndim != 1 or not rms.size or not np.all(np.isfinite(rms)):
        return path
    hop_seconds = float(analysis.model_output.get("source_rms_hop_seconds", 0.01))
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        return path
    # 0.5% of the source peak RMS is about -46 dB. A cell must be mostly at
    # this floor; a merely quiet note or a short articulation gap stays intact.
    silence_floor = float(np.max(rms)) * 0.005
    seconds_per_step = (240.0 / bpm) / grid
    result = path.copy()
    for step in range(len(result)):
        first = max(0, int(math.floor(step * seconds_per_step / hop_seconds)))
        last = min(rms.size, int(math.ceil((step + 1) * seconds_per_step / hop_seconds)))
        window = rms[first:last]
        if window.size and np.mean(window <= silence_floor) >= 0.75:
            result[step] = None
    return result


def smooth_octave_contour(
    path: list[Optional[int]],
    max_melody_span: int = DEFAULT_MAX_MELODY_SPAN,
    original_weight: float = DEFAULT_OCTAVE_ORIGINAL_WEIGHT,
    center_weight: float = DEFAULT_OCTAVE_CENTER_WEIGHT,
    jump_weight: float = DEFAULT_OCTAVE_JUMP_WEIGHT,
    big_jump_weight: float = DEFAULT_OCTAVE_BIG_JUMP_WEIGHT,
) -> list[Optional[int]]:
    """
    Re-spell each detected pitch class into a ringtone-friendly octave range.

    Basic Pitch is good at note names, but dense MP3s can produce octave-sized
    harmonic jumps that sound strange as a square-wave ringtone. If the detected
    range is already compact, leave it alone. Otherwise, search for the best
    compact octave window and preserve pitch classes inside it.
    """
    if not path:
        return []

    pitches = [pitch for pitch in path if pitch is not None]
    if not pitches:
        return path
    if max(pitches) - min(pitches) <= max_melody_span:
        return path

    max_melody_span = max(12, min(max_melody_span, RTTTL_MAX_MIDI - RTTTL_MIN_MIDI))
    best_score = -float("inf")
    best_path = path

    for low in range(RTTTL_MIN_MIDI, RTTTL_MAX_MIDI - max_melody_span + 1):
        high = low + max_melody_span
        previous_scores: dict[Optional[int], float] = {None: 0.0}
        backpointers: list[dict[Optional[int], Optional[int]]] = []
        valid_window = True

        for pitch in path:
            if pitch is None:
                states: list[Optional[int]] = [None]
            else:
                states = [
                    candidate for candidate in range(low, high + 1) if candidate % 12 == pitch % 12
                ]
                if not states:
                    valid_window = False
                    break

            current_scores: dict[Optional[int], float] = {}
            current_back: dict[Optional[int], Optional[int]] = {}

            for state in states:
                if state is None:
                    emission = 0.0 if pitch is None else -0.2
                else:
                    emission = -original_weight * abs(state - pitch) - center_weight * abs(
                        state - 72
                    )

                best_state_score = -float("inf")
                best_previous: Optional[int] = None

                for previous, previous_score in previous_scores.items():
                    if previous is None or state is None:
                        transition = 0.0 if previous == state else -0.2
                    else:
                        jump = abs(state - previous)
                        transition = -jump_weight * jump - big_jump_weight * max(0, jump - 7)

                    candidate_score = previous_score + emission + transition
                    if candidate_score > best_state_score:
                        best_state_score = candidate_score
                        best_previous = previous

                current_scores[state] = best_state_score
                current_back[state] = best_previous

            previous_scores = current_scores
            backpointers.append(current_back)

        if not valid_window or not previous_scores:
            continue

        state = max(previous_scores, key=previous_scores.get)
        score = previous_scores[state] - 0.02 * abs((low + high) / 2 - 72)
        candidate_path: list[Optional[int]] = [state]
        for index in range(len(path) - 1, 0, -1):
            state = backpointers[index][state]
            candidate_path.append(state)
        candidate_path.reverse()

        if score > best_score:
            best_score = score
            best_path = candidate_path

    return best_path


def collapse_runs(
    path: list[Optional[int]],
    onset_steps: set[int],
    repeated_note_min_steps: int = DEFAULT_REPEATED_NOTE_MIN_STEPS,
    onset_pitches: Optional[list[dict[int, float]]] = None,
    onset_pitch_shift: int = 0,
    trim_silence: bool = True,
) -> list[tuple[Optional[int], int]]:
    """Collapse grid cells into note/rest runs while preserving real re-attacks."""

    if not path:
        return []

    runs: list[tuple[Optional[int], int]] = []
    current = path[0]
    length = 1

    for index in range(1, len(path)):
        matching_pitch_onset = (
            onset_pitches is not None
            and index < len(onset_pitches)
            and current is not None
            and any(
                onset_pitch % 12 == current % 12
                and abs(onset_pitch + onset_pitch_shift - current) <= 12
                for onset_pitch in onset_pitches[index]
            )
        )
        repeated_onset = (
            (matching_pitch_onset or (onset_pitches is None and index in onset_steps))
            and path[index] is not None
            and path[index] == current
            and length >= repeated_note_min_steps
        )
        if path[index] != current or repeated_onset:
            runs.append((current, length))
            current = path[index]
            length = 1
        else:
            length += 1

    runs.append((current, length))

    # Direct callers retain the legacy trim option. The renderer keeps the
    # excerpt timeline so audition and evidence scores remain aligned.
    if trim_silence:
        while runs and runs[0][0] is None:
            runs.pop(0)
        while runs and runs[-1][0] is None:
            runs.pop()

    return runs


def duration_options(grid: int) -> list[tuple[int, int, bool]]:
    """Return (step_count, denominator, dotted), longest first."""
    if not isinstance(grid, int) or isinstance(grid, bool) or grid <= 0:
        raise ValueError("Grid resolution must be a positive integer.")
    options: dict[int, tuple[int, int, bool]] = {}

    for denominator in (1, 2, 4, 8, 16, 32):
        base = grid / denominator
        if base >= 1 and float(base).is_integer():
            steps = int(base)
            options.setdefault(steps, (steps, denominator, False))

        dotted = base * 1.5
        if dotted >= 1 and float(dotted).is_integer():
            steps = int(dotted)
            options.setdefault(steps, (steps, denominator, True))

    return sorted(options.values(), reverse=True)


def split_duration(step_count: int, grid: int) -> list[tuple[int, bool]]:
    """Exact minimum-token decomposition into RTTTL durations."""
    if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count <= 0:
        raise ValueError("Duration must contain at least one integer grid step.")
    options = duration_options(grid)
    best: list[Optional[list[tuple[int, bool]]]] = [None] * (step_count + 1)
    best[0] = []

    for total in range(1, step_count + 1):
        for units, denominator, dotted in options:
            if units <= total and best[total - units] is not None:
                candidate = best[total - units] + [(denominator, dotted)]
                if best[total] is None or len(candidate) < len(best[total]):
                    best[total] = candidate

    if best[step_count] is None:
        raise ValueError(f"Cannot encode duration of {step_count} grid steps")
    return best[step_count]


def midi_to_rtttl_note(midi_pitch: int) -> tuple[str, int]:
    """Convert a supported MIDI pitch into an RTTTL note name and octave."""
    if not RTTTL_MIN_MIDI <= midi_pitch <= RTTTL_MAX_MIDI:
        raise ValueError(f"MIDI pitch must be between {RTTTL_MIN_MIDI} and {RTTTL_MAX_MIDI}.")
    note = NOTE_NAMES[midi_pitch % 12]
    octave = midi_pitch // 12 - 1
    return note, octave


def runs_to_rtttl(
    runs: list[tuple[Optional[int], int]],
    title: str,
    bpm: int,
    grid: int,
) -> str:
    """Encode quantized melody runs as a complete RTTTL string."""
    if not 25 <= bpm <= 900:
        raise ValueError("RTTTL tempo must be between 25 and 900 BPM.")
    duration_options(grid)  # Validate the grid before walking the runs.
    tokens: list[str] = []

    for pitch, steps in runs:
        if steps <= 0:
            raise ValueError("Melody runs must contain at least one grid step.")
        for denominator, dotted in split_duration(steps, grid):
            if pitch is None:
                token = f"{denominator}p"
            else:
                note, octave = midi_to_rtttl_note(pitch)
                token = f"{denominator}{note}{octave}"
            if dotted:
                token += "."
            tokens.append(token)

    if not tokens:
        raise RuntimeError("No melody survived the filtering and quantization steps.")

    return f"{sanitize_title(title)}:d=4,o=5,b={bpm}:" + ",".join(tokens)


def score_rtttl_fidelity(
    analysis: TranscriptionAnalysis,
    rtttl: str,
    grid: int = DEFAULT_GRID,
) -> float:
    """Return a 0..1 source-evidence score for ranking melody hypotheses."""
    try:
        tokens = parse_rtttl_tokens(rtttl, grid=grid)
    except (TypeError, ValueError):
        return 0.0
    notes = np.asarray(analysis.model_output.get("note"), dtype=np.float32)
    onsets = np.asarray(analysis.model_output.get("onset"), dtype=np.float32)
    if notes.ndim != 2 or notes.shape[0] == 0 or not tokens:
        return 0.0
    if onsets.shape != notes.shape:
        onsets = np.zeros_like(notes)

    total_seconds = sum(token.seconds for token in tokens)
    if total_seconds <= 0.0 or not math.isfinite(analysis.duration) or analysis.duration <= 0.0:
        return 0.0
    low_index = max(0, DETECTION_MIN_MIDI - BASIC_PITCH_MIDI_BASE)
    high_index = min(notes.shape[1], DETECTION_MAX_MIDI - BASIC_PITCH_MIDI_BASE + 1)
    if high_index <= low_index:
        return 0.0

    segments: list[tuple[object, int, int, float]] = []
    cursor = 0.0
    pitched_duration = 0.0
    for token in tokens:
        # Use real seconds, not the candidate's normalized position. Stretching
        # every candidate to the source made half/double-tempo errors invisible.
        start_ratio = cursor / analysis.duration
        cursor += token.seconds
        end_ratio = cursor / analysis.duration
        first = min(notes.shape[0], max(0, int(math.floor(start_ratio * notes.shape[0])) - 1))
        last = min(
            notes.shape[0],
            max(first + 1, int(math.ceil(end_ratio * notes.shape[0])) + 1),
        )
        segments.append((token, first, last, token.seconds))
        if token.midi is not None:
            pitched_duration += token.seconds

    best_score = 0.0
    for octave_shift in range(-48, 49, 12):
        note_total = 0.0
        relative_total = 0.0
        duration_weight = pitched_duration
        onset_total = 0.0
        onset_weight = 0.0
        previous_midi: Optional[int] = None

        for token, first, last, duration in segments:
            midi = token.midi
            if midi is None:
                previous_midi = None
                continue
            if first >= last:
                previous_midi = midi
                continue
            model_index = midi + octave_shift - BASIC_PITCH_MIDI_BASE
            if not low_index <= model_index < high_index:
                previous_midi = midi
                continue
            window = notes[first:last]
            pitch_values = window[:, model_index]
            support = 0.60 * float(np.max(pitch_values)) + 0.40 * float(np.mean(pitch_values))
            source_peak = 0.60 * float(np.max(window[:, low_index:high_index])) + 0.40 * float(
                np.mean(np.max(window[:, low_index:high_index], axis=1))
            )
            note_total += support * duration
            relative_total += support / max(source_peak, 1e-6) * duration

            if previous_midi != midi:
                onset_first = max(0, first - 2)
                onset_last = min(onsets.shape[0], first + 3)
                onset_total += float(np.max(onsets[onset_first:onset_last, model_index]))
                onset_weight += 1.0
            previous_midi = midi

        if duration_weight <= 0.0:
            continue
        note_score = note_total / duration_weight
        relative_score = min(1.0, relative_total / duration_weight)
        onset_score = onset_total / onset_weight if onset_weight else note_score
        coverage = pitched_duration / total_seconds
        score = 0.58 * note_score + 0.22 * relative_score + 0.15 * onset_score + 0.05 * coverage
        best_score = max(best_score, score)

    onset_by_pitch_class = np.zeros((onsets.shape[0], 12), dtype=np.float32)
    for model_index in range(low_index, high_index):
        midi = model_index + BASIC_PITCH_MIDI_BASE
        onset_by_pitch_class[:, midi % 12] = np.maximum(
            onset_by_pitch_class[:, midi % 12],
            onsets[:, model_index],
        )

    candidate_attacks: list[int] = []
    previous_pitch_class: Optional[int] = None
    for token, first, _last, _duration in segments:
        if token.midi is None:
            previous_pitch_class = None
            continue
        if first >= notes.shape[0]:
            continue
        pitch_class = token.midi % 12
        local_onset = float(
            np.max(
                onset_by_pitch_class[
                    max(0, first - 4) : min(onsets.shape[0], first + 5),
                    pitch_class,
                ]
            )
        )
        if pitch_class != previous_pitch_class or local_onset >= 0.20:
            candidate_attacks.append(first)
        previous_pitch_class = pitch_class

    reference_candidates: list[tuple[float, int]] = []
    for token, first, last, _duration in segments:
        if token.midi is None:
            continue
        values = onset_by_pitch_class[:, token.midi % 12]
        for frame in range(max(1, first), min(last, values.size - 1)):
            value = float(values[frame])
            if value >= 0.24 and value >= values[frame - 1] and value >= values[frame + 1]:
                reference_candidates.append((value, frame))
    reference_attacks: list[int] = []
    for _value, frame in sorted(reference_candidates, reverse=True):
        if all(abs(frame - existing) > 5 for existing in reference_attacks):
            reference_attacks.append(frame)
    reference_attacks.sort()

    tolerance = max(2, round(notes.shape[0] / max(analysis.duration, 0.1) * 0.05))
    used_candidates: set[int] = set()
    matched = 0
    for reference in reference_attacks:
        choices = [
            candidate
            for candidate in candidate_attacks
            if candidate not in used_candidates and abs(candidate - reference) <= tolerance
        ]
        if choices:
            candidate = min(choices, key=lambda value: abs(value - reference))
            used_candidates.add(candidate)
            matched += 1
    if matched and reference_attacks and candidate_attacks:
        precision = matched / len(candidate_attacks)
        recall = matched / len(reference_attacks)
        attack_score = 2.0 * precision * recall / (precision + recall)
    else:
        attack_score = 0.0

    pitched_tokens = [token.midi for token in tokens if token.midi is not None]
    jumps = [
        abs(current - previous)
        for previous, current in zip(pitched_tokens, pitched_tokens[1:], strict=False)
    ]
    big_jump_penalty = 0.025 * sum(jump >= 12 for jump in jumps)
    density = len(pitched_tokens) / max(analysis.duration, 0.001)
    density_penalty = 0.02 * max(0.0, density - 4.0)
    evidence_score = max(0.0, best_score - big_jump_penalty - density_penalty)
    attack_density = min(
        1.0,
        len(candidate_attacks) / max(1.0, analysis.duration * 1.2),
    )
    combined = 0.50 * evidence_score + 0.35 * attack_score + 0.15 * attack_density

    duration_by_pitch_class: dict[int, float] = {}
    pitch_class_sequence: list[int] = []
    for token in tokens:
        if token.midi is None:
            continue
        pitch_class = token.midi % 12
        duration_by_pitch_class[pitch_class] = (
            duration_by_pitch_class.get(pitch_class, 0.0) + token.seconds
        )
        if not pitch_class_sequence or pitch_class_sequence[-1] != pitch_class:
            pitch_class_sequence.append(pitch_class)
    total_pitched = sum(duration_by_pitch_class.values())
    dominant_ratio = (
        max(duration_by_pitch_class.values()) / total_pitched if total_pitched > 0.0 else 1.0
    )
    dominance_penalty = 0.28 * max(0.0, dominant_ratio - 0.55) / 0.45
    unique_pitch_score = min(1.0, len(duration_by_pitch_class) / 4.0)
    motion_score = min(
        1.0,
        max(0, len(pitch_class_sequence) - 1) / max(1.0, analysis.duration * 0.70),
    )
    melodic_shape = 0.45 * unique_pitch_score + 0.55 * motion_score
    duration_agreement = min(total_seconds, analysis.duration) / max(
        total_seconds, analysis.duration
    )
    return duration_agreement * max(
        0.0,
        min(1.0, combined + 0.10 * melodic_shape - dominance_penalty),
    )


def render_analysis(
    analysis: TranscriptionAnalysis,
    *,
    title: str,
    settings: MelodySettings,
    bpm_override: Optional[float] = None,
    midi_path: Optional[Path] = None,
    diagnostics: Optional[dict[str, float]] = None,
) -> str:
    """Render cached model output into RTTTL using inexpensive settings."""
    estimated_bpm = bpm_override if bpm_override is not None else analysis.estimated_bpm
    rtttl_bpm = choose_rtttl_bpm(estimated_bpm, settings.strict_rtttl_bpm)
    decoder_mode = settings.decoder_mode.strip().lower()
    detected_events = 0
    if decoder_mode == "contour":
        note_grid, onset_grid = build_model_salience_grid(analysis, rtttl_bpm, settings.grid)
        path, onset_steps, onset_pitches = select_salience_contour(note_grid, onset_grid, settings)
        detected_events = sum(len(slot) for slot in onset_pitches)
        if midi_path:
            midi_data, _raw_events = extract_note_events(analysis, settings, rtttl_bpm)
            midi_data.write(str(midi_path))
    elif decoder_mode in ("events", "upper-events"):
        midi_data, raw_events = extract_note_events(analysis, settings, rtttl_bpm)
        if midi_path:
            midi_data.write(str(midi_path))
        events = [
            (float(start), float(end), int(pitch), float(confidence))
            for start, end, pitch, confidence, _pitch_bends in raw_events
            if end > start
            and confidence >= settings.min_confidence
            and (end - start) * 1000.0 >= settings.minimum_note_ms
        ]
        if not events:
            raise RuntimeError(
                "No notes were detected. Try a lower --min-confidence, "
                "a lower --onset-threshold, or an isolated vocal/lead stem."
            )
        detected_events = len(events)
        if decoder_mode == "upper-events":
            path, onset_steps, onset_pitches = select_upper_event_melody(
                events,
                rtttl_bpm,
                settings.grid,
                settings,
            )
        else:
            slots, onset_steps, onset_pitches = build_quantized_melody_grid(
                events, rtttl_bpm, settings.grid
            )
            path = select_monophonic_melody(
                slots,
                top_voice_bias=settings.top_voice_bias,
                jump_penalty=settings.jump_penalty,
                onset_pitches=onset_pitches,
                onset_change_bonus=settings.onset_change_bonus,
                onset_change_relief=settings.onset_change_relief,
                onset_same_pitch_bonus=settings.onset_same_pitch_bonus,
                sustain_bonus=settings.sustain_bonus,
                note_change_penalty=settings.note_change_penalty,
            )
    else:
        raise ValueError(f"Unsupported melody decoder mode: {settings.decoder_mode}")
    if not path or not any(pitch is not None for pitch in path):
        raise RuntimeError("No predominant melody contour survived the source evidence.")

    path = remove_one_step_blips(path, onset_pitches=onset_pitches)
    path = merge_short_note_runs(
        path,
        minimum_steps=settings.minimum_melody_run_steps,
        onset_pitches=onset_pitches,
    )
    # Preserve excerpt-relative timing for source/ringtone comparison. Basic
    # Pitch can end slightly early/late, so pad or clip to the selected grid.
    seconds_per_step = (240.0 / rtttl_bpm) / settings.grid
    excerpt_steps = max(1, round(analysis.duration / seconds_per_step))
    path = (path + [None] * max(0, excerpt_steps - len(path)))[:excerpt_steps]
    path = preserve_source_silence(path, analysis, rtttl_bpm, settings.grid)
    if not any(pitch is not None for pitch in path):
        raise RuntimeError("No melody remained outside the source's near-silent regions.")
    detected_path = path
    path = fit_to_rtttl_octaves(path)
    if settings.octave_smoothing:
        path = smooth_octave_contour(
            path,
            max_melody_span=settings.max_melody_span,
        )
    octave_shifts = [
        fitted - detected
        for detected, fitted in zip(detected_path, path, strict=False)
        if detected is not None and fitted is not None
    ]
    onset_pitch_shift = (
        int(round(float(np.median(octave_shifts)) / 12.0)) * 12 if octave_shifts else 0
    )
    runs = collapse_runs(
        path,
        onset_steps,
        repeated_note_min_steps=settings.repeated_note_min_steps,
        onset_pitches=onset_pitches,
        onset_pitch_shift=onset_pitch_shift,
        trim_silence=False,
    )

    rtttl = runs_to_rtttl(runs, title, rtttl_bpm, settings.grid)
    if diagnostics is not None:
        diagnostics["estimated_bpm"] = float(estimated_bpm)
        diagnostics["rtttl_bpm"] = float(rtttl_bpm)
        diagnostics["detected_events"] = float(detected_events)
        diagnostics["rtttl_tokens"] = float(rtttl.rsplit(":", 1)[-1].count(",") + 1)
        diagnostics["source_offset_seconds"] = 0.0
        diagnostics["duration_seconds"] = excerpt_steps * seconds_per_step
    return rtttl


def settings_from_args(args: argparse.Namespace) -> MelodySettings:
    """Build rendering settings from CLI-compatible namespace attributes."""

    return MelodySettings(
        grid=int(args.grid),
        min_confidence=float(args.min_confidence),
        onset_threshold=float(args.onset_threshold),
        frame_threshold=float(args.frame_threshold),
        minimum_note_ms=float(args.minimum_note_ms),
        top_voice_bias=float(args.top_voice_bias),
        jump_penalty=float(args.jump_penalty),
        note_change_penalty=float(
            getattr(args, "note_change_penalty", DEFAULT_NOTE_CHANGE_PENALTY)
        ),
        onset_change_bonus=float(getattr(args, "onset_change_bonus", DEFAULT_ONSET_CHANGE_BONUS)),
        onset_change_relief=float(
            getattr(args, "onset_change_relief", DEFAULT_ONSET_CHANGE_RELIEF)
        ),
        onset_same_pitch_bonus=float(
            getattr(args, "onset_same_pitch_bonus", DEFAULT_ONSET_SAME_PITCH_BONUS)
        ),
        sustain_bonus=float(getattr(args, "sustain_bonus", DEFAULT_SUSTAIN_BONUS)),
        repeated_note_min_steps=int(
            getattr(args, "repeated_note_min_steps", DEFAULT_REPEATED_NOTE_MIN_STEPS)
        ),
        minimum_melody_run_steps=int(
            getattr(args, "minimum_melody_run_steps", DEFAULT_MINIMUM_MELODY_RUN_STEPS)
        ),
        strict_rtttl_bpm=bool(getattr(args, "strict_rtttl_bpm", False)),
        octave_smoothing=bool(getattr(args, "octave_smoothing", DEFAULT_OCTAVE_SMOOTHING)),
        max_melody_span=int(getattr(args, "max_melody_span", DEFAULT_MAX_MELODY_SPAN)),
        decoder_mode=str(getattr(args, "decoder_mode", DEFAULT_DECODER_MODE)),
        contour_candidate_floor=float(
            getattr(args, "contour_candidate_floor", DEFAULT_CONTOUR_CANDIDATE_FLOOR)
        ),
        contour_relative_floor=float(
            getattr(args, "contour_relative_floor", DEFAULT_CONTOUR_RELATIVE_FLOOR)
        ),
        contour_max_candidates=int(
            getattr(args, "contour_max_candidates", DEFAULT_CONTOUR_MAX_CANDIDATES)
        ),
        contour_time_support=float(
            getattr(args, "contour_time_support", DEFAULT_CONTOUR_TIME_SUPPORT)
        ),
        octave_jump_penalty=float(
            getattr(args, "octave_jump_penalty", DEFAULT_OCTAVE_JUMP_PENALTY)
        ),
        upper_relative_floor=float(
            getattr(args, "upper_relative_floor", DEFAULT_UPPER_RELATIVE_FLOOR)
        ),
        upper_octave_guard_ratio=float(
            getattr(args, "upper_octave_guard_ratio", DEFAULT_UPPER_OCTAVE_GUARD_RATIO)
        ),
        upper_dropout_semitones=int(
            getattr(args, "upper_dropout_semitones", DEFAULT_UPPER_DROPOUT_SEMITONES)
        ),
        upper_dropout_max_steps=int(
            getattr(args, "upper_dropout_max_steps", DEFAULT_UPPER_DROPOUT_MAX_STEPS)
        ),
        upper_onset_relative_floor=float(
            getattr(
                args,
                "upper_onset_relative_floor",
                DEFAULT_UPPER_ONSET_RELATIVE_FLOOR,
            )
        ),
        upper_onset_max_semitones=int(
            getattr(args, "upper_onset_max_semitones", DEFAULT_UPPER_ONSET_MAX_SEMITONES)
        ),
    )


def convert(args: argparse.Namespace) -> str:
    """Analyze and convert one CLI namespace without writing the output file."""

    input_path = Path(args.input)
    analysis = analyze_audio(
        input_path,
        start=args.start,
        duration=args.duration,
        bpm_hint=args.bpm,
        source_mode=getattr(args, "source_mode", "harmonic"),
        progress=print,
    )
    diagnostics: dict[str, float] = {}
    rtttl = render_analysis(
        analysis,
        title=args.title or input_path.stem,
        settings=settings_from_args(args),
        bpm_override=args.bpm,
        midi_path=args.midi,
        diagnostics=diagnostics,
    )
    melody = rtttl.rsplit(":", 1)[-1]

    print(f"Estimated tempo: {diagnostics['estimated_bpm']:.1f} BPM")
    print(f"RTTTL tempo:     {int(diagnostics['rtttl_bpm'])} BPM")
    print(f"Detected events: {int(diagnostics['detected_events'])}")
    print(f"RTTTL tokens:    {melody.count(',') + 1}")
    return rtttl


def _finite_float_argument(value: str) -> float:
    """Parse a finite floating-point command-line value."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _positive_float_argument(value: str) -> float:
    parsed = _finite_float_argument(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float_argument(value: str) -> float:
    parsed = _finite_float_argument(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _probability_argument(value: str) -> float:
    parsed = _finite_float_argument(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _bpm_argument(value: str) -> float:
    parsed = _finite_float_argument(value)
    if not 25.0 <= parsed <= 900.0:
        raise argparse.ArgumentTypeError("must be between 25 and 900 BPM")
    return parsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line conversion options."""
    parser = argparse.ArgumentParser(
        description="Convert an MP3 excerpt into a monophonic Nokia RTTTL ringtone."
    )
    parser.add_argument("input", help="Input MP3/WAV/FLAC/M4A file")
    parser.add_argument("-o", "--output", default="ringtone.rtttl")
    parser.add_argument("--title", help="RTTTL name; sanitized and limited to 10 characters")
    parser.add_argument(
        "--start",
        type=_nonnegative_float_argument,
        default=0.0,
        help="Excerpt start in seconds",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float_argument,
        default=30.0,
        help="Excerpt length in seconds",
    )
    parser.add_argument(
        "--source-mode",
        choices=("harmonic", "full", "auto"),
        default="harmonic",
        help="Analyze harmonic content (including accompaniment), the full mix, or auto-detect.",
    )
    parser.add_argument("--bpm", type=_bpm_argument, help="Override automatic tempo detection")
    parser.add_argument("--grid", type=int, choices=(16, 32), default=DEFAULT_GRID)
    parser.add_argument(
        "--min-confidence", type=_probability_argument, default=DEFAULT_MIN_CONFIDENCE
    )
    parser.add_argument(
        "--onset-threshold", type=_probability_argument, default=DEFAULT_ONSET_THRESHOLD
    )
    parser.add_argument(
        "--frame-threshold", type=_probability_argument, default=DEFAULT_FRAME_THRESHOLD
    )
    parser.add_argument(
        "--minimum-note-ms", type=_positive_float_argument, default=DEFAULT_MINIMUM_NOTE_MS
    )
    parser.add_argument("--top-voice-bias", type=float, default=DEFAULT_TOP_VOICE_BIAS)
    parser.add_argument(
        "--jump-penalty", type=_nonnegative_float_argument, default=DEFAULT_JUMP_PENALTY
    )
    parser.add_argument(
        "--note-change-penalty",
        type=_nonnegative_float_argument,
        default=DEFAULT_NOTE_CHANGE_PENALTY,
        help="Penalty for changing pitch; higher values produce a simpler melody.",
    )
    parser.add_argument(
        "--onset-change-bonus",
        type=_nonnegative_float_argument,
        default=DEFAULT_ONSET_CHANGE_BONUS,
        help="Boost notes that begin on detected attacks; higher values follow the beat more closely.",
    )
    parser.add_argument(
        "--onset-change-relief",
        type=_nonnegative_float_argument,
        default=DEFAULT_ONSET_CHANGE_RELIEF,
        help="Reduce jump penalties when the destination pitch has a detected attack.",
    )
    parser.add_argument(
        "--onset-same-pitch-bonus",
        type=_nonnegative_float_argument,
        default=DEFAULT_ONSET_SAME_PITCH_BONUS,
        help="Sustain bonus on attack steps; lower values let new attacks replace older overlaps.",
    )
    parser.add_argument(
        "--sustain-bonus",
        type=_nonnegative_float_argument,
        default=DEFAULT_SUSTAIN_BONUS,
        help="Sustain bonus away from attacks; higher values make the melody smoother.",
    )
    parser.add_argument(
        "--max-melody-span", type=_positive_int_argument, default=DEFAULT_MAX_MELODY_SPAN
    )
    parser.add_argument(
        "--repeated-note-min-steps",
        type=_positive_int_argument,
        default=DEFAULT_REPEATED_NOTE_MIN_STEPS,
        help="Only split a repeated same-pitch onset after this many grid steps; reduces false stutter.",
    )
    parser.add_argument(
        "--minimum-melody-run-steps",
        type=_positive_int_argument,
        default=DEFAULT_MINIMUM_MELODY_RUN_STEPS,
        help="Merge pitch runs shorter than this many grid steps into nearby melody notes.",
    )
    parser.add_argument(
        "--strict-rtttl-bpm",
        action="store_true",
        help="Use the historical RTTTL preset tempo list instead of the closest integer BPM.",
    )
    parser.add_argument(
        "--no-octave-smoothing",
        dest="octave_smoothing",
        action="store_false",
        help="Disable contour-based octave smoothing for excerpts with intentional wide octave jumps.",
    )
    parser.set_defaults(octave_smoothing=DEFAULT_OCTAVE_SMOOTHING)
    parser.add_argument("--midi", type=Path, help="Optional debug MIDI output")
    return parser.parse_args(argv)


def main() -> None:
    """Run the command-line converter and write its RTTTL output."""
    args = parse_args()
    try:
        rtttl = convert(args)
    except FileNotFoundError as exc:
        raise SystemExit(f"error: input file not found: {exc}") from None
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    output_path = Path(args.output)
    try:
        output_path.write_text(rtttl + "\n", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"error: could not write {output_path}: {exc}") from None
    print(f"Wrote {output_path}")
    print(rtttl)


if __name__ == "__main__":
    main()
