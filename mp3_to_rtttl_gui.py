#!/usr/bin/env python3
"""PySide6 studio for generating, comparing, and editing RTTTL melodies."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import traceback
import wave
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import mp3_to_rtttl
from rtttl_utils import parse_rtttl_tokens, synthesize_rtttl

try:
    from PySide6.QtCore import (
        QObject,
        QPointF,
        QRectF,
        QRunnable,
        Qt,
        QThreadPool,
        QTimer,
        QUrl,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QTextCursor
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PySide6 is required for the desktop studio. Install dependencies with: "
        "python3 -m pip install -r requirements.txt"
    ) from exc


APPLICATION_NAME = "MP3 to RTTTL Studio"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
SAMPLE_RATE = 22_050
PREVIEW_SAMPLE_RATE = 2_000
MAX_CACHE_ITEMS = 4


@dataclass(frozen=True)
class LoadedAudio:
    """Lightweight waveform data used for selection and preview rendering."""

    path: Path
    duration: float
    waveform: np.ndarray


@dataclass(frozen=True)
class VariationProfile:
    """Named decoder settings representing one musical interpretation."""

    name: str
    character: str
    detail: float
    settings: mp3_to_rtttl.MelodySettings


@dataclass(frozen=True)
class VariationResult:
    """Rendered RTTTL plus evidence and readability metrics used for ranking."""

    profile: VariationProfile
    rtttl: str
    token_count: int
    notes_per_second: float
    span: int
    big_jumps: int
    average_jump: float
    match_score: float = 0.0


@dataclass(frozen=True)
class GenerationPayload:
    """Worker result tagged with its request and analysis-cache identity."""

    request_id: int
    cache_key: tuple
    analysis: mp3_to_rtttl.TranscriptionAnalysis
    results: tuple[VariationResult, ...]
    failures: tuple[str, ...]
    used_cache: bool


class WorkerSignals(QObject):
    """Qt signals used to return background-worker progress and results safely."""

    result = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    """Run one callable in Qt's thread pool and translate exceptions to signals."""

    def __init__(self, function: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(self.signals.progress.emit)
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.signals.error.emit(detail)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - minutes * 60:05.2f}"


def safe_file_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "variation"


def load_audio_preview(path: Path) -> LoadedAudio:
    samples = np.array([], dtype=np.float32)
    sr = PREVIEW_SAMPLE_RATE
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    str(PREVIEW_SAMPLE_RATE),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
            samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
        except (OSError, subprocess.SubprocessError):
            samples = np.array([], dtype=np.float32)

    if not samples.size:
        try:
            import librosa
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "librosa is missing. Install requirements.txt in this Python environment."
            ) from exc
        samples, sr = librosa.load(path, sr=PREVIEW_SAMPLE_RATE, mono=True)
    if samples.size == 0:
        raise RuntimeError("The audio file contains no samples.")

    duration = float(samples.size / sr)
    target_bins = 4_000
    if samples.size > target_bins:
        chunk = int(math.ceil(samples.size / target_bins))
        padded = np.pad(samples, (0, (-samples.size) % chunk))
        waveform = np.max(np.abs(padded.reshape(-1, chunk)), axis=1)
    else:
        waveform = np.abs(samples)

    peak = float(np.max(waveform)) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak
    return LoadedAudio(path=path.resolve(), duration=duration, waveform=waveform.astype(np.float32))


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class AudioPlayer(QObject):
    """Own media playback, bounded excerpt timing, and temporary audition WAVs."""

    positionChanged = Signal(float)
    playingChanged = Signal(bool)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.output = QAudioOutput(self)
        self.output.setVolume(0.82)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self._finish_interval)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.pending_position: Optional[float] = None
        self.interval_start = 0.0
        self.interval_end = 86_400.0
        self.temp_paths: list[Path] = []

    @property
    def has_media(self) -> bool:
        return not self.player.source().isEmpty()

    @property
    def is_playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def position(self) -> float:
        return self.player.position() / 1000.0

    def play_file(
        self,
        path: Path,
        start: float = 0.0,
        end: Optional[float] = None,
        position: Optional[float] = None,
    ) -> None:
        self.stop_timer.stop()
        start = max(0.0, float(start))
        end = max(start + 0.05, float(end)) if end is not None else start + 86_400.0
        self.interval_start = start
        self.interval_end = end
        self.pending_position = clamp(
            float(position) if position is not None else start,
            start,
            end,
        )
        source = QUrl.fromLocalFile(str(path.resolve()))
        if self.player.source() == source:
            self._start_pending()
        else:
            self.player.setSource(source)
            QTimer.singleShot(120, self._start_pending)

    def play_samples(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        handle, raw_path = tempfile.mkstemp(prefix="rtttl_studio_", suffix=".wav")
        os.close(handle)
        path = Path(raw_path)
        write_wav(path, samples, sample_rate)
        self.temp_paths.append(path)
        duration = max(0.05, float(np.asarray(samples).size / sample_rate))
        self.play_file(path, start=0.0, end=duration)
        return duration

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._start_pending()

    def _start_pending(self) -> None:
        if self.pending_position is None:
            return
        if self.player.mediaStatus() in (
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.LoadingMedia,
        ):
            return
        position = self.pending_position
        self.pending_position = None
        self.player.setPosition(int(position * 1000))
        self.player.play()
        self._arm_stop_timer()

    def _arm_stop_timer(self) -> None:
        self.stop_timer.stop()
        remaining_ms = max(1, int((self.interval_end - self.position) * 1000))
        if remaining_ms < 86_000_000:
            self.stop_timer.start(remaining_ms)

    def _position_changed(self, milliseconds: int) -> None:
        position = milliseconds / 1000.0
        self.positionChanged.emit(position)
        if self.is_playing and position >= self.interval_end - 0.01:
            self._finish_interval()

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.playingChanged.emit(state == QMediaPlayer.PlaybackState.PlayingState)

    def _finish_interval(self) -> None:
        self.stop_timer.stop()
        self.player.pause()
        self.player.setPosition(int(self.interval_end * 1000))
        self.positionChanged.emit(self.interval_end)

    def toggle_pause(self) -> bool:
        if not self.has_media:
            return False
        if self.is_playing:
            self.stop_timer.stop()
            self.player.pause()
            return False
        if self.position < self.interval_start or self.position >= self.interval_end - 0.02:
            self.player.setPosition(int(self.interval_start * 1000))
        self.player.play()
        self._arm_stop_timer()
        return True

    def seek_to(self, seconds: float) -> float:
        position = clamp(float(seconds), self.interval_start, self.interval_end)
        if position >= self.interval_end - 0.001 and self.is_playing:
            self.player.pause()
        self.player.setPosition(int(position * 1000))
        self.positionChanged.emit(position)
        if self.is_playing:
            self._arm_stop_timer()
        else:
            self.stop_timer.stop()
        return position

    def seek_relative(self, seconds: float) -> float:
        return self.seek_to(self.position + float(seconds))

    def stop(self) -> None:
        self.pending_position = None
        self.stop_timer.stop()
        if self.has_media:
            self.player.pause()
            self.player.setPosition(int(self.interval_start * 1000))
            self.positionChanged.emit(self.interval_start)

    def cleanup(self) -> None:
        self.pending_position = None
        self.stop_timer.stop()
        self.player.stop()
        for path in self.temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.temp_paths.clear()


class WaveformWidget(QWidget):
    """Interactive source waveform with draggable selection and playhead."""

    selectionChanged = Signal(float, float)
    selectionCommitted = Signal(float, float)
    playheadChanged = Signal(float)
    playheadCommitted = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.duration = 0.0
        self.waveform = np.array([], dtype=np.float32)
        self.start = 0.0
        self.end = 0.0
        self.playhead = 0.0
        self.drag_mode: Optional[str] = None
        self.drag_anchor = 0.0
        self.drag_original = (0.0, 0.0)

    def set_audio(self, duration: float, waveform: np.ndarray) -> None:
        self.duration = max(0.0, float(duration))
        self.waveform = np.asarray(waveform, dtype=np.float32)
        self.start = 0.0
        self.end = min(30.0, self.duration)
        self.playhead = 0.0
        self.update()
        self.selectionChanged.emit(self.start, self.end)

    def set_selection(self, start: float, end: float, commit: bool = False) -> None:
        if self.duration <= 0:
            return
        start = clamp(float(start), 0.0, self.duration)
        end = clamp(float(end), 0.0, self.duration)
        if end < start:
            start, end = end, start
        if end - start < 0.1:
            end = min(self.duration, start + 0.1)
            start = max(0.0, end - 0.1)
        self.start, self.end = start, end
        self.update()
        self.selectionChanged.emit(start, end)
        if commit:
            self.selectionCommitted.emit(start, end)

    def set_playhead(self, value: float, notify: bool = False) -> None:
        self.playhead = clamp(float(value), 0.0, self.duration) if self.duration else 0.0
        self.update()
        if notify:
            self.playheadChanged.emit(self.playhead)

    def _time_to_x(self, value: float) -> float:
        usable = max(1.0, self.width() - 24.0)
        return 12.0 + usable * value / max(0.001, self.duration)

    def _x_to_time(self, value: float) -> float:
        usable = max(1.0, self.width() - 24.0)
        return clamp((value - 12.0) / usable * self.duration, 0.0, self.duration)

    def _draw_waveform(self, painter: QPainter, color: QColor) -> None:
        if not self.waveform.size:
            return
        left = 12.0
        usable = max(1, self.width() - 24)
        middle = self.height() * 0.53
        amplitude_height = self.height() * 0.34
        count = min(usable, self.waveform.size)
        indices = np.linspace(0, self.waveform.size - 1, count, dtype=int)
        painter.setPen(QPen(color, 1.2))
        for x_index, sample_index in enumerate(indices):
            amplitude = float(self.waveform[sample_index]) * amplitude_height
            x = left + x_index * usable / max(1, count - 1)
            painter.drawLine(QPointF(x, middle - amplitude), QPointF(x, middle + amplitude))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(QColor("#2b3735"), 1.0))
        painter.setBrush(QColor("#0b1112"))
        painter.drawRoundedRect(bounds, 14.0, 14.0)

        painter.setPen(QPen(QColor("#1d2928"), 1.0))
        for index in range(1, 8):
            x = self.width() * index / 8.0
            painter.drawLine(QPointF(x, 18.0), QPointF(x, self.height() - 18.0))

        self._draw_waveform(painter, QColor("#49625e"))
        if self.duration > 0:
            start_x = self._time_to_x(self.start)
            end_x = self._time_to_x(self.end)
            selection = QRectF(start_x, 1.0, max(1.0, end_x - start_x), self.height() - 2.0)
            painter.fillRect(selection, QColor(244, 166, 58, 24))
            painter.save()
            painter.setClipRect(selection)
            self._draw_waveform(painter, QColor("#f4a63a"))
            painter.restore()

            painter.setPen(QPen(QColor("#f4a63a"), 3.0))
            painter.drawLine(QPointF(start_x, 9.0), QPointF(start_x, self.height() - 9.0))
            painter.drawLine(QPointF(end_x, 9.0), QPointF(end_x, self.height() - 9.0))
            painter.setPen(QColor("#e7eee7"))
            painter.drawText(QPointF(14.0, 20.0), format_time(self.start))
            end_label = format_time(self.end)
            painter.drawText(QPointF(self.width() - 68.0, 20.0), end_label)

            playhead_x = self._time_to_x(self.playhead)
            playhead_color = QColor("#fff0b3")
            painter.setPen(QPen(playhead_color, 2.4))
            painter.drawLine(QPointF(playhead_x, 7.0), QPointF(playhead_x, self.height() - 7.0))
            painter.setBrush(playhead_color)
            painter.drawEllipse(QPointF(playhead_x, 10.0), 4.5, 4.5)
            label = format_time(self.playhead)
            label_width = painter.fontMetrics().horizontalAdvance(label) + 14.0
            label_left = clamp(playhead_x - label_width / 2, 6.0, self.width() - label_width - 6.0)
            label_rect = QRectF(label_left, self.height() - 28.0, label_width, 21.0)
            painter.setPen(QPen(QColor("#53615d"), 1.0))
            painter.setBrush(QColor(8, 13, 14, 225))
            painter.drawRoundedRect(label_rect, 6.0, 6.0)
            painter.setPen(playhead_color)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event) -> None:
        if self.duration <= 0 or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        x = event.position().x()
        start_x = self._time_to_x(self.start)
        end_x = self._time_to_x(self.end)
        if abs(x - start_x) <= 12:
            self.drag_mode = "start"
        elif abs(x - end_x) <= 12:
            self.drag_mode = "end"
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier and start_x < x < end_x:
            self.drag_mode = "move"
            self.drag_anchor = self._x_to_time(x)
            self.drag_original = (self.start, self.end)
        else:
            self.drag_mode = "playhead"
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self.drag_mode:
            return
        value = self._x_to_time(event.position().x())
        if self.drag_mode == "start":
            self.set_selection(min(value, self.end - 0.1), self.end)
        elif self.drag_mode == "end":
            self.set_selection(self.start, max(value, self.start + 0.1))
        elif self.drag_mode == "move":
            original_start, original_end = self.drag_original
            width = original_end - original_start
            delta = value - self.drag_anchor
            start = clamp(original_start + delta, 0.0, max(0.0, self.duration - width))
            self.set_selection(start, start + width)
        else:
            self.set_playhead(value, notify=True)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.drag_mode:
            drag_mode = self.drag_mode
            self.drag_mode = None
            if drag_mode == "playhead":
                self.playheadCommitted.emit(self.playhead)
            else:
                self.selectionCommitted.emit(self.start, self.end)


class PianoRollWidget(QWidget):
    """Display source salience and support direct pitch/duration token edits."""

    tokenSelected = Signal(int)
    pitchChangeRequested = Signal(int, int)
    durationChangeRequested = Signal(int, int)
    restToggleRequested = Signal(int)
    editorCommand = Signal(str)

    MIDI_BASE = 21
    PITCH_LOW = 36
    PITCH_HIGH = 96

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(290)
        self.analysis: Optional[mp3_to_rtttl.TranscriptionAnalysis] = None
        self.rtttl = ""
        self.heatmap: Optional[QImage] = None
        self.active_token_index: Optional[int] = None
        self.selected_token_index: Optional[int] = None
        self.grid = 16
        self.token_rects: dict[int, QRectF] = {}
        self.token_x_ranges: list[tuple[float, float]] = []
        self.token_steps: list[int] = []
        self.plot_rect = QRectF()
        self.octave_shift = 0
        self.magnetic_snap_enabled = True
        self.editor_drag_mode: Optional[str] = None
        self.editor_drag_token: Optional[int] = None
        self.editor_drag_start = QPointF()
        self.editor_drag_position = QPointF()

    def clear(self) -> None:
        self.analysis = None
        self.rtttl = ""
        self.heatmap = None
        self.active_token_index = None
        self.selected_token_index = None
        self.token_rects.clear()
        self.token_x_ranges.clear()
        self.token_steps.clear()
        self.update()

    def set_comparison(
        self,
        analysis: mp3_to_rtttl.TranscriptionAnalysis,
        rtttl: str,
        grid: int = 16,
    ) -> None:
        if analysis is not self.analysis:
            self.analysis = analysis
            self.heatmap = self._build_heatmap(analysis)
        self.rtttl = rtttl
        self.grid = grid
        self.active_token_index = None
        self.selected_token_index = None
        self.update()

    def set_active_token(self, token_index: Optional[int]) -> None:
        if token_index == self.active_token_index:
            return
        self.active_token_index = token_index
        self.update()

    def set_selected_token(self, token_index: Optional[int]) -> None:
        if token_index == self.selected_token_index:
            return
        self.selected_token_index = token_index
        self.update()

    def set_magnetic_snap(self, enabled: bool) -> None:
        self.magnetic_snap_enabled = bool(enabled)

    def _snap_pitch_to_analysis(self, token_index: int, requested_midi: int) -> int:
        if (
            not self.magnetic_snap_enabled
            or self.analysis is None
            or not 0 <= token_index < len(self.token_x_ranges)
            or self.plot_rect.width() <= 0
        ):
            return requested_midi

        notes = np.asarray(self.analysis.model_output.get("note"), dtype=np.float32)
        onsets = np.asarray(self.analysis.model_output.get("onset"), dtype=np.float32)
        if notes.ndim != 2 or notes.shape[0] == 0:
            return requested_midi

        start_x, end_x = self.token_x_ranges[token_index]
        start_ratio = clamp(
            (start_x - self.plot_rect.left()) / self.plot_rect.width(),
            0.0,
            1.0,
        )
        end_ratio = clamp(
            (end_x - self.plot_rect.left()) / self.plot_rect.width(),
            start_ratio,
            1.0,
        )
        first_frame = max(0, int(math.floor(start_ratio * notes.shape[0])) - 1)
        final_frame = min(
            notes.shape[0],
            max(first_frame + 1, int(math.ceil(end_ratio * notes.shape[0])) + 1),
        )
        note_window = notes[first_frame:final_frame]
        strengths = 0.72 * np.max(note_window, axis=0) + 0.28 * np.mean(note_window, axis=0)
        if onsets.shape == notes.shape:
            strengths += 0.28 * np.max(onsets[first_frame:final_frame], axis=0)

        low_index = max(0, self.PITCH_LOW - self.MIDI_BASE)
        high_index = min(notes.shape[1] - 1, self.PITCH_HIGH - self.MIDI_BASE)
        if high_index < low_index:
            return requested_midi
        visible = strengths[low_index : high_index + 1]
        peak = float(np.max(visible)) if visible.size else 0.0
        if peak < 0.10:
            return requested_midi

        threshold = max(0.10, peak * 0.32)
        candidates: list[tuple[int, float]] = []
        for local_index, strength in enumerate(visible):
            if float(strength) < threshold:
                continue
            left = float(visible[local_index - 1]) if local_index > 0 else -1.0
            right = float(visible[local_index + 1]) if local_index + 1 < visible.size else -1.0
            if float(strength) >= left and float(strength) >= right:
                candidates.append((self.MIDI_BASE + low_index + local_index, float(strength)))
        if not candidates:
            return requested_midi

        requested_display_midi = requested_midi + self.octave_shift
        snapped_display_midi, _strength = min(
            candidates,
            key=lambda candidate: (
                abs(candidate[0] - requested_display_midi),
                -candidate[1],
            ),
        )
        snapped_midi = snapped_display_midi - self.octave_shift
        while snapped_midi < mp3_to_rtttl.RTTTL_MIN_MIDI:
            snapped_midi += 12
        while snapped_midi > mp3_to_rtttl.RTTTL_MAX_MIDI:
            snapped_midi -= 12
        return snapped_midi

    def _build_heatmap(self, analysis: mp3_to_rtttl.TranscriptionAnalysis) -> QImage:
        notes = np.asarray(analysis.model_output.get("note"), dtype=np.float32)
        onsets = np.asarray(analysis.model_output.get("onset"), dtype=np.float32)
        if notes.ndim != 2 or onsets.shape != notes.shape:
            return QImage()

        low_index = max(0, self.PITCH_LOW - self.MIDI_BASE)
        high_index = min(notes.shape[1] - 1, self.PITCH_HIGH - self.MIDI_BASE)
        note_view = notes[:, low_index : high_index + 1].T[::-1]
        onset_view = onsets[:, low_index : high_index + 1].T[::-1]
        note_strength = np.clip((note_view - 0.08) / 0.70, 0.0, 1.0)
        onset_strength = np.clip((onset_view - 0.10) / 0.72, 0.0, 1.0)

        height, width = note_strength.shape
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = np.clip(10 + 26 * note_strength + 225 * onset_strength, 0, 255)
        pixels[..., 1] = np.clip(16 + 142 * note_strength + 78 * onset_strength, 0, 255)
        pixels[..., 2] = np.clip(18 + 132 * note_strength + 12 * onset_strength, 0, 255)
        pixels = np.ascontiguousarray(pixels)
        return QImage(
            pixels.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _pitch_y(self, midi: int, plot: QRectF) -> float:
        rows = self.PITCH_HIGH - self.PITCH_LOW + 1
        position = (midi - self.PITCH_LOW + 0.5) / rows
        return plot.bottom() - position * plot.height()

    def _best_octave_shift(self, tokens) -> int:
        if self.analysis is None or not tokens:
            return 0
        notes = np.asarray(self.analysis.model_output.get("note"), dtype=np.float32)
        if notes.ndim != 2:
            return 0
        total_seconds = sum(token.seconds for token in tokens) or 1.0
        best_shift = 0
        best_score = -float("inf")
        for shift in range(-48, 49, 12):
            cursor = 0.0
            weighted_score = 0.0
            weight = 0.0
            for token in tokens:
                midpoint = cursor + token.seconds / 2
                cursor += token.seconds
                if token.midi is None:
                    continue
                model_index = token.midi + shift - self.MIDI_BASE
                if not 0 <= model_index < notes.shape[1]:
                    continue
                frame = min(
                    notes.shape[0] - 1,
                    max(0, round(midpoint / total_seconds * (notes.shape[0] - 1))),
                )
                weighted_score += float(notes[frame, model_index]) * token.seconds
                weight += token.seconds
            score = weighted_score / max(weight, 1e-9) - 0.0005 * abs(shift)
            if score > best_score:
                best_score = score
                best_shift = shift
        return best_shift

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(QColor("#2b3735"), 1.0))
        painter.setBrush(QColor("#0b1112"))
        painter.drawRoundedRect(bounds, 12.0, 12.0)

        if self.analysis is None or self.heatmap is None or self.heatmap.isNull():
            painter.setPen(QColor("#899795"))
            painter.drawText(
                bounds, Qt.AlignmentFlag.AlignCenter, "Generate versions to see analyzed notes"
            )
            return

        plot = QRectF(48.0, 32.0, max(20.0, self.width() - 62.0), max(20.0, self.height() - 60.0))
        self.plot_rect = plot
        painter.drawImage(plot, self.heatmap)
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1.0))
        duration = max(0.001, self.analysis.duration)
        for second in range(0, int(math.ceil(duration)) + 1):
            x = plot.left() + plot.width() * second / duration
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor("#9aa9a5"))
            painter.drawText(QPointF(x + 3.0, plot.bottom() + 17.0), f"{second}s")
            painter.setPen(QPen(QColor(255, 255, 255, 35), 1.0))

        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.0))
        for midi in range(36, 97, 12):
            y = self._pitch_y(midi, plot)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#aab6b2"))
            painter.drawText(QPointF(10.0, y + 4.0), f"C{midi // 12 - 1}")
            painter.setPen(QPen(QColor(255, 255, 255, 45), 1.0))

        try:
            tokens = parse_rtttl_tokens(self.rtttl, grid=self.grid)
        except (TypeError, ValueError):
            tokens = []
        octave_shift = self._best_octave_shift(tokens)
        self.octave_shift = octave_shift
        total_seconds = sum(token.seconds for token in tokens) or 1.0
        cursor = 0.0
        self.token_rects.clear()
        self.token_x_ranges.clear()
        self.token_steps = [token.steps for token in tokens]
        row_height = max(5.0, plot.height() / (self.PITCH_HIGH - self.PITCH_LOW + 1) * 0.82)
        for token_index, token in enumerate(tokens):
            start_x = plot.left() + plot.width() * cursor / total_seconds
            cursor += token.seconds
            end_x = plot.left() + plot.width() * cursor / total_seconds
            self.token_x_ranges.append((start_x, end_x))
            display_midi = token.midi + octave_shift if token.midi is not None else None
            if display_midi is None or not self.PITCH_LOW <= display_midi <= self.PITCH_HIGH:
                if token_index == self.selected_token_index:
                    rest_rect = QRectF(start_x, plot.bottom() - 6.0, max(3.0, end_x - start_x), 5.0)
                    painter.setPen(QPen(QColor("#4db8ff"), 2.2))
                    painter.setBrush(QColor(77, 184, 255, 70))
                    painter.drawRoundedRect(rest_rect, 2.0, 2.0)
                continue
            y = self._pitch_y(display_midi, plot)
            note_rect = QRectF(start_x, y - row_height / 2, max(2.0, end_x - start_x), row_height)
            self.token_rects[token_index] = note_rect
            if token_index == self.active_token_index:
                painter.setPen(QPen(QColor("#63ffd7"), 3.4))
            elif token_index == self.selected_token_index:
                painter.setPen(QPen(QColor("#4db8ff"), 2.8))
            else:
                painter.setPen(QPen(QColor("#fff6df"), 1.8))
            painter.setBrush(QColor(244, 166, 58, 150))
            painter.drawRoundedRect(note_rect, 2.0, 2.0)
            if token_index == self.selected_token_index:
                handle_color = (
                    QColor("#63ffd7")
                    if token_index == self.active_token_index
                    else QColor("#4db8ff")
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(handle_color)
                painter.drawRect(
                    QRectF(note_rect.right() - 3.0, note_rect.center().y() - 5.0, 6.0, 10.0)
                )

        if self.editor_drag_mode and self.plot_rect.contains(self.editor_drag_position):
            painter.setPen(QPen(QColor("#4db8ff"), 2.0, Qt.PenStyle.DashLine))
            if self.editor_drag_mode == "resize":
                x = clamp(self.editor_drag_position.x(), plot.left(), plot.right())
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            else:
                y = clamp(self.editor_drag_position.y(), plot.top(), plot.bottom())
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        painter.setPen(QColor("#68c8b4"))
        painter.drawText(QPointF(52.0, 21.0), "TEAL: analyzed notes")
        painter.setPen(QColor("#f08c36"))
        painter.drawText(QPointF(205.0, 21.0), "ORANGE: detected onsets")
        painter.setPen(QColor("#fff6df"))
        painter.drawText(QPointF(390.0, 21.0), "WHITE: RTTTL | BLUE: selected | MINT: playing")
        painter.setPen(QColor("#9fb0ab"))
        painter.drawText(
            QPointF(max(660.0, self.width() - 250.0), 21.0),
            f"source: {self.analysis.source_mode} | overlay shift: {octave_shift:+d}",
        )

    def _token_at_x(self, x: float) -> Optional[int]:
        for index, (start_x, end_x) in enumerate(self.token_x_ranges):
            if start_x <= x <= end_x:
                return index
        return None

    def _actual_pitch_at_y(self, y: float) -> int:
        rows = self.PITCH_HIGH - self.PITCH_LOW + 1
        position = clamp(
            (self.plot_rect.bottom() - y) / max(1.0, self.plot_rect.height()), 0.0, 0.999999
        )
        display_midi = self.PITCH_LOW + int(position * rows)
        midi = display_midi - self.octave_shift
        while midi < mp3_to_rtttl.RTTTL_MIN_MIDI:
            midi += 12
        while midi > mp3_to_rtttl.RTTTL_MAX_MIDI:
            midi -= 12
        return midi

    def mousePressEvent(self, event) -> None:
        if not self.token_x_ranges or not self.plot_rect.contains(event.position()):
            super().mousePressEvent(event)
            return
        self.setFocus()
        token_index = self._token_at_x(event.position().x())
        if token_index is None:
            return
        self.tokenSelected.emit(token_index)
        self.set_selected_token(token_index)
        if event.button() == Qt.MouseButton.RightButton:
            self.restToggleRequested.emit(token_index)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        self.editor_drag_token = token_index
        self.editor_drag_start = event.position()
        self.editor_drag_position = event.position()
        note_rect = self.token_rects.get(token_index)
        if (
            token_index == self.selected_token_index
            and note_rect is not None
            and abs(event.position().x() - note_rect.right()) <= 8.0
            and abs(event.position().y() - note_rect.center().y()) <= 14.0
        ):
            self.editor_drag_mode = "resize"
        elif note_rect is not None and note_rect.adjusted(-3.0, -6.0, 3.0, 6.0).contains(
            event.position()
        ):
            self.editor_drag_mode = "pitch"
        else:
            self.editor_drag_mode = "replace"
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.editor_drag_mode is None:
            super().mouseMoveEvent(event)
            return
        self.editor_drag_position = event.position()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.editor_drag_mode is None or self.editor_drag_token is None:
            super().mouseReleaseEvent(event)
            return
        drag_mode = self.editor_drag_mode
        token_index = self.editor_drag_token
        moved = (event.position() - self.editor_drag_start).manhattanLength()
        self.editor_drag_mode = None
        self.editor_drag_token = None
        self.editor_drag_position = event.position()
        self.update()

        if drag_mode == "resize" and self.token_steps:
            total_steps = sum(self.token_steps)
            start_step = sum(self.token_steps[:token_index])
            ratio = clamp(
                (event.position().x() - self.plot_rect.left()) / max(1.0, self.plot_rect.width()),
                0.0,
                1.0,
            )
            target_step = round(ratio * total_steps)
            new_steps = max(1, target_step - start_step)
            self.durationChangeRequested.emit(
                token_index, new_steps - self.token_steps[token_index]
            )
        elif drag_mode == "replace" or (drag_mode == "pitch" and moved >= 4.0):
            requested_midi = self._actual_pitch_at_y(event.position().y())
            self.pitchChangeRequested.emit(
                token_index,
                self._snap_pitch_to_analysis(token_index, requested_midi),
            )
        event.accept()

    def keyPressEvent(self, event) -> None:
        modifiers = event.modifiers()
        command = modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        )
        if command and event.key() == Qt.Key.Key_Z:
            self.editorCommand.emit(
                "redo" if modifiers & Qt.KeyboardModifier.ShiftModifier else "undo"
            )
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                command_name = "first" if event.key() == Qt.Key.Key_Left else "last"
            else:
                command_name = "previous" if event.key() == Qt.Key.Key_Left else "next"
            self.editorCommand.emit(command_name)
            event.accept()
            return
        key_commands = {
            Qt.Key.Key_Minus: "shorter",
            Qt.Key.Key_Equal: "longer",
            Qt.Key.Key_Plus: "longer",
            Qt.Key.Key_Delete: "rest",
            Qt.Key.Key_Backspace: "rest",
            Qt.Key.Key_R: "rest",
            Qt.Key.Key_Space: "play_pause",
        }
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            direction = 1 if event.key() == Qt.Key.Key_Up else -1
            amount = 12 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            self.editorCommand.emit(f"transpose:{direction * amount}")
            event.accept()
            return
        command_name = key_commands.get(event.key())
        if command_name:
            self.editorCommand.emit(command_name)
            event.accept()
            return
        super().keyPressEvent(event)


def _evenly_spaced_profiles(
    profiles: list[VariationProfile],
    limit: int,
) -> tuple[VariationProfile, ...]:
    """Sample an ordered profile family without favoring one end of its range."""
    if limit <= 0 or not profiles:
        return ()
    if limit >= len(profiles):
        return tuple(profiles)
    if limit == 1:
        return (profiles[0],)

    last_index = len(profiles) - 1
    return tuple(profiles[round(sample * last_index / (limit - 1))] for sample in range(limit))


def _balanced_profile_pool(
    profiles: list[VariationProfile],
    requested_count: int,
) -> tuple[VariationProfile, ...]:
    """Return an oversampled, family-balanced pool for final evidence ranking."""
    families = ("contour", "upper-events", "events")
    # Roughly two candidates per requested result leaves useful ranking choice
    # without paying to render the former fixed 43-profile pool every time.
    per_family_limit = max(1, math.ceil(max(1, requested_count) * 2 / len(families)))
    selected: list[VariationProfile] = []
    for family in families:
        family_profiles = [
            profile
            for profile in profiles
            if profile.settings.decoder_mode.strip().lower() == family
        ]
        selected.extend(_evenly_spaced_profiles(family_profiles, per_family_limit))
    return tuple(selected)


def variation_profiles(
    *,
    center_detail: float,
    smoothness: float,
    lead_preference: float,
    count: int,
    spread: float,
    grid: int,
) -> tuple[VariationProfile, ...]:
    """Build a bounded candidate pool spanning all three melody decoders."""
    archetypes = (
        (
            "Contour match",
            "Balanced lead contour",
            0.00,
            0.00,
            0.00,
            1.00,
            1.00,
            1.00,
            0,
            0.0,
            0.000,
            0.00,
        ),
        (
            "Smooth lead",
            "Long singable phrases",
            -0.18,
            0.22,
            0.00,
            0.82,
            1.18,
            1.24,
            -2,
            0.8,
            0.015,
            0.05,
        ),
        (
            "Phrase lead",
            "Stable phrase movement",
            -0.10,
            0.12,
            -0.04,
            0.90,
            1.10,
            1.12,
            -1,
            0.5,
            0.008,
            0.03,
        ),
        (
            "Clear attacks",
            "Source-aligned note attacks",
            0.18,
            -0.08,
            0.00,
            1.28,
            0.84,
            0.90,
            1,
            -0.5,
            -0.012,
            -0.04,
        ),
        (
            "Rhythmic lead",
            "Fast attacked melody",
            0.34,
            -0.18,
            0.03,
            1.42,
            0.72,
            0.76,
            2,
            -0.8,
            -0.022,
            -0.07,
        ),
        (
            "Upper lead",
            "Higher predominant voice",
            0.02,
            0.00,
            0.30,
            1.00,
            1.00,
            1.00,
            0,
            0.0,
            0.000,
            0.00,
        ),
        (
            "Mid lead",
            "Central melody register",
            0.02,
            0.05,
            -0.18,
            1.00,
            1.05,
            1.05,
            0,
            0.0,
            0.000,
            0.01,
        ),
        (
            "Lower hook",
            "Lower coherent motif",
            -0.02,
            0.08,
            -0.38,
            0.95,
            1.08,
            1.08,
            0,
            0.2,
            0.004,
            0.02,
        ),
        (
            "Compact hook",
            "Tight ringtone range",
            -0.28,
            0.28,
            -0.08,
            0.74,
            1.25,
            1.34,
            -3,
            1.2,
            0.022,
            0.08,
        ),
        (
            "Repeated motif",
            "Preserved repeated attacks",
            0.12,
            -0.02,
            0.00,
            1.18,
            0.90,
            0.94,
            0,
            -0.7,
            -0.006,
            -0.02,
        ),
        (
            "Open contour",
            "More source candidates",
            0.28,
            -0.10,
            -0.02,
            1.15,
            0.86,
            0.86,
            2,
            -0.5,
            -0.022,
            -0.08,
        ),
        (
            "Strict contour",
            "Only strong continuous notes",
            -0.20,
            0.10,
            0.05,
            0.84,
            1.12,
            1.18,
            -2,
            0.8,
            0.028,
            0.08,
        ),
        (
            "Agile contour",
            "Detailed melodic movement",
            0.42,
            -0.22,
            0.08,
            1.34,
            0.70,
            0.70,
            3,
            -1.0,
            -0.030,
            -0.10,
        ),
    )
    candidate_grids = (grid, 32 if grid == 16 else 16)
    profiles: list[VariationProfile] = []

    for candidate_grid in candidate_grids:
        grid_label = (
            "" if candidate_grid == grid else (" fine" if candidate_grid > grid else " coarse")
        )
        step_scale = candidate_grid / 16.0
        for (
            name,
            character,
            detail_offset,
            smooth_offset,
            lead_offset,
            onset_scale,
            sustain_scale,
            jump_scale,
            span_adjust,
            run_adjust,
            floor_adjust,
            relative_adjust,
        ) in archetypes:
            detail = clamp(center_detail + detail_offset * spread, 0.0, 1.0)
            local_smoothness = clamp(smoothness + smooth_offset, 0.0, 1.0)
            local_lead = clamp(lead_preference + lead_offset, 0.0, 1.0)
            continuity = local_smoothness - 0.5
            settings = mp3_to_rtttl.MelodySettings(
                grid=candidate_grid,
                min_confidence=lerp(0.48, 0.22, detail),
                onset_threshold=lerp(0.52, 0.30, detail),
                frame_threshold=lerp(0.32, 0.18, detail),
                minimum_note_ms=lerp(180.0, 60.0, detail),
                top_voice_bias=lerp(-0.012, 0.052, local_lead),
                jump_penalty=clamp(
                    (lerp(0.155, 0.075, detail) + continuity * 0.045) * jump_scale,
                    0.045,
                    0.24,
                ),
                note_change_penalty=clamp(
                    lerp(0.34, 0.12, detail) + continuity * 0.10,
                    0.08,
                    0.48,
                ),
                onset_change_bonus=clamp(
                    (lerp(0.46, 0.86, detail) - continuity * 0.08) * onset_scale,
                    0.28,
                    1.20,
                ),
                onset_change_relief=lerp(0.18, 0.36, detail),
                onset_same_pitch_bonus=lerp(0.10, 0.18, detail),
                sustain_bonus=clamp(
                    (lerp(0.50, 0.30, detail) + continuity * 0.18) * sustain_scale,
                    0.18,
                    0.78,
                ),
                repeated_note_min_steps=max(2, round(lerp(4.0, 2.0, detail) * step_scale)),
                minimum_melody_run_steps=max(
                    1, round((lerp(3.5, 1.5, detail) + run_adjust) * step_scale)
                ),
                octave_smoothing=True,
                max_melody_span=max(12, round(lerp(13, 17, detail) + span_adjust)),
                decoder_mode="contour",
                contour_candidate_floor=clamp(lerp(0.14, 0.07, detail) + floor_adjust, 0.045, 0.20),
                contour_relative_floor=clamp(
                    lerp(0.40, 0.23, detail) + relative_adjust, 0.16, 0.55
                ),
                contour_max_candidates=max(4, round(lerp(6, 11, detail))),
                contour_time_support=clamp(
                    lerp(0.28, 0.16, detail) + continuity * 0.04, 0.10, 0.34
                ),
                octave_jump_penalty=clamp(0.15 * jump_scale, 0.07, 0.24),
            )
            profiles.append(
                VariationProfile(
                    f"{name}{grid_label}",
                    f"{character} | {candidate_grid}th grid",
                    detail,
                    settings,
                )
            )

    upper_step_scale = grid / 16.0
    upper_base = mp3_to_rtttl.MelodySettings(
        grid=grid,
        min_confidence=0.22,
        onset_threshold=0.28,
        frame_threshold=0.16,
        minimum_note_ms=50.0,
        top_voice_bias=0.0,
        jump_penalty=0.0,
        note_change_penalty=0.0,
        onset_change_bonus=0.0,
        onset_change_relief=0.0,
        onset_same_pitch_bonus=0.0,
        sustain_bonus=0.0,
        repeated_note_min_steps=max(2, round(2 * upper_step_scale)),
        minimum_melody_run_steps=max(1, round(upper_step_scale)),
        octave_smoothing=False,
        max_melody_span=36,
        decoder_mode="upper-events",
        upper_relative_floor=0.40,
        upper_octave_guard_ratio=1.20,
        upper_dropout_semitones=14,
        upper_dropout_max_steps=max(4, round(10 * upper_step_scale)),
        upper_onset_relative_floor=0.75,
        upper_onset_max_semitones=8,
    )
    upper_profiles = (
        ("Upper lead isolate", "Highest credible voice with dropout rests", upper_base),
        (
            "Upper lead turns",
            "Upper voice with stronger attacked turns",
            replace(upper_base, upper_onset_relative_floor=0.45),
        ),
        (
            "Upper lead focused",
            "Stricter upper event envelope",
            replace(
                upper_base,
                upper_relative_floor=0.50,
                upper_octave_guard_ratio=1.40,
                upper_dropout_max_steps=max(3, round(6 * upper_step_scale)),
            ),
        ),
        (
            "Upper lead focused turns",
            "Focused envelope with more attacked turns",
            replace(
                upper_base,
                upper_relative_floor=0.50,
                upper_octave_guard_ratio=1.40,
                upper_dropout_max_steps=max(3, round(6 * upper_step_scale)),
                upper_onset_relative_floor=0.45,
            ),
        ),
        (
            "Upper lead bright",
            "Upper envelope with stronger harmonic rejection",
            replace(upper_base, upper_octave_guard_ratio=1.40),
        ),
        (
            "Upper lead open",
            "Permissive upper voice for quieter lead notes",
            replace(
                upper_base,
                min_confidence=0.18,
                upper_relative_floor=0.30,
            ),
        ),
        (
            "Upper lead ornament",
            "Upper voice with more source-supported turns",
            replace(upper_base, upper_onset_relative_floor=0.30),
        ),
        (
            "Upper lead strict",
            "Only strong upper-voice events",
            replace(
                upper_base,
                min_confidence=0.34,
                upper_relative_floor=0.50,
                upper_octave_guard_ratio=1.40,
            ),
        ),
        (
            "Upper lead bright turns",
            "Bright envelope with attacked turns",
            replace(
                upper_base,
                upper_octave_guard_ratio=1.40,
                upper_onset_relative_floor=0.45,
            ),
        ),
        (
            "Upper lead compact",
            "Focused lead with short dropout recovery",
            replace(
                upper_base,
                upper_relative_floor=0.50,
                upper_octave_guard_ratio=1.40,
                upper_dropout_max_steps=max(3, round(6 * upper_step_scale)),
                upper_onset_max_semitones=5,
            ),
        ),
        (
            "Upper lead short turns",
            "Upper lead with tightly bounded attacked turns",
            replace(
                upper_base,
                upper_onset_relative_floor=0.45,
                upper_onset_max_semitones=5,
            ),
        ),
        (
            "Upper lead open turns",
            "Quiet upper events with source-supported turns",
            replace(
                upper_base,
                min_confidence=0.18,
                upper_relative_floor=0.30,
                upper_onset_relative_floor=0.45,
            ),
        ),
    )
    for name, character, settings in upper_profiles:
        profiles.append(VariationProfile(name, character, center_detail, settings))

    profiles.append(
        VariationProfile(
            "Proven event",
            "Validated onset-aware melody path",
            center_detail,
            mp3_to_rtttl.MelodySettings(
                grid=grid,
                min_confidence=0.32,
                onset_threshold=0.45,
                frame_threshold=0.25,
                minimum_note_ms=100.0,
                top_voice_bias=0.10,
                jump_penalty=0.055,
                note_change_penalty=0.14,
                onset_change_bonus=0.70,
                onset_change_relief=0.24,
                onset_same_pitch_bonus=0.15,
                sustain_bonus=0.42,
                repeated_note_min_steps=2,
                minimum_melody_run_steps=2,
                octave_smoothing=True,
                max_melody_span=14,
                decoder_mode="events",
            ),
        )
    )
    event_base = mp3_to_rtttl.MelodySettings(grid=grid, decoder_mode="events")
    event_profiles = (
        ("Event smooth", "Conservative extracted-note line", -0.16, 0.46, 0.075, 0.13),
        ("Event attacks", "Attack-led extracted notes", 0.18, 0.28, 0.035, 0.28),
        ("Event upper", "Upper extracted-note line", 0.00, 0.38, 0.090, 0.20),
        ("Event sparse", "Strongest extracted-note hook", -0.30, 0.52, 0.050, 0.16),
    )
    for name, character, offset, change_penalty, voice_bias, onset_relief in event_profiles:
        detail = clamp(center_detail + offset * spread, 0.0, 1.0)
        settings = replace(
            event_base,
            min_confidence=lerp(0.50, 0.24, detail),
            onset_threshold=lerp(0.56, 0.30, detail),
            frame_threshold=lerp(0.34, 0.18, detail),
            minimum_note_ms=lerp(190.0, 65.0, detail),
            top_voice_bias=voice_bias,
            jump_penalty=lerp(0.095, 0.050, detail),
            note_change_penalty=change_penalty,
            onset_change_bonus=lerp(0.48, 0.84, detail),
            onset_change_relief=onset_relief,
            sustain_bonus=lerp(0.58, 0.34, detail),
            repeated_note_min_steps=max(2, round(lerp(5, 2, detail))),
            minimum_melody_run_steps=max(1, round(lerp(4, 2, detail))),
            max_melody_span=round(lerp(12, 16, detail)),
        )
        profiles.append(VariationProfile(name, character, detail, settings))
    return _balanced_profile_pool(profiles, count)


def summarize_variation(
    profile: VariationProfile,
    rtttl: str,
    analysis: Optional[mp3_to_rtttl.TranscriptionAnalysis] = None,
) -> VariationResult:
    tokens = parse_rtttl_tokens(rtttl, grid=profile.settings.grid)
    pitches = [token.midi for token in tokens if token.midi is not None]
    jumps = [
        abs(current - previous) for previous, current in zip(pitches, pitches[1:], strict=False)
    ]
    duration = max(0.001, sum(token.seconds for token in tokens))
    return VariationResult(
        profile=profile,
        rtttl=rtttl,
        token_count=len(tokens),
        notes_per_second=len(pitches) / duration,
        span=max(pitches) - min(pitches) if pitches else 0,
        big_jumps=sum(jump >= 12 for jump in jumps),
        average_jump=float(np.mean(jumps)) if jumps else 0.0,
        match_score=(
            mp3_to_rtttl.score_rtttl_fidelity(analysis, rtttl, profile.settings.grid)
            if analysis is not None
            else 0.0
        ),
    )


def _variation_signature(result: VariationResult) -> tuple[tuple[Optional[int], int], ...]:
    tokens = parse_rtttl_tokens(result.rtttl, grid=result.profile.settings.grid)
    return tuple(
        (token.midi % 12 if token.midi is not None else None, token.steps) for token in tokens
    )


def _variation_frames(result: VariationResult) -> tuple[Optional[int], ...]:
    frames: list[Optional[int]] = []
    for token in parse_rtttl_tokens(result.rtttl, grid=result.profile.settings.grid):
        frames.extend([token.midi % 12 if token.midi is not None else None] * token.steps)
    return tuple(frames)


def _variation_distance(left: VariationResult, right: VariationResult) -> float:
    """Measure pitch-class disagreement after normalizing both melodies in time."""
    left_frames = _variation_frames(left)
    right_frames = _variation_frames(right)
    if not left_frames or not right_frames:
        return 1.0
    samples = 64
    mismatches = 0
    for index in range(samples):
        left_index = min(
            len(left_frames) - 1, round(index * (len(left_frames) - 1) / (samples - 1))
        )
        right_index = min(
            len(right_frames) - 1, round(index * (len(right_frames) - 1) / (samples - 1))
        )
        mismatches += left_frames[left_index] != right_frames[right_index]
    return mismatches / samples


def rank_variations(
    results: list[VariationResult],
    limit: int,
) -> tuple[VariationResult, ...]:
    """Choose distinct, evidence-backed results while balancing decoder families."""
    if limit <= 0 or not results:
        return ()

    # Different decoders can emit the same ringtone. Keep the strongest source
    # match for each melody before assigning family quotas.
    best_by_signature: dict[tuple[tuple[Optional[int], int], ...], VariationResult] = {}
    for result in results:
        signature = _variation_signature(result)
        previous = best_by_signature.get(signature)
        if previous is None or result.match_score > previous.match_score:
            best_by_signature[signature] = result
    unique = list(best_by_signature.values())
    if not unique:
        return ()

    groups: dict[str, list[VariationResult]] = {}
    for result in unique:
        family = result.profile.settings.decoder_mode.strip().lower()
        groups.setdefault(family, []).append(result)

    preferred_order = ("contour", "upper-events", "events")
    family_order = [family for family in preferred_order if family in groups]
    family_order.extend(sorted(set(groups) - set(family_order)))
    target = min(limit, len(unique))
    family_strength = {
        family: max(result.match_score for result in groups[family]) for family in family_order
    }
    quotas = dict.fromkeys(family_order, 0)

    # Water-filling keeps the requested set as even as family capacity allows;
    # evidence breaks ties when the result count is not divisible by the number
    # of available decoder families.
    for _slot in range(target):
        available = [family for family in family_order if quotas[family] < len(groups[family])]
        if not available:
            break
        family = max(
            available,
            key=lambda name: (-quotas[name], family_strength[name]),
        )
        quotas[family] += 1

    selected: list[VariationResult] = []
    selected_per_family = dict.fromkeys(family_order, 0)
    remaining = list(unique)
    while remaining and len(selected) < target:
        eligible = [
            result
            for result in remaining
            if selected_per_family[result.profile.settings.decoder_mode.strip().lower()]
            < quotas[result.profile.settings.decoder_mode.strip().lower()]
        ]
        if not eligible:
            break

        def selection_score(item: VariationResult) -> tuple[float, float, str]:
            distance = (
                min(_variation_distance(item, chosen) for chosen in selected) if selected else 0.0
            )
            return (
                item.match_score + 0.08 * distance,
                item.match_score,
                item.profile.name,
            )

        candidate = max(
            eligible,
            key=selection_score,
        )
        remaining.remove(candidate)
        selected.append(candidate)
        family = candidate.profile.settings.decoder_mode.strip().lower()
        selected_per_family[family] += 1

    selected.sort(key=lambda item: item.match_score, reverse=True)
    return tuple(selected)


class StudioWindow(QMainWindow):
    """Main window coordinating audio analysis, ranking, playback, and editing."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APPLICATION_NAME)
        self.resize(1500, 1000)
        self.setMinimumSize(1160, 780)
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)
        self.player = AudioPlayer(self)
        self.player.positionChanged.connect(self._player_position_changed)
        self.player.playingChanged.connect(self._player_playing_changed)
        self.transport_mapping: Optional[tuple[float, float, float, float]] = None
        self.transport_kind: Optional[str] = None
        self.rtttl_token_spans: list[tuple[int, int]] = []
        self.rtttl_token_times: list[tuple[float, float]] = []
        self.active_rtttl_token_index: Optional[int] = None
        self.editor_result_index: Optional[int] = None
        self.editor_runs: list[tuple[Optional[int], int]] = []
        self.editor_grid = 16
        self.editor_bpm = 120
        self.editor_title = "Ringtone"
        self.editor_selected_token_index: Optional[int] = None
        self.editor_undo_stack: list[str] = []
        self.editor_redo_stack: list[str] = []
        self.audio: Optional[LoadedAudio] = None
        self.current_analysis: Optional[mp3_to_rtttl.TranscriptionAnalysis] = None
        self.results: list[VariationResult] = []
        self.analysis_cache: OrderedDict[tuple, mp3_to_rtttl.TranscriptionAnalysis] = OrderedDict()
        self.request_id = 0
        self.busy = False
        self.syncing_times = False
        self._build_ui()
        self._apply_style()
        self._update_control_labels()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("Hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 18, 20, 18)
        title_column = QVBoxLayout()
        title = QLabel("RTTTL MELODY LAB")
        title.setObjectName("HeroTitle")
        subtitle = QLabel(
            "Analyze once. Shape the tune. Compare several ringtone interpretations without rerunning ONNX."
        )
        subtitle.setObjectName("HeroSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        hero_layout.addLayout(title_column, 1)
        self.status_chip = QLabel("OPEN AN AUDIO FILE")
        self.status_chip.setObjectName("StatusChip")
        hero_layout.addWidget(self.status_chip)
        self.open_button = QPushButton("Open audio")
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.clicked.connect(self.open_audio)
        hero_layout.addWidget(self.open_button)
        root.addWidget(hero)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        source_card, source_layout = self._make_card(
            "SOURCE EXCERPT",
            "Drag amber edges to resize. Click to seek. Shift-drag inside to move the selection.",
        )
        source_card.setMinimumWidth(390)
        source_card.setMaximumWidth(500)
        self.file_label = QLabel("No audio loaded")
        self.file_label.setObjectName("FileName")
        self.file_label.setWordWrap(True)
        self.duration_label = QLabel("Open an MP3, WAV, FLAC, M4A, or OGG file.")
        self.duration_label.setObjectName("Muted")
        source_layout.addWidget(self.file_label)
        source_layout.addWidget(self.duration_label)

        self.waveform = WaveformWidget()
        self.waveform.selectionChanged.connect(self._waveform_changed)
        self.waveform.selectionCommitted.connect(self._selection_committed)
        self.waveform.playheadCommitted.connect(self._playhead_committed)
        source_layout.addWidget(self.waveform)

        time_grid = QGridLayout()
        time_grid.setHorizontalSpacing(12)
        time_grid.addWidget(QLabel("START"), 0, 0)
        time_grid.addWidget(QLabel("END"), 0, 1)
        self.start_spin = self._time_spinbox()
        self.end_spin = self._time_spinbox()
        self.start_spin.editingFinished.connect(self._times_edited)
        self.end_spin.editingFinished.connect(self._times_edited)
        time_grid.addWidget(self.start_spin, 1, 0)
        time_grid.addWidget(self.end_spin, 1, 1)
        source_layout.addLayout(time_grid)

        transport = QHBoxLayout()
        self.play_source_button = QPushButton("Play selection")
        self.play_source_button.setObjectName("SecondaryButton")
        self.play_source_button.clicked.connect(self.play_source)
        self.toggle_play_button = QPushButton("Play / Pause")
        self.toggle_play_button.setObjectName("SecondaryButton")
        self.toggle_play_button.clicked.connect(self.toggle_transport)
        stop_button = QPushButton("Stop")
        stop_button.setObjectName("QuietButton")
        stop_button.clicked.connect(self.stop_transport)
        transport.addWidget(self.play_source_button, 1)
        transport.addWidget(self.toggle_play_button)
        transport.addWidget(stop_button)
        source_layout.addLayout(transport)
        keyboard_hint = QLabel(
            "Space: play/pause | Left/Right: seek 1s | Shift: 5s | Home/End: selection bounds"
        )
        keyboard_hint.setObjectName("Muted")
        keyboard_hint.setWordWrap(True)
        source_layout.addWidget(keyboard_hint)

        analysis_note = QLabel(
            "ONNX analysis is cached for this exact excerpt. Threshold and melody settings below only rerender the cached model output."
        )
        analysis_note.setObjectName("InfoBox")
        analysis_note.setWordWrap(True)
        source_layout.addWidget(analysis_note)
        self.cache_label = QLabel("Nothing analyzed yet")
        self.cache_label.setObjectName("CacheState")
        source_layout.addWidget(self.cache_label)
        source_layout.addStretch(1)
        splitter.addWidget(source_card)

        right = QWidget()
        right.setObjectName("RightPane")
        right.setMinimumHeight(1380)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        controls_card, controls_layout = self._make_card(
            "VARIATION LAB",
            "The center setting is surrounded by simpler and richer alternatives.",
        )
        controls_card.setMinimumHeight(500)
        identity_grid = QGridLayout()
        identity_grid.setHorizontalSpacing(12)
        identity_grid.setVerticalSpacing(10)
        identity_grid.setColumnStretch(0, 1)
        identity_grid.setColumnStretch(1, 1)
        identity_grid.addWidget(QLabel("RTTTL TITLE"), 0, 0)
        identity_grid.addWidget(QLabel("BPM (0 = AUTO)"), 0, 2)
        identity_grid.addWidget(QLabel("GRID"), 0, 3)
        self.title_edit = QLineEdit("Ringtone")
        self.title_edit.setMaxLength(10)
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(0, 900)
        self.bpm_spin.setDecimals(0)
        self.bpm_spin.setSpecialValueText("Auto")
        self.grid_combo = QComboBox()
        self.grid_combo.addItem("16th", 16)
        self.grid_combo.addItem("32nd", 32)
        self.count_combo = QComboBox()
        for count in (9, 12, 15):
            self.count_combo.addItem(str(count), count)
        self.count_combo.setCurrentIndex(1)
        self.spread_combo = QComboBox()
        self.spread_combo.addItem("Narrow", 0.24)
        self.spread_combo.addItem("Useful", 0.50)
        self.spread_combo.addItem("Wide", 0.78)
        self.spread_combo.setCurrentIndex(1)
        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItem("Auto (recommended)", "auto")
        self.source_mode_combo.addItem("Full mix - games/chiptune", "full")
        self.source_mode_combo.addItem("Harmonic lead - mixed songs", "harmonic")
        identity_grid.addWidget(self.title_edit, 1, 0, 1, 2)
        identity_grid.addWidget(self.bpm_spin, 1, 2)
        identity_grid.addWidget(self.grid_combo, 1, 3)
        identity_grid.addWidget(QLabel("VERSIONS"), 2, 0)
        identity_grid.addWidget(QLabel("SPREAD"), 2, 1)
        identity_grid.addWidget(QLabel("ANALYSIS SOURCE"), 2, 2)
        identity_grid.addWidget(self.count_combo, 3, 0)
        identity_grid.addWidget(self.spread_combo, 3, 1)
        identity_grid.addWidget(self.source_mode_combo, 3, 2, 1, 2)
        controls_layout.addLayout(identity_grid)

        self.detail_slider, self.detail_value = self._add_slider_control(
            controls_layout,
            "MELODY DETAIL",
            "Left keeps the hook and rejects ornamental notes; right follows more detected movement.",
            35,
        )
        self.smooth_slider, self.smooth_value = self._add_slider_control(
            controls_layout,
            "CONTINUITY",
            "Higher values prefer held, singable lines and penalize rapid pitch changes.",
            68,
        )
        self.lead_slider, self.lead_value = self._add_slider_control(
            controls_layout,
            "LEAD PREFERENCE",
            "Higher values favor the upper voice when accompaniment overlaps the melody.",
            52,
        )
        for slider in (self.detail_slider, self.smooth_slider, self.lead_slider):
            slider.valueChanged.connect(self._update_control_labels)
            slider.sliderReleased.connect(self._settings_changed)
        for combo in (self.grid_combo, self.count_combo, self.spread_combo):
            combo.currentIndexChanged.connect(self._settings_changed)
        self.source_mode_combo.currentIndexChanged.connect(self._analysis_mode_changed)
        self.bpm_spin.editingFinished.connect(self._settings_changed)
        self.title_edit.editingFinished.connect(self._settings_changed)

        action_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        self.progress.hide()
        self.generate_button = QPushButton("Analyze + generate 12 versions")
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.clicked.connect(self.generate_variations)
        action_row.addWidget(self.progress, 1)
        action_row.addWidget(self.generate_button)
        controls_layout.addLayout(action_row)
        right_layout.addWidget(controls_card)

        results_card, results_layout = self._make_card(
            "VERSIONS",
            "Evidence-ranked results stay balanced across contour, upper-lead, and onset-aware decoders. Double-click to play.",
        )
        results_card.setMinimumHeight(250)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ("Variation", "Character", "Evidence", "Tokens", "Notes/sec", "Span", "Big jumps")
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in range(2, 7):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.table.itemSelectionChanged.connect(self._result_selected)
        self.table.cellDoubleClicked.connect(lambda _row, _column: self.play_selected())
        results_layout.addWidget(self.table)
        right_layout.addWidget(results_card, 1)

        visual_card, visual_layout = self._make_card(
            "ANALYSIS VS SELECTED RTTTL",
            "Color shows Basic Pitch confidence and attacks; the bright overlay is the selected ringtone.",
        )
        visual_card.setMinimumHeight(455)
        self.piano_roll = PianoRollWidget()
        self.piano_roll.tokenSelected.connect(self._editor_select_token)
        self.piano_roll.pitchChangeRequested.connect(self._editor_change_pitch)
        self.piano_roll.durationChangeRequested.connect(self._editor_resize_token)
        self.piano_roll.restToggleRequested.connect(self._editor_toggle_rest)
        self.piano_roll.editorCommand.connect(self._editor_command)
        visual_layout.addWidget(self.piano_roll)

        editor_toolbar = QHBoxLayout()
        editor_toolbar.setSpacing(10)
        self.editor_undo_button = QPushButton("Undo")
        self.editor_redo_button = QPushButton("Redo")
        self.editor_rest_button = QPushButton("Rest / Note")
        self.editor_shorter_button = QPushButton("Shorter (-)")
        self.editor_longer_button = QPushButton("Longer (+)")
        self.editor_snap_button = QPushButton("Magnetic snap: ON")
        self.editor_snap_button.setCheckable(True)
        self.editor_snap_button.setChecked(True)
        for button in (
            self.editor_undo_button,
            self.editor_redo_button,
            self.editor_rest_button,
            self.editor_shorter_button,
            self.editor_longer_button,
            self.editor_snap_button,
        ):
            button.setObjectName("QuietButton")
            editor_toolbar.addWidget(button)
        editor_toolbar.addStretch(1)
        self.editor_undo_button.clicked.connect(self._editor_undo)
        self.editor_redo_button.clicked.connect(self._editor_redo)
        self.editor_rest_button.clicked.connect(self._editor_toggle_selected_rest)
        self.editor_shorter_button.clicked.connect(
            lambda: self._editor_change_selected_duration(-1)
        )
        self.editor_longer_button.clicked.connect(lambda: self._editor_change_selected_duration(1))
        self.editor_snap_button.toggled.connect(self._editor_magnetic_snap_toggled)
        self.editor_snap_button.setToolTip(
            "When enabled, mouse pitch edits lock to strong notes detected in the source during this token."
        )
        visual_layout.addLayout(editor_toolbar)

        editor_hint = QLabel(
            "Click a note to select it. Click another colored lane at the same time or drag vertically "
            "to replace its pitch; drag the blue right handle to resize. Arrows select/transpose, "
            "-/+ resize, Shift+Left/Right jump to first/last, R makes a rest, and Cmd/Ctrl+Z undoes."
        )
        editor_hint.setObjectName("Muted")
        editor_hint.setWordWrap(True)
        visual_layout.addWidget(editor_hint)
        right_layout.addWidget(visual_card)

        output_card, output_layout = self._make_card("SELECTED RTTTL", "")
        output_card.setMinimumHeight(220)
        output_header = QHBoxLayout()
        self.selected_name = QLabel("No version selected")
        self.selected_name.setObjectName("SelectedName")
        self.settings_summary = QLabel("")
        self.settings_summary.setObjectName("Muted")
        self.settings_summary.setAlignment(Qt.AlignmentFlag.AlignRight)
        output_header.addWidget(self.selected_name)
        output_header.addWidget(self.settings_summary, 1)
        output_layout.addLayout(output_header)
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumBlockCount(20)
        self.output_text.setMinimumHeight(125)
        output_layout.addWidget(self.output_text)
        output_actions = QHBoxLayout()
        for label, callback in (
            ("Play", self.play_selected),
            ("Copy", self.copy_selected),
            ("Use as center", self.use_selected_as_center),
            ("Save selected", self.save_selected),
            ("Save all", self.save_all),
        ):
            button = QPushButton(label)
            button.setObjectName(
                "SecondaryButton" if label in ("Play", "Save selected") else "QuietButton"
            )
            button.clicked.connect(callback)
            output_actions.addWidget(button)
        output_actions.addStretch(1)
        output_layout.addLayout(output_actions)
        right_layout.addWidget(output_card)
        right_scroll = QScrollArea()
        right_scroll.setObjectName("StudioScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((430, 950))

        self.footer_status = QLabel("Ready")
        self.footer_status.setObjectName("FooterStatus")
        root.addWidget(self.footer_status)

    def _make_card(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        if subtitle:
            subheading = QLabel(subtitle)
            subheading.setObjectName("Muted")
            subheading.setWordWrap(True)
            layout.addWidget(subheading)
        return card, layout

    def _time_spinbox(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 99_999.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setSuffix(" s")
        return spin

    def _add_slider_control(
        self,
        parent: QVBoxLayout,
        label: str,
        explanation: str,
        value: int,
    ) -> tuple[QSlider, QLabel]:
        block = QFrame()
        block.setObjectName("ControlBlock")
        block.setMinimumHeight(100)
        row = QGridLayout(block)
        row.setContentsMargins(12, 9, 12, 9)
        row.setVerticalSpacing(5)
        row.setColumnStretch(1, 1)
        heading = QLabel(label)
        heading.setObjectName("ControlLabel")
        value_label = QLabel()
        value_label.setObjectName("ControlValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.setMinimumHeight(28)
        hint = QLabel(explanation)
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        hint.setMinimumHeight(32)
        row.addWidget(heading, 0, 0)
        row.addWidget(value_label, 0, 2)
        row.addWidget(slider, 1, 0, 1, 3)
        row.addWidget(hint, 2, 0, 1, 3)
        parent.addWidget(block)
        return slider, value_label

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #090d0e;
                color: #e9eee8;
                font-family: "Avenir Next", "Helvetica Neue";
                font-size: 15px;
            }
            QFrame#Hero {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #17302d, stop:0.62 #172421, stop:1 #2b2116);
                border: 1px solid #38514c;
                border-radius: 18px;
            }
            QLabel#HeroTitle { font-size: 26px; font-weight: 800; letter-spacing: 2px; color: #f3f6ef; }
            QLabel#HeroSubtitle { color: #a9bbb5; font-size: 14px; }
            QLabel#StatusChip {
                background: #10201e; color: #8fd3bd; border: 1px solid #31534b;
                border-radius: 12px; padding: 8px 12px; font-size: 12px; font-weight: 700;
            }
            QFrame#Card { background: #111718; border: 1px solid #273233; border-radius: 15px; }
            QFrame#ControlBlock { background: #0d1415; border: 1px solid #222e2e; border-radius: 10px; }
            QLabel#SectionTitle { color: #f4a63a; font-size: 14px; font-weight: 800; letter-spacing: 1.4px; }
            QLabel#Muted { color: #9caaa7; font-size: 13px; }
            QLabel#FileName { color: #f2f4ef; font-size: 18px; font-weight: 700; }
            QLabel#SelectedName { color: #f2f4ef; font-size: 16px; font-weight: 700; }
            QLabel#ControlLabel { color: #cbd5d0; font-size: 12px; font-weight: 800; letter-spacing: 1px; }
            QLabel#ControlValue { color: #f4a63a; font-size: 13px; font-weight: 700; }
            QLabel#InfoBox {
                background: #10211f; color: #9fc5b8; border: 1px solid #27433d;
                border-radius: 10px; padding: 11px;
            }
            QLabel#CacheState { color: #f4a63a; font-weight: 700; padding: 3px; }
            QLabel#FooterStatus { color: #7f908c; padding-left: 4px; }
            QPushButton {
                min-height: 40px; padding: 0 16px; border-radius: 9px;
                font-weight: 700; border: 1px solid transparent;
            }
            QPushButton#PrimaryButton { background: #f4a63a; color: #191208; border-color: #f7bd67; }
            QPushButton#PrimaryButton:hover { background: #ffc166; }
            QPushButton#PrimaryButton:pressed { background: #d98b25; }
            QPushButton#SecondaryButton { background: #18322f; color: #d6e9e2; border-color: #31534d; }
            QPushButton#SecondaryButton:hover { background: #21433e; border-color: #4c746b; }
            QPushButton#QuietButton { background: #181f20; color: #aebbb7; border-color: #303b3c; }
            QPushButton#QuietButton:hover { background: #222b2c; color: #eef2ed; }
            QPushButton:disabled { background: #202526; color: #606b69; border-color: #2b3233; }
            QLineEdit, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: #0b1112; color: #e7ede8; border: 1px solid #303c3c;
                border-radius: 8px; padding: 9px; min-height: 26px;
                selection-background-color: #7f5723;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
                border-color: #d99335;
            }
            QComboBox::drop-down { border: 0; width: 22px; }
            QComboBox QAbstractItemView { background: #111718; selection-background-color: #31534d; }
            QSlider::groove:horizontal { height: 5px; background: #263131; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #b97a2c; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #ffd08a; border: 2px solid #f4a63a; width: 15px;
                margin: -6px 0; border-radius: 8px;
            }
            QTableWidget {
                background: #0b1112; alternate-background-color: #0f1617;
                border: 1px solid #283334; border-radius: 9px; gridline-color: #202a2b;
                selection-background-color: #284a43; selection-color: #f7f8f3;
            }
            QHeaderView::section {
                background: #182021; color: #8fa29d; border: 0; border-bottom: 1px solid #303b3c;
                padding: 8px; font-size: 11px; font-weight: 800;
            }
            QProgressBar { background: #202a2a; border: 0; border-radius: 3px; }
            QProgressBar::chunk { background: #f4a63a; border-radius: 3px; }
            QSplitter::handle { background: transparent; width: 12px; }
            QScrollArea#StudioScroll { border: 0; background: transparent; }
            QScrollBar:vertical {
                background: #101718; width: 12px; margin: 2px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #41504e; min-height: 40px; border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover { background: #637572; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QToolTip { background: #172020; color: #edf2ed; border: 1px solid #3b4d4a; padding: 6px; }
            """
        )

    def _update_control_labels(self) -> None:
        detail = self.detail_slider.value()
        if detail < 20:
            detail_text = "Hook only"
        elif detail < 45:
            detail_text = "Sparse"
        elif detail < 70:
            detail_text = "Balanced"
        else:
            detail_text = "Detailed"
        smooth = self.smooth_slider.value()
        smooth_text = "Agile" if smooth < 35 else "Natural" if smooth < 65 else "Smooth"
        lead = self.lead_slider.value()
        lead_text = "Neutral" if lead < 40 else "Lead-aware" if lead < 72 else "Top voice"
        self.detail_value.setText(detail_text)
        self.smooth_value.setText(smooth_text)
        self.lead_value.setText(lead_text)
        if hasattr(self, "generate_button"):
            count = int(self.count_combo.currentData())
            cached = self.audio is not None and self._current_cache_key() in self.analysis_cache
            prefix = "Generate" if cached else "Analyze + generate"
            self.generate_button.setText(f"{prefix} {count} versions")

    def _set_status(self, text: str, chip: Optional[str] = None) -> None:
        self.footer_status.setText(text)
        if chip is not None:
            self.status_chip.setText(chip.upper())

    def _set_busy(self, busy: bool) -> None:
        """Lock generation inputs so completed work always matches visible controls."""
        self.busy = busy
        self.open_button.setEnabled(not busy)
        self.generate_button.setEnabled(not busy)
        generation_inputs = (
            self.waveform,
            self.start_spin,
            self.end_spin,
            self.title_edit,
            self.bpm_spin,
            self.grid_combo,
            self.count_combo,
            self.spread_combo,
            self.source_mode_combo,
            self.detail_slider,
            self.smooth_slider,
            self.lead_slider,
        )
        for control in generation_inputs:
            control.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)

    def _task_finished(self) -> None:
        self._set_busy(False)

    def _task_error(self, message: str) -> None:
        self._set_status(message, "Action failed")
        QMessageBox.critical(self, APPLICATION_NAME, message)

    def open_audio(self) -> None:
        initial = str(self.audio.path.parent) if self.audio else str(Path.cwd())
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open audio",
            initial,
            "Audio (*.mp3 *.wav *.flac *.m4a *.ogg);;All files (*)",
        )
        if selected:
            self.load_audio(Path(selected))

    def load_audio(self, path: Path) -> None:
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            self._task_error(f"Unsupported audio file: {path.name}")
            return
        self.request_id += 1
        request_id = self.request_id
        self._set_busy(True)
        self._set_status(f"Loading waveform for {path.name}...", "Loading source")

        def work(progress) -> LoadedAudio:
            progress("Decoding a lightweight waveform preview...")
            return load_audio_preview(path)

        worker = Worker(work)
        worker.signals.progress.connect(lambda message: self._set_status(message, "Loading source"))
        worker.signals.result.connect(
            lambda loaded: self._audio_loaded(loaded) if request_id == self.request_id else None
        )
        worker.signals.error.connect(self._task_error)
        worker.signals.finished.connect(self._task_finished)
        self.thread_pool.start(worker)

    def _audio_loaded(self, loaded: LoadedAudio) -> None:
        self.transport_mapping = None
        self.transport_kind = None
        self.player.stop()
        self.audio = loaded
        self.file_label.setText(loaded.path.name)
        self.duration_label.setText(f"{format_time(loaded.duration)} | {loaded.path.parent}")
        self.title_edit.setText(mp3_to_rtttl.sanitize_title(loaded.path.stem))
        self.start_spin.setMaximum(loaded.duration)
        self.end_spin.setMaximum(loaded.duration)
        self.waveform.set_audio(loaded.duration, loaded.waveform)
        self._clear_results()
        self._update_cache_state()
        self._set_status("Choose an excerpt, then generate variations.", "Source ready")

    def _waveform_changed(self, start: float, end: float) -> None:
        self.syncing_times = True
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.syncing_times = False

    def _selection_committed(self, start: float, _end: float) -> None:
        self.transport_mapping = None
        self.transport_kind = None
        self.player.stop()
        self.waveform.set_playhead(start)
        self.request_id += 1
        self._clear_results()
        self._update_cache_state()
        self._set_status("Selection changed. Generate a new comparison.", "Selection changed")

    def _times_edited(self) -> None:
        if self.syncing_times or not self.audio:
            return
        self.waveform.set_selection(self.start_spin.value(), self.end_spin.value(), commit=True)

    def _settings_changed(self) -> None:
        self._update_control_labels()
        if self.results:
            self._set_status(
                "Settings changed. Generate again to compare the new range.", "Ready to rerender"
            )

    def _analysis_mode_changed(self) -> None:
        self.request_id += 1
        self._clear_results()
        self._update_cache_state()
        self._set_status(
            "Analysis source changed. Generate a new comparison.", "Source mode changed"
        )

    def _current_cache_key(self) -> Optional[tuple]:
        if not self.audio:
            return None
        try:
            modified = self.audio.path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return (
            str(self.audio.path),
            modified,
            round(self.waveform.start, 3),
            round(self.waveform.end - self.waveform.start, 3),
            str(self.source_mode_combo.currentData()),
        )

    def _cache_put(self, key: tuple, analysis: mp3_to_rtttl.TranscriptionAnalysis) -> None:
        self.analysis_cache[key] = analysis
        self.analysis_cache.move_to_end(key)
        while len(self.analysis_cache) > MAX_CACHE_ITEMS:
            self.analysis_cache.popitem(last=False)

    def _update_cache_state(self) -> None:
        key = self._current_cache_key()
        if key is not None and key in self.analysis_cache:
            self.cache_label.setText("ONNX analysis cached - settings rerender quickly")
        else:
            self.cache_label.setText("This excerpt needs one ONNX analysis")
        self._update_control_labels()

    def generate_variations(self) -> None:
        if not self.audio:
            self._task_error("Open an audio file first.")
            return
        if self.busy:
            return

        start = self.waveform.start
        duration = self.waveform.end - start
        if duration < 0.1:
            self._task_error("Select at least 0.1 seconds of audio.")
            return

        cache_key = self._current_cache_key()
        if cache_key is None:
            return
        cached_analysis = self.analysis_cache.get(cache_key)
        count = int(self.count_combo.currentData())
        profiles = variation_profiles(
            center_detail=self.detail_slider.value() / 100.0,
            smoothness=self.smooth_slider.value() / 100.0,
            lead_preference=self.lead_slider.value() / 100.0,
            count=count,
            spread=float(self.spread_combo.currentData()),
            grid=int(self.grid_combo.currentData()),
        )
        title = self.title_edit.text().strip() or self.audio.path.stem
        bpm = self.bpm_spin.value() or None
        source_mode = str(self.source_mode_combo.currentData())
        source_path = self.audio.path

        self.request_id += 1
        request_id = self.request_id
        self._set_busy(True)
        self._set_status("Preparing melody analysis...", "Working")

        def work(progress) -> GenerationPayload:
            analysis = cached_analysis
            used_cache = analysis is not None
            if analysis is None:
                analysis = mp3_to_rtttl.analyze_audio(
                    source_path,
                    start=start,
                    duration=duration,
                    source_mode=source_mode,
                    progress=progress,
                )
            else:
                progress("Using cached ONNX model output...")

            generated: list[VariationResult] = []
            failures: list[str] = []
            for index, profile in enumerate(profiles, start=1):
                progress(f"Rendering version {index} of {len(profiles)}: {profile.name}...")
                try:
                    rtttl = mp3_to_rtttl.render_analysis(
                        analysis,
                        title=title,
                        settings=profile.settings,
                        bpm_override=bpm,
                    )
                except RuntimeError as exc:
                    failures.append(f"{profile.name}: {exc}")
                    continue
                generated.append(summarize_variation(profile, rtttl, analysis))

            if not generated:
                detail = "; ".join(failures) or "No melody survived the selected settings."
                raise RuntimeError(detail)
            ranked = rank_variations(generated, count)
            if not ranked:
                raise RuntimeError("The candidate search produced no distinct melody versions.")
            return GenerationPayload(
                request_id=request_id,
                cache_key=cache_key,
                analysis=analysis,
                results=ranked,
                failures=tuple(failures),
                used_cache=used_cache,
            )

        worker = Worker(work)
        worker.signals.progress.connect(lambda message: self._set_status(message, "Working"))
        worker.signals.result.connect(self._generation_complete)
        worker.signals.error.connect(self._task_error)
        worker.signals.finished.connect(self._task_finished)
        self.thread_pool.start(worker)

    def _generation_complete(self, payload: GenerationPayload) -> None:
        if payload.request_id != self.request_id:
            return
        self._cache_put(payload.cache_key, payload.analysis)
        self.current_analysis = payload.analysis
        self.results = list(payload.results)
        self._populate_table()
        self._update_cache_state()
        cache_text = "cached analysis" if payload.used_cache else "one new ONNX analysis"
        suffix = (
            f" {len(payload.failures)} sparse setting(s) produced no notes."
            if payload.failures
            else ""
        )
        self._set_status(
            f"Generated {len(self.results)} versions from {cache_text} using {payload.analysis.source_mode} source.{suffix}",
            "Versions ready",
        )

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.results))
        for row, result in enumerate(self.results):
            values = (
                result.profile.name,
                result.profile.character,
                f"{result.match_score * 100:.0f}%",
                str(result.token_count),
                f"{result.notes_per_second:.2f}",
                str(result.span),
                str(result.big_jumps),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row)
                if column >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

        if self.results:
            center = self.detail_slider.value() / 100.0
            selected_row = min(
                range(len(self.results)),
                key=lambda index: abs(self.results[index].profile.detail - center),
            )
            self.table.selectRow(selected_row)

    def _refresh_rtttl_text_highlights(self) -> None:
        selections: list[QTextEdit.ExtraSelection] = []
        reveal_index: Optional[int] = None
        highlights = (
            (self.editor_selected_token_index, QColor("#245b80"), QColor("#f4fbff")),
            (self.active_rtttl_token_index, QColor("#63ffd7"), QColor("#06110f")),
        )
        for index, background, foreground in highlights:
            if index is None or not 0 <= index < len(self.rtttl_token_spans):
                continue
            start, end = self.rtttl_token_spans[index]
            selection = QTextEdit.ExtraSelection()
            selection.cursor = QTextCursor(self.output_text.document())
            selection.cursor.setPosition(start)
            selection.cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            selection.format.setBackground(background)
            selection.format.setForeground(foreground)
            selections.append(selection)
            reveal_index = index
        self.output_text.setExtraSelections(selections)
        if reveal_index is not None:
            cursor = QTextCursor(self.output_text.document())
            cursor.setPosition(self.rtttl_token_spans[reveal_index][0])
            self.output_text.setTextCursor(cursor)
            self.output_text.ensureCursorVisible()

    def _editor_result(self) -> Optional[VariationResult]:
        if self.editor_result_index is None:
            return None
        if not 0 <= self.editor_result_index < len(self.results):
            return None
        return self.results[self.editor_result_index]

    @staticmethod
    def _editor_rtttl_header(rtttl: str) -> tuple[str, int]:
        parts = rtttl.split(":", 2)
        title = parts[0].strip() if parts and parts[0].strip() else "Ringtone"
        bpm = 120
        if len(parts) > 1:
            for setting in parts[1].split(","):
                key, separator, value = setting.strip().partition("=")
                if separator and key.lower() == "b":
                    try:
                        bpm = max(25, min(900, int(value)))
                    except ValueError:
                        pass
        return title, bpm

    def _editor_parse_runs(self, rtttl: str) -> list[tuple[Optional[int], int]]:
        return [
            (token.midi, max(1, int(token.steps)))
            for token in parse_rtttl_tokens(rtttl, grid=self.editor_grid)
        ]

    def _prepare_editor(self, result: VariationResult) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.results):
            return
        self.editor_result_index = row
        self.editor_grid = max(1, int(result.profile.settings.grid))
        self.editor_title, self.editor_bpm = self._editor_rtttl_header(result.rtttl)
        self.editor_runs = self._editor_parse_runs(result.rtttl)
        self.editor_selected_token_index = 0 if self.editor_runs else None
        self.editor_undo_stack.clear()
        self.editor_redo_stack.clear()
        self.piano_roll.set_selected_token(self.editor_selected_token_index)
        self._refresh_rtttl_text_highlights()
        self._update_editor_buttons()

    def _editor_selected_step(self, index: Optional[int] = None) -> int:
        if index is None:
            index = self.editor_selected_token_index
        if index is None:
            return 0
        return sum(steps for _midi, steps in self.editor_runs[: max(0, index)])

    def _editor_index_at_step(self, step: int) -> Optional[int]:
        if not self.editor_runs:
            return None
        cursor = 0
        for index, (_midi, steps) in enumerate(self.editor_runs):
            if step < cursor + steps:
                return index
            cursor += steps
        return len(self.editor_runs) - 1

    def _update_editor_buttons(self) -> None:
        if not hasattr(self, "editor_undo_button"):
            return
        has_note = self._editor_result() is not None and bool(self.editor_runs)
        self.editor_undo_button.setEnabled(bool(self.editor_undo_stack))
        self.editor_redo_button.setEnabled(bool(self.editor_redo_stack))
        self.editor_rest_button.setEnabled(has_note)
        self.editor_shorter_button.setEnabled(has_note)
        self.editor_longer_button.setEnabled(has_note)

    def _editor_select_token(self, index: int) -> None:
        if not self.editor_runs:
            self.editor_selected_token_index = None
        else:
            self.editor_selected_token_index = max(0, min(int(index), len(self.editor_runs) - 1))
        self.piano_roll.set_selected_token(self.editor_selected_token_index)
        self._refresh_rtttl_text_highlights()
        self._update_editor_buttons()

    def _editor_refresh_result(self, rtttl: str, selection_step: int) -> None:
        result = self._editor_result()
        if result is None or self.editor_result_index is None:
            return
        updated = summarize_variation(result.profile, rtttl, self.current_analysis)
        self.results[self.editor_result_index] = updated
        self.editor_runs = self._editor_parse_runs(updated.rtttl)

        self.output_text.setPlainText(updated.rtttl)
        self.piano_roll.set_comparison(
            self.current_analysis,
            updated.rtttl,
            grid=self.editor_grid,
        )
        self._prepare_rtttl_animation(updated.rtttl)
        self._editor_select_token(self._editor_index_at_step(selection_step) or 0)

        row = self.editor_result_index
        values = (
            f"{updated.match_score * 100:.0f}%",
            str(updated.token_count),
            f"{updated.notes_per_second:.1f}",
            str(updated.span),
            str(updated.big_jumps),
        )
        for column, value in enumerate(values, start=2):
            item = self.table.item(row, column)
            if item is not None:
                item.setText(value)
        self.selected_name.setText(f"{updated.profile.name} (edited)")
        self._update_editor_buttons()

    def _editor_commit_runs(
        self,
        runs: list[tuple[Optional[int], int]],
        selection_step: int,
        *,
        record_history: bool = True,
    ) -> None:
        result = self._editor_result()
        if result is None or not runs:
            return
        normalized = [(midi, max(1, int(steps))) for midi, steps in runs]
        rtttl = mp3_to_rtttl.runs_to_rtttl(
            normalized,
            self.editor_title,
            self.editor_bpm,
            self.editor_grid,
        )
        if rtttl == result.rtttl:
            return
        if record_history:
            self.editor_undo_stack.append(result.rtttl)
            del self.editor_undo_stack[:-100]
            self.editor_redo_stack.clear()
        if self.transport_kind == "rtttl":
            self.stop_transport()
        self._editor_refresh_result(rtttl, selection_step)

    def _editor_change_pitch(self, index: int, midi: int) -> None:
        if not 0 <= index < len(self.editor_runs):
            return
        midi = max(mp3_to_rtttl.RTTTL_MIN_MIDI, min(int(midi), mp3_to_rtttl.RTTTL_MAX_MIDI))
        runs = list(self.editor_runs)
        if runs[index][0] == midi:
            self._editor_select_token(index)
            return
        selection_step = self._editor_selected_step(index)
        runs[index] = (midi, runs[index][1])
        self._editor_commit_runs(runs, selection_step)

    def _editor_resize_token(self, index: int, delta_steps: int) -> None:
        if not 0 <= index < len(self.editor_runs) or delta_steps == 0:
            return
        runs = list(self.editor_runs)
        current_midi, current_steps = runs[index]
        if index + 1 < len(runs):
            next_midi, next_steps = runs[index + 1]
            delta_steps = max(1 - current_steps, min(int(delta_steps), next_steps - 1))
            if delta_steps == 0:
                return
            runs[index] = (current_midi, current_steps + delta_steps)
            runs[index + 1] = (next_midi, next_steps - delta_steps)
        else:
            delta_steps = max(1 - current_steps, min(int(delta_steps), self.editor_grid * 4))
            if delta_steps == 0:
                return
            runs[index] = (current_midi, current_steps + delta_steps)
        self._editor_commit_runs(runs, self._editor_selected_step(index))

    def _editor_fallback_pitch(self, index: int) -> int:
        for candidate in range(index - 1, -1, -1):
            midi = self.editor_runs[candidate][0]
            if midi is not None:
                return midi
        for candidate in range(index + 1, len(self.editor_runs)):
            midi = self.editor_runs[candidate][0]
            if midi is not None:
                return midi
        return max(mp3_to_rtttl.RTTTL_MIN_MIDI, min(72, mp3_to_rtttl.RTTTL_MAX_MIDI))

    def _editor_toggle_rest(self, index: int) -> None:
        if not 0 <= index < len(self.editor_runs):
            return
        runs = list(self.editor_runs)
        midi, steps = runs[index]
        runs[index] = (self._editor_fallback_pitch(index) if midi is None else None, steps)
        self._editor_commit_runs(runs, self._editor_selected_step(index))

    def _editor_toggle_selected_rest(self) -> None:
        if self.editor_selected_token_index is not None:
            self._editor_toggle_rest(self.editor_selected_token_index)

    def _editor_change_selected_duration(self, delta_steps: int) -> None:
        if self.editor_selected_token_index is not None:
            self._editor_resize_token(self.editor_selected_token_index, delta_steps)

    def _editor_transpose_selected(self, semitones: int) -> None:
        index = self.editor_selected_token_index
        if index is None or not 0 <= index < len(self.editor_runs):
            return
        midi = self.editor_runs[index][0]
        if midi is None:
            midi = self._editor_fallback_pitch(index)
        self._editor_change_pitch(index, midi + semitones)

    def _editor_move_selection(self, movement: str) -> None:
        if not self.editor_runs:
            return
        current = self.editor_selected_token_index or 0
        if movement == "previous":
            target = current - 1
        elif movement == "next":
            target = current + 1
        elif movement == "first":
            target = 0
        else:
            target = len(self.editor_runs) - 1
        self._editor_select_token(target)

    def _editor_undo(self) -> None:
        result = self._editor_result()
        if result is None or not self.editor_undo_stack:
            return
        selection_step = self._editor_selected_step()
        target = self.editor_undo_stack.pop()
        self.editor_redo_stack.append(result.rtttl)
        if self.transport_kind == "rtttl":
            self.stop_transport()
        self._editor_refresh_result(target, selection_step)

    def _editor_redo(self) -> None:
        result = self._editor_result()
        if result is None or not self.editor_redo_stack:
            return
        selection_step = self._editor_selected_step()
        target = self.editor_redo_stack.pop()
        self.editor_undo_stack.append(result.rtttl)
        if self.transport_kind == "rtttl":
            self.stop_transport()
        self._editor_refresh_result(target, selection_step)

    def _editor_command(self, command: str) -> None:
        if command == "play_pause":
            self.toggle_transport()
        elif command == "undo":
            self._editor_undo()
        elif command == "redo":
            self._editor_redo()
        elif command in {"previous", "next", "first", "last"}:
            self._editor_move_selection(command)
        elif command == "shorter":
            self._editor_change_selected_duration(-1)
        elif command == "longer":
            self._editor_change_selected_duration(1)
        elif command == "rest":
            self._editor_toggle_selected_rest()
        elif command.startswith("transpose:"):
            try:
                self._editor_transpose_selected(int(command.partition(":")[2]))
            except ValueError:
                pass

    def _editor_magnetic_snap_toggled(self, enabled: bool) -> None:
        self.piano_roll.set_magnetic_snap(enabled)
        self.editor_snap_button.setText(f"Magnetic snap: {'ON' if enabled else 'OFF'}")

    def _clear_results(self) -> None:
        self.editor_result_index = None
        self.editor_runs.clear()
        self.editor_selected_token_index = None
        self.editor_undo_stack.clear()
        self.editor_redo_stack.clear()
        self._update_editor_buttons()
        self.current_analysis = None
        self.results.clear()
        self.table.setRowCount(0)
        self.output_text.clear()
        self.output_text.setExtraSelections([])
        self.rtttl_token_spans.clear()
        self.rtttl_token_times.clear()
        self.active_rtttl_token_index = None
        self.selected_name.setText("No version selected")
        self.settings_summary.clear()
        self.piano_roll.clear()

    def _selected_result(self) -> Optional[VariationResult]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self.results):
            return self.results[row]
        return None

    def _result_selected(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        if self.transport_kind == "rtttl" and self.player.has_media:
            self.transport_mapping = None
            self.transport_kind = None
            self.player.stop()
        settings = result.profile.settings
        self.editor_grid = max(1, int(settings.grid))
        self.selected_name.setText(f"{result.profile.name} - {result.profile.character}")
        self.settings_summary.setText(
            f"{self.current_analysis.source_mode if self.current_analysis else 'unknown'} source | "
            f"evidence {result.match_score * 100:.0f}% | {settings.decoder_mode} decoder | "
            f"confidence {settings.min_confidence:.2f} | min note {settings.minimum_note_ms:.0f} ms | "
            f"change cost {settings.note_change_penalty:.2f} | short-run floor {settings.minimum_melody_run_steps}"
        )
        self.output_text.setPlainText(result.rtttl)
        if self.current_analysis is not None:
            self.piano_roll.set_comparison(
                self.current_analysis,
                result.rtttl,
                grid=result.profile.settings.grid,
            )
        self._prepare_rtttl_animation(result.rtttl)
        self._prepare_editor(result)

    def _prepare_rtttl_animation(self, rtttl: str) -> None:
        self.rtttl_token_spans.clear()
        self.rtttl_token_times.clear()
        self.active_rtttl_token_index = None
        self.output_text.setExtraSelections([])
        self.piano_roll.set_active_token(None)
        try:
            tokens = parse_rtttl_tokens(rtttl, grid=self.editor_grid)
        except ValueError:
            return

        first_colon = rtttl.find(":")
        second_colon = rtttl.find(":", first_colon + 1)
        if second_colon < 0:
            return
        melody_start = second_colon + 1
        character = melody_start
        for raw_token in rtttl[melody_start:].split(","):
            self.rtttl_token_spans.append((character, character + len(raw_token)))
            character += len(raw_token) + 1

        cursor = 0.0
        for token in tokens:
            self.rtttl_token_times.append((cursor, cursor + token.seconds))
            cursor += token.seconds
        usable = min(len(self.rtttl_token_spans), len(self.rtttl_token_times))
        self.rtttl_token_spans = self.rtttl_token_spans[:usable]
        self.rtttl_token_times = self.rtttl_token_times[:usable]

    def _set_active_rtttl_token(self, token_index: Optional[int]) -> None:
        if token_index == self.active_rtttl_token_index:
            return
        self.active_rtttl_token_index = token_index
        self.piano_roll.set_active_token(token_index)
        self._refresh_rtttl_text_highlights()

    def _update_rtttl_animation(
        self,
        media_position: float,
        waveform_position: float,
    ) -> None:
        if not self.rtttl_token_times:
            self._set_active_rtttl_token(None)
            return
        total_seconds = self.rtttl_token_times[-1][1]
        if self.transport_kind == "rtttl":
            token_position = media_position
        elif self.transport_kind == "source":
            waveform_span = max(0.001, self.waveform.end - self.waveform.start)
            ratio = clamp(
                (waveform_position - self.waveform.start) / waveform_span,
                0.0,
                1.0,
            )
            token_position = ratio * total_seconds
        else:
            self._set_active_rtttl_token(None)
            return

        active_index = None
        if token_position < total_seconds:
            for index, (start, end) in enumerate(self.rtttl_token_times):
                if start <= token_position < end:
                    active_index = index
                    break
        self._set_active_rtttl_token(active_index)

    def _transport_focus_blocked(self) -> bool:
        focus = QApplication.focusWidget()
        return isinstance(
            focus,
            (
                QLineEdit,
                QPlainTextEdit,
                QDoubleSpinBox,
                QComboBox,
                QSlider,
                QTableWidget,
                QPushButton,
            ),
        )

    def keyPressEvent(self, event) -> None:
        if self._transport_focus_blocked():
            super().keyPressEvent(event)
            return
        modifiers = event.modifiers()
        command_modifiers = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if modifiers & command_modifiers:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.toggle_transport()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            amount = 5.0 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1.0
            self.seek_transport(-amount if key == Qt.Key.Key_Left else amount)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self.seek_transport_bound(to_end=False)
            event.accept()
            return
        if key == Qt.Key.Key_End:
            self.seek_transport_bound(to_end=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def _player_position_changed(self, media_position: float) -> None:
        if self.transport_mapping is None or not hasattr(self, "waveform"):
            return
        media_start, media_end, waveform_start, waveform_end = self.transport_mapping
        media_span = max(0.001, media_end - media_start)
        ratio = clamp((media_position - media_start) / media_span, 0.0, 1.0)
        waveform_position = waveform_start + ratio * (waveform_end - waveform_start)
        self.waveform.set_playhead(waveform_position)
        self._update_rtttl_animation(media_position, waveform_position)

    def _player_playing_changed(self, playing: bool) -> None:
        if hasattr(self, "toggle_play_button"):
            self.toggle_play_button.setText("Pause" if playing else "Play / Pause")

    def _playhead_committed(self, waveform_position: float) -> None:
        if self.transport_mapping is not None and self.player.has_media:
            media_start, media_end, waveform_start, waveform_end = self.transport_mapping
            waveform_span = max(0.001, waveform_end - waveform_start)
            ratio = clamp((waveform_position - waveform_start) / waveform_span, 0.0, 1.0)
            self.player.seek_to(media_start + ratio * (media_end - media_start))
        self._set_status(f"Playhead: {format_time(self.waveform.playhead)}", "Ready")

    def toggle_transport(self) -> None:
        if self.player.has_media and self.transport_mapping is not None:
            playing = self.player.toggle_pause()
            state = "Playing" if playing else "Paused"
            self._set_status(f"{state} at {format_time(self.waveform.playhead)}.", state)
            return
        if self.audio:
            self.play_source()

    def seek_transport(self, waveform_seconds: float) -> None:
        if not self.audio:
            return
        if self.transport_mapping is not None and self.player.has_media:
            media_start, media_end, waveform_start, waveform_end = self.transport_mapping
            waveform_span = max(0.001, waveform_end - waveform_start)
            media_delta = waveform_seconds * (media_end - media_start) / waveform_span
            self.player.seek_relative(media_delta)
        else:
            target = clamp(
                self.waveform.playhead + waveform_seconds,
                self.waveform.start,
                self.waveform.end,
            )
            self.waveform.set_playhead(target)
        self._set_status(f"Playhead: {format_time(self.waveform.playhead)}", "Seeking")

    def seek_transport_bound(self, *, to_end: bool) -> None:
        if not self.audio:
            return
        if self.transport_mapping is not None and self.player.has_media:
            media_start, media_end, _waveform_start, _waveform_end = self.transport_mapping
            self.player.seek_to(media_end if to_end else media_start)
        else:
            self.waveform.set_playhead(self.waveform.end if to_end else self.waveform.start)
        self._set_status(f"Playhead: {format_time(self.waveform.playhead)}", "Seeking")

    def stop_transport(self) -> None:
        self.player.stop()
        self._set_active_rtttl_token(None)
        self._set_status("Playback stopped at the selection start.", "Stopped")

    def play_source(self) -> None:
        if not self.audio:
            self._task_error("Open an audio file first.")
            return
        start = self.waveform.start
        end = self.waveform.end
        position = self.waveform.playhead if start <= self.waveform.playhead < end else start
        self.waveform.set_playhead(position)
        self.transport_mapping = (start, end, start, end)
        self.transport_kind = "source"
        self.player.play_file(self.audio.path, start, end, position=position)
        self._set_status("Playing the selected source excerpt.", "Playing source")

    def play_selected(self) -> None:
        result = self._selected_result()
        if result is None:
            self._task_error("Select a generated version first.")
            return
        try:
            samples = synthesize_rtttl(result.rtttl, sample_rate=SAMPLE_RATE)
            media_duration = max(0.05, float(samples.size / SAMPLE_RATE))
            self.transport_mapping = (
                0.0,
                media_duration,
                self.waveform.start,
                self.waveform.end,
            )
            self.transport_kind = "rtttl"
            self.player.play_samples(samples, SAMPLE_RATE)
        except Exception as exc:
            self._task_error(str(exc))
            return
        self._set_status(f"Playing {result.profile.name}.", "Playing RTTTL")

    def copy_selected(self) -> None:
        result = self._selected_result()
        if result is None:
            self._task_error("Select a generated version first.")
            return
        QApplication.clipboard().setText(result.rtttl)
        self._set_status(f"Copied {result.profile.name} to the clipboard.", "Copied")

    def use_selected_as_center(self) -> None:
        result = self._selected_result()
        if result is None:
            self._task_error("Select a generated version first.")
            return
        self.detail_slider.setValue(round(result.profile.detail * 100))
        self._settings_changed()
        self._set_status(
            f"{result.profile.name} is now the center. Generate again for nearby alternatives.",
            "Center updated",
        )

    def save_selected(self) -> None:
        result = self._selected_result()
        if result is None:
            self._task_error("Select a generated version first.")
            return
        title = mp3_to_rtttl.sanitize_title(self.title_edit.text())
        initial = f"{title}_{safe_file_part(result.profile.name).lower()}.rtttl"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save RTTTL",
            initial,
            "RTTTL (*.rtttl);;Text (*.txt)",
        )
        if path:
            try:
                Path(path).write_text(result.rtttl + "\n", encoding="utf-8")
            except OSError as exc:
                self._task_error(f"Could not save {Path(path).name}: {exc}")
                return
            self._set_status(f"Saved {Path(path).name}.", "Saved")

    def save_all(self) -> None:
        if not self.results:
            self._task_error("Generate versions first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save all versions", str(Path.cwd()))
        if not folder:
            return
        output_dir = Path(folder)
        title = mp3_to_rtttl.sanitize_title(self.title_edit.text())
        targets: list[tuple[Path, VariationResult]] = []
        for index, result in enumerate(self.results, start=1):
            name = safe_file_part(result.profile.name).lower()
            targets.append((output_dir / f"{title}_{index:02d}_{name}.rtttl", result))

        existing = [path.name for path, _result in targets if path.exists()]
        if existing:
            preview = ", ".join(existing[:3])
            if len(existing) > 3:
                preview += f", and {len(existing) - 3} more"
            answer = QMessageBox.question(
                self,
                "Replace existing RTTTL files?",
                f"The following files already exist: {preview}. Replace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        for path, result in targets:
            try:
                path.write_text(result.rtttl + "\n", encoding="utf-8")
            except OSError as exc:
                self._task_error(f"Could not save {path.name}: {exc}")
                return
        self._set_status(f"Saved {len(self.results)} versions to {output_dir}.", "Saved all")

    def closeEvent(self, event) -> None:
        self.player.cleanup()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native MP3 to RTTTL variation studio")
    parser.add_argument("input", nargs="?", type=Path, help="Optional audio file to open")
    args, _qt_args = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APPLICATION_NAME)
    app.setStyle("Fusion")
    window = StudioWindow()
    window.show()
    if args.input:
        QTimer.singleShot(0, lambda: window.load_audio(args.input))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
