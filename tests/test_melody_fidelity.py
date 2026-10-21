"""Regression checks for melody and timing preservation without audio models."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mp3_to_rtttl as converter
from rtttl_utils import parse_rtttl_tokens, rtttl_frames


def reference_analysis() -> converter.TranscriptionAnalysis:
    """A two-second, four-note reference with known absolute onset times."""
    notes = np.zeros((200, 88), dtype=np.float32)
    onsets = np.zeros_like(notes)
    for index, pitch in enumerate((60, 62, 64, 65)):
        notes[index * 50 : (index + 1) * 50, pitch - 21] = 0.9
        onsets[index * 50, pitch - 21] = 0.9
    return converter.TranscriptionAnalysis(
        input_path=Path("reference.wav"),
        start=0,
        duration=2,
        estimated_bpm=120,
        sample_rate=22_050,
        source_mode="full",
        model_output={"note": notes, "onset": onsets},
    )


class MelodyFidelityTests(unittest.TestCase):
    def test_cleanup_preserves_attacked_short_notes_and_short_rests(self) -> None:
        melody = [60, 60, 62, 60, 60, None, 60, 60]
        attacks = [{60: 0.9}, {}, {62: 0.9}, {60: 0.9}, {}, {}, {60: 0.9}, {}]
        result = converter.remove_one_step_blips(melody, attacks)
        result = converter.merge_short_note_runs(result, 2, onset_pitches=attacks)
        self.assertEqual(result, melody)

    def test_cleanup_still_removes_unsupported_pitch_spikes(self) -> None:
        melody = [60, 60, 75, 60, 60]
        self.assertEqual(converter.remove_one_step_blips(melody), [60] * 5)
        self.assertEqual(converter.merge_short_note_runs(melody, 2), [60] * 5)

    def test_small_event_tails_do_not_extend_into_the_next_grid_cell(self) -> None:
        slots, _, _ = converter.build_quantized_melody_grid(
            [(0.0, 0.126, 60, 1.0), (0.125, 0.5, 62, 0.8)], bpm=120, grid=16
        )
        self.assertNotIn(60, slots[1])
        self.assertAlmostEqual(slots[1][62], 0.8)

    def test_notes_and_attacks_snap_to_the_same_grid(self) -> None:
        slots, onsets, _ = converter.build_quantized_melody_grid(
            [(0.242, 0.507, 60, 0.9)], bpm=120, grid=16
        )
        self.assertEqual(onsets, {2})
        self.assertEqual(slots[:2], [{}, {}])
        self.assertEqual(len(slots), 4)

    def test_fidelity_ranking_penalizes_half_and_double_time(self) -> None:
        analysis = reference_analysis()
        scores = {
            bpm: converter.score_rtttl_fidelity(analysis, f"Tune:d=4,o=4,b={bpm}:c,d,e,f")
            for bpm in (60, 120, 240)
        }
        self.assertGreater(scores[120], 0.8)
        self.assertGreater(scores[120], scores[60] + 0.25)
        self.assertGreater(scores[120], scores[240] + 0.25)

    def test_fidelity_ranking_penalizes_wrong_pitch(self) -> None:
        analysis = reference_analysis()
        correct = converter.score_rtttl_fidelity(analysis, "Tune:d=4,o=4,b=120:c,d,e,f")
        wrong = converter.score_rtttl_fidelity(analysis, "Tune:d=4,o=4,b=120:c#,d#,f#,g#")
        self.assertGreater(correct, wrong + 0.4)

    def test_ranking_handles_a_candidate_longer_than_the_source(self) -> None:
        score = converter.score_rtttl_fidelity(
            reference_analysis(), "Long:d=1,o=4,b=25:c,d,e,f,g,a,b,c5"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.1)

    def test_render_keeps_leading_internal_and_trailing_rests_on_source_timeline(self) -> None:
        raw_events = [(0.25, 0.75, 60, 0.9, None), (1.0, 1.75, 62, 0.9, None)]
        diagnostics: dict[str, float] = {}
        with patch.object(converter, "extract_note_events", return_value=(None, raw_events)):
            rtttl = converter.render_analysis(
                reference_analysis(),
                title="Timing",
                settings=converter.MelodySettings(octave_smoothing=False),
                diagnostics=diagnostics,
            )
        frames, _ = rtttl_frames(rtttl)
        expected_classes = [None] * 2 + [0] * 4 + [None] * 2 + [2] * 6 + [None] * 2
        self.assertEqual([p % 12 if p is not None else None for p in frames], expected_classes)
        self.assertAlmostEqual(sum(t.seconds for t in parse_rtttl_tokens(rtttl)), 2.0)
        self.assertEqual(diagnostics["source_offset_seconds"], 0.0)
        self.assertEqual(diagnostics["duration_seconds"], 2.0)

    def test_direct_collapse_call_can_still_trim_silence(self) -> None:
        self.assertEqual(converter.collapse_runs([None, 60, None], set()), [(60, 1)])
        self.assertEqual(
            converter.collapse_runs([None, 60, None], set(), trim_silence=False),
            [(None, 1), (60, 1), (None, 1)],
        )

    def test_source_silence_veto_preserves_quiet_notes(self) -> None:
        analysis = reference_analysis()
        analysis.model_output["source_rms"] = np.repeat([1.0, 0.02, 0.0, 0.5], 50)
        analysis.model_output["source_rms_hop_seconds"] = np.asarray(0.01)
        path = [60] * 16
        self.assertEqual(
            converter.preserve_source_silence(path, analysis, 120, 16),
            [60] * 8 + [None] * 4 + [60] * 4,
        )

    def test_source_silence_veto_is_optional_for_cached_older_analysis(self) -> None:
        path = [60] * 16
        self.assertEqual(
            converter.preserve_source_silence(path, reference_analysis(), 120, 16), path
        )


if __name__ == "__main__":
    unittest.main()
