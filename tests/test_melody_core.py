"""Fast tests for melody quantization and RTTTL emission."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import mp3_to_rtttl
from rtttl_utils import parse_rtttl_tokens


class MelodyCoreTests(unittest.TestCase):
    def test_sanitizes_titles_and_normalizes_tempo(self) -> None:
        self.assertEqual(mp3_to_rtttl.sanitize_title("My Song! 2026"), "MySong2026")
        self.assertEqual(mp3_to_rtttl.sanitize_title("***"), "Ringtone")
        self.assertEqual(mp3_to_rtttl.choose_rtttl_bpm(math.nan), 120)
        self.assertEqual(mp3_to_rtttl.choose_rtttl_bpm(126, strict_presets=True), 125)

    def test_quantized_grid_preserves_attacks(self) -> None:
        events = [
            (0.0, 0.5, 60, 0.8),
            (0.5, 1.0, 62, 0.9),
        ]

        slots, onset_steps, onset_pitches = mp3_to_rtttl.build_quantized_melody_grid(
            events, bpm=120, grid=16
        )

        self.assertEqual(len(slots), 8)
        self.assertEqual(onset_steps, {0, 4})
        self.assertEqual(onset_pitches[0], {60: 0.8})
        self.assertEqual(onset_pitches[4], {62: 0.9})
        self.assertIn(60, slots[0])
        self.assertIn(62, slots[4])

    def test_emitted_rtttl_round_trips_grid_durations(self) -> None:
        rtttl = mp3_to_rtttl.runs_to_rtttl(
            [(60, 4), (None, 2), (62, 6)],
            title="Round trip",
            bpm=120,
            grid=16,
        )
        tokens = parse_rtttl_tokens(rtttl, grid=16)

        self.assertEqual(rtttl.split(":", 1)[0], "Roundtrip")
        self.assertEqual([token.midi for token in tokens], [60, None, 62])
        self.assertEqual([token.steps for token in tokens], [4, 2, 6])

    def test_repeated_onset_splits_a_sustained_pitch(self) -> None:
        runs = mp3_to_rtttl.collapse_runs(
            [60, 60, 60, 60],
            {2},
            repeated_note_min_steps=2,
            onset_pitches=[{60: 0.8}, {}, {60: 0.9}, {}],
        )

        self.assertEqual(runs, [(60, 2), (60, 2)])

    def test_cli_argument_validation(self) -> None:
        args = mp3_to_rtttl.parse_args(
            ["song.mp3", "--start", "1.5", "--duration", "6", "--bpm", "120"]
        )
        self.assertEqual(args.start, 1.5)
        self.assertEqual(args.duration, 6.0)
        self.assertEqual(args.bpm, 120.0)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mp3_to_rtttl.parse_args(["song.mp3", "--duration", "0"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mp3_to_rtttl.parse_args(["song.mp3", "--min-confidence", "1.1"])

    def test_rejects_invalid_encoding_inputs(self) -> None:
        with self.assertRaises(ValueError):
            mp3_to_rtttl.build_quantized_melody_grid([], bpm=0, grid=16)
        with self.assertRaises(ValueError):
            mp3_to_rtttl.runs_to_rtttl([(60, 0)], "Bad", bpm=120, grid=16)
        with self.assertRaises(ValueError):
            mp3_to_rtttl.midi_to_rtttl_note(40)


if __name__ == "__main__":
    unittest.main()
