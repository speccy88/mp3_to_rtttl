"""Tests for strict RTTTL parsing, timing, and preview synthesis."""

from __future__ import annotations

import unittest

import numpy as np

from rtttl_utils import parse_rtttl_tokens, rtttl_frames, synthesize_rtttl


class RtttlParserTests(unittest.TestCase):
    def test_parses_notes_rests_dots_and_octaves(self) -> None:
        tokens = parse_rtttl_tokens("Demo:d=4,o=5,b=120:8c,16d#6.,4p")

        self.assertEqual([token.midi for token in tokens], [72, 87, None])
        self.assertEqual([token.steps for token in tokens], [2, 2, 4])
        self.assertAlmostEqual(tokens[0].seconds, 0.25)
        self.assertAlmostEqual(tokens[1].seconds, 0.1875)
        self.assertAlmostEqual(tokens[2].seconds, 0.5)

    def test_expands_grid_frames_and_onsets(self) -> None:
        frames, onsets = rtttl_frames("Demo:d=4,o=5,b=120:4c,8p,8d")

        self.assertEqual(frames, [72] * 4 + [None] * 2 + [74] * 2)
        self.assertEqual(onsets, {4, 6})

    def test_rejects_invalid_defaults_and_trailing_token_text(self) -> None:
        invalid_values = (
            "Demo:d=0,o=5,b=120:4c",
            "Demo:d=4,o=5,b=0:4c",
            "Demo:d=4,o=10,b=120:4c",
            "Demo:d=4,o=5,b=120:4c5junk",
            "Demo:d=4,o=5,b=120:0c",
        )

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_rtttl_tokens(value)

    def test_synthesizes_mono_float32_audio(self) -> None:
        audio = synthesize_rtttl("Demo:d=4,o=5,b=120:4a,4p", sample_rate=8_000)

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 8_000)
        self.assertGreater(float(np.max(np.abs(audio[:4_000]))), 0.0)
        self.assertTrue(np.all(audio[4_000:] == 0.0))

    def test_rejects_invalid_grid_and_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            parse_rtttl_tokens("Demo:d=4,o=5,b=120:4c", grid=0)
        with self.assertRaises(ValueError):
            synthesize_rtttl("Demo:d=4,o=5,b=120:4c", sample_rate=0)


if __name__ == "__main__":
    unittest.main()
