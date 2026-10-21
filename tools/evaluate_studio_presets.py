#!/usr/bin/env python3
"""Compare bounded Studio presets on one generated WAV and its MP3 encoding."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mp3_to_rtttl as converter  # noqa: E402
from rtttl_utils import parse_rtttl_tokens  # noqa: E402
from tools.compare_melody_revision import score_fixture  # noqa: E402
from tools.evaluate_rtttl_quality import FIXTURES, synthesize_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence destination.")
    args = parser.parse_args()
    fixture = next(item for item in FIXTURES if item.name == "short_passing_notes")
    base = converter.MelodySettings(minimum_melody_run_steps=1, octave_smoothing=False)
    variants = (
        ("fine_60_repeat1", 32, 60.0, 1),
        ("fine_100_repeat2", 32, 100.0, 2),
        ("standard_100_repeat2", 16, 100.0, 2),
        ("standard_60_repeat2", 16, 60.0, 2),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="rtttl_presets_") as temporary:
        wav = Path(temporary) / "fixture.wav"
        mp3 = Path(temporary) / "fixture.mp3"
        synthesize_fixture(fixture, wav)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", str(mp3)],
            check=True,
        )
        for path in (wav, mp3):
            analysis = converter.analyze_audio(path, bpm_hint=fixture.bpm, source_mode="full")
            for name, grid, minimum_ms, repeat_steps in variants:
                settings = replace(
                    base,
                    grid=grid,
                    minimum_note_ms=minimum_ms,
                    repeated_note_min_steps=repeat_steps,
                )
                rtttl = converter.render_analysis(analysis, title="Preset", settings=settings)
                row = {
                    "format": path.suffix[1:],
                    "encoded_bytes": path.stat().st_size,
                    "preset": name,
                    "grid": grid,
                    "minimum_note_ms": minimum_ms,
                    "repeated_note_min_steps": repeat_steps,
                    "tokens": len(parse_rtttl_tokens(rtttl)),
                    "source_evidence": converter.score_rtttl_fidelity(analysis, rtttl, grid),
                    **score_fixture(fixture, rtttl, 32),
                }
                results.append(row)
                print(
                    f"{row['format']:3} {name:24} score={row['composite']:.3f} "
                    f"onset={row['onset_f1']:.3f} tokens={row['tokens']}",
                    file=sys.stderr,
                )
    report = {
        "fixture": fixture.name,
        "expected_note_events": len(fixture.notes),
        "known_bpm": fixture.bpm,
        "analysis_source": "full",
        "inference_calls": 2,
        "mp3_encoder": "ffmpeg -v error -y -i fixture.wav -codec:a libmp3lame fixture.mp3",
        "shared_settings": {"minimum_melody_run_steps": 1, "octave_smoothing": False},
        "evaluation_grid": 32,
        "onset_tolerance_steps": 1,
        "composite_formula": "0.55 * pitch_class + 0.25 * octave_adjusted_midi + 0.20 * onset_f1",
        "results": results,
        "limits": [
            "One synthetic short melody and two encodings; this is a focused default-setting check.",
            "All variants use the same 32nd-note evaluation grid and onset tolerance.",
            "Fine 32nd-note grids and low minimum durations expose model re-attacks and may stutter.",
            "Evidence ranking can reward those false re-attacks; audition is still necessary.",
            "A 16th-note grid cannot represent genuine 32nd-note detail or arbitrary timing.",
            "A 100 ms minimum note duration can discard genuinely faster notes.",
        ],
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
