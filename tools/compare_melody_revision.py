#!/usr/bin/env python3
"""Compare a trusted Git revision's renderer with the working tree on shared evidence.

Each synthetic fixture is analyzed once. Both renderers receive the same ONNX
output and note extractor, so the measurements isolate decoder/rhythm changes.
All generated audio and the imported baseline module live in a temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mp3_to_rtttl as converter  # noqa: E402
from rtttl_utils import rtttl_frames  # noqa: E402
from tools.evaluate_rtttl_quality import (  # noqa: E402
    FIXTURES,
    SAMPLE_RATE,
    expected_frames,
    octave_adjusted_midi_accuracy,
    onset_f1,
    pitch_class_accuracy,
    synthesize_fixture,
)

FIXTURE_NAMES = (
    "short_notes_and_rests",
    "short_passing_notes",
    "gated_repeated_notes",
)
EVIDENCE_FILES = (
    "mp3_to_rtttl.py",
    "rtttl_utils.py",
    "tests/test_melody_fidelity.py",
    "tools/evaluate_rtttl_quality.py",
    "tools/compare_melody_revision.py",
)


def git(*arguments: str) -> str:
    """Read the local repository without fetching or changing it."""
    try:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            exc.stderr.strip() or "Git could not read the requested revision."
        ) from exc


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(ref: str, directory: Path):
    """Import converter code from an existing trusted local commit."""
    commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    source = git("show", f"{commit}:mp3_to_rtttl.py")
    path = directory / "baseline_converter.py"
    path.write_text(source + "\n", encoding="utf-8")
    name = f"rtttl_baseline_{commit}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the baseline converter module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.extract_note_events = converter.extract_note_events
    return commit, module


def score_fixture(fixture, rtttl: str, grid: int) -> dict:
    expected, expected_onsets = expected_frames(fixture, grid)
    actual, actual_onsets = rtttl_frames(rtttl, grid)
    pitch = pitch_class_accuracy(expected, actual)
    octave = octave_adjusted_midi_accuracy(expected, actual)
    onset = onset_f1(expected_onsets, actual_onsets)
    return {
        "composite": 0.55 * pitch + 0.25 * octave + 0.20 * onset,
        "pitch_class_accuracy": pitch,
        "octave_adjusted_midi_accuracy": octave,
        "onset_f1": onset,
        "rtttl": rtttl,
    }


def read_quality_gate(path: Path) -> dict:
    """Attach a previously captured gate result without rerunning inference."""
    output = path.read_text(encoding="utf-8")
    match = re.search(r"^current: average=([0-9.]+)$", output, re.MULTILINE)
    if match is None:
        raise ValueError("The quality-gate report must contain a current average.")
    section = output[match.end() :]
    metrics = []
    for match in re.finditer(
        r"^  ([\w-]+): score=([0-9.]+) pitch=([0-9.]+) "
        r"octave=([0-9.]+) onset=([0-9.]+) tokens=(\d+)$",
        section,
        re.MULTILINE,
    ):
        name, score, pitch, octave, onset, tokens = match.groups()
        metrics.append(
            {
                "fixture": name,
                "composite": float(score),
                "pitch_class_accuracy": float(pitch),
                "octave_adjusted_midi_accuracy": float(octave),
                "onset_f1": float(onset),
                "tokens": int(tokens),
            }
        )
    return {
        "command": "python tools/evaluate_rtttl_quality.py",
        "report_file": path.name,
        "report_sha256": file_hash(path),
        "current_average": float(
            re.search(r"^current: average=([0-9.]+)$", output, re.MULTILINE)[1]
        ),
        "required_average": 0.95,
        "fixtures": metrics,
        "note": "Captured earlier with the same revised sources; not rerun by this comparison.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref",
        required=True,
        help="Trusted local Git commit/ref to compare; no fetch occurs.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the complete JSON evidence to this path."
    )
    parser.add_argument(
        "--quality-gate-report",
        type=Path,
        help="Attach a previously captured full quality-gate log.",
    )
    args = parser.parse_args()
    selected = [fixture for fixture in FIXTURES if fixture.name in FIXTURE_NAMES]
    if len(selected) != len(FIXTURE_NAMES):
        parser.error("The quality-gate module is missing a required comparison fixture.")

    results = []
    with tempfile.TemporaryDirectory(prefix="rtttl_revision_") as temporary:
        directory = Path(temporary)
        baseline_commit, baseline = load_baseline(args.baseline_ref, directory)
        for fixture in selected:
            audio_path = directory / f"{fixture.name}.wav"
            synthesize_fixture(fixture, audio_path)
            analysis = converter.analyze_audio(
                audio_path, bpm_hint=fixture.bpm, source_mode="harmonic"
            )
            row = {"fixture": asdict(fixture)}
            for label, module in (("baseline", baseline), ("revised", converter)):
                settings = module.MelodySettings()
                rtttl = module.render_analysis(analysis, title=fixture.name, settings=settings)
                row[label] = score_fixture(fixture, rtttl, settings.grid)
            results.append(row)
            print(
                f"{fixture.name}: baseline={row['baseline']['composite']:.3f} "
                f"revised={row['revised']['composite']:.3f}",
                file=sys.stderr,
            )

    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "working_tree_head": git("rev-parse", "HEAD"),
        "working_tree_scope": "Modified renderer and analysis-envelope code, before commit.",
        "source_sha256": {name: file_hash(ROOT / name) for name in EVIDENCE_FILES},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("basic-pitch", "onnxruntime", "librosa", "numpy", "soundfile")
            },
        },
        "method": {
            "comparison": "Original-commit renderer versus revised renderer on shared ONNX evidence.",
            "inference_calls": len(selected),
            "sample_rate": SAMPLE_RATE,
            "source_mode": "harmonic",
            "tempo": "Known fixture BPM supplied to both renderers.",
            "grid": 16,
            "composite_formula": "0.55 * pitch_class + 0.25 * octave_adjusted_midi + 0.20 * onset_f1",
            "onset_tolerance_grid_steps": 1,
            "baseline_note_extractor": "Shared current extractor; unchanged by this revision.",
            "synthesizer": "tools/evaluate_rtttl_quality.py:synthesize_fixture",
        },
        "comparisons": results,
        "limits": [
            "Synthetic short melodies only; no real song or dense commercial mix was supplied.",
            "Known BPM removes automatic tempo estimation from these comparisons.",
            "Octave-adjusted accuracy permits one global whole-octave transposition.",
            "Onset F1 permits one grid step of timing error.",
            "The source-envelope veto detects global near-silence; it does not isolate a melody.",
            "Source-evidence ranking scores are not measured recognition accuracy.",
            "The quality gate's legacy row is a legacy parameter set using revised code, not this baseline commit.",
        ],
    }
    if args.quality_gate_report:
        report["quality_gate"] = read_quality_gate(args.quality_gate_report)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
