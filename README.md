# MP3 to RTTTL Studio

Convert short audio excerpts into monophonic Nokia RTTTL ringtones. The project
includes a native PySide6 variation studio for interactive work and a command-line
converter for repeatable exports.

The converter runs Basic Pitch's bundled ONNX model locally, follows a likely lead
voice through the detected note evidence, and simplifies it into the pitch and rhythm
constraints of RTTTL.

## Highlights

- Analyze MP3, WAV, FLAC, M4A, and OGG files in a native desktop interface.
- Generate 9, 12, or 15 contrasting melody interpretations from one analysis.
- Re-render cached model output while tuning detail, continuity, lead preference,
  grid resolution, and candidate spread.
- Compare candidates by source evidence and melodic diversity.
- Audition the source and generated square-wave ringtone, edit notes on a piano roll,
  and save one version or a complete comparison set.
- Use the same conversion pipeline from the command line.

## Privacy and network behavior

Audio processing is local. After the Python dependencies are installed, the studio
does not upload audio, contact a transcription service, or start a web server. Package
installation may require network access, and the application reads only the audio
files you explicitly open.

## Requirements

- Python 3.11. This is the tested runtime.
- A platform supported by PySide6 and ONNX Runtime. macOS is the primary development
  platform; Linux and Windows behavior can vary with their audio and Qt multimedia
  backends.
- FFmpeg on `PATH` is optional but recommended for faster waveform previews. The
  studio falls back to librosa when FFmpeg is unavailable.

`requirements.txt` provides compatible dependency ranges for normal development.
`requirements.lock` records the exact Python 3.11 resolution used for reproducible
installs.

## Installation

Clone the repository, change into its directory, and choose one setup method.

### Standard `venv` and `pip`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an exact dependency resolution, install the lock file instead:

```bash
python -m pip install -r requirements.lock
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\activate`.

### `uv`

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For a reproducible install using the exact lock:

```bash
uv pip install --python .venv/bin/python -r requirements.lock
```

Windows users can replace `.venv/bin/python` with
`.venv\Scripts\python.exe` in commands that name the interpreter explicitly.

## Run the desktop studio

Start with an empty workspace:

```bash
python mp3_to_rtttl_gui.py
```

Or open an audio file immediately:

```bash
python mp3_to_rtttl_gui.py "/path/to/your-audio.mp3"
```

### Suggested workflow

1. Open an audio file and select a short, recognizable passage in the waveform.
2. Leave **Analysis source** on **Auto** initially. It can preserve the full mix for
   chiptune-like material or isolate harmonic content from a conventional mix.
3. Choose 9, 12, or 15 versions. Twelve is the default balance between variety and
   review time.
4. Move **Melody detail** left for a simpler hook, or right for more detected turns.
5. Raise **Continuity** when the result changes pitch too often.
6. Click **Analyze + generate**. The first pass runs ONNX inference; later changes to
   rendering controls can reuse the cached analysis.
7. Audition and edit promising versions, use **Use as center** to search around one,
   then save the selected RTTTL or the complete set.

The studio may return fewer versions than requested when candidates collapse to the
same RTTTL melody or a candidate cannot be rendered.

## Command-line usage

```bash
python mp3_to_rtttl.py "/path/to/your-audio.mp3" \
  --title Ringtone \
  --start 12 \
  --duration 6 \
  --source-mode auto \
  -o ringtone.rtttl
```

Run `python mp3_to_rtttl.py --help` for every option. Frequently useful controls
include:

- `--bpm` to override tempo detection.
- `--source-mode harmonic|full|auto` to choose the analysis source.
- `--grid 16|32` to choose rhythmic resolution.
- `--min-confidence`, `--onset-threshold`, `--frame-threshold`, and
  `--minimum-note-ms` to filter Basic Pitch evidence.
- `--top-voice-bias` to favor a lead above accompaniment.
- `--note-change-penalty` and `--minimum-melody-run-steps` to simplify a busy line.
- `--onset-change-bonus`, `--onset-change-relief`, and `--sustain-bonus` to balance
  detected attacks against a smoother, hummable contour.
- `--max-melody-span` and `--no-octave-smoothing` to control octave compaction.
- `--midi PATH` to write optional diagnostic MIDI output.

## Formats

The studio file picker accepts `.mp3`, `.wav`, `.flac`, `.m4a`, and `.ogg`. Actual
decode support depends on the codecs available through FFmpeg, librosa, and the local
platform. The command-line interface advertises MP3, WAV, FLAC, and M4A input.

The normal output is a UTF-8 text file containing one RTTTL tune. Diagnostic MIDI and
listening-oriented WAV files are optional generated artifacts and are not source
files.

## Limitations

- RTTTL is monophonic. Chords, accompaniment, percussion, and overlapping voices must
  be reduced to one line.
- Short passages with a clear lead usually work better than complete songs or dense
  mixes.
- The output is a melody interpretation, not a guaranteed note-for-note
  transcription.
- RTTTL's pitch range, duration vocabulary, and title length require simplification.
- Initial analysis can be CPU intensive. The GUI caches the four most recent excerpt
  analyses in memory, but the cache is not retained after the application closes.
- Preview decoding and playback depend on the codecs and multimedia services present
  on the local machine.

## Verification

The default quality gate is hermetic: it synthesizes its own known melodies in a
temporary directory and requires no checked-in audio.

```bash
python tools/evaluate_rtttl_quality.py
```

This full gate performs model inference and therefore requires the dependencies from
`requirements.txt` or `requirements.lock`.

### Optional real-excerpt checks

You can add one or more local excerpts that you are legally permitted to use. Each
argument supplies a display name, path, start time in seconds, and duration in seconds:

```bash
python tools/evaluate_rtttl_quality.py \
  --real-excerpt "My licensed hook" "/absolute/path/to/audio.wav" 12 6
```

Repeat `--real-excerpt` to evaluate several files. To create private source and
rendered WAVs for listening, combine it with an audition directory:

```bash
python tools/evaluate_rtttl_quality.py \
  --real-excerpt "My licensed hook" "/absolute/path/to/audio.wav" 12 6 \
  --write-audition-dir audition_exports
```

`audition_exports/` is generated local output and must not be committed unless every
included recording is explicitly licensed for redistribution.

## Development checks

The lightweight checks used by GitHub Actions do not install the transcription model
or use audio fixtures:

```bash
python -m pip install numpy ruff
ruff check .
ruff format --check .
python -m unittest discover -s tests -p "test_*.py" -v
python -m py_compile \
  mp3_to_rtttl.py \
  mp3_to_rtttl_gui.py \
  rtttl_utils.py \
  tools/evaluate_rtttl_quality.py
```

Run the full synthetic quality gate as an additional check when changing audio
analysis, melody decoding, scoring, or RTTTL rendering.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and design boundaries, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the development and review checklist.

## Project layout

- `mp3_to_rtttl.py` — analysis, melody decoding, RTTTL rendering, and CLI.
- `mp3_to_rtttl_gui.py` — PySide6 studio, previews, caching, editing, and playback.
- `rtttl_utils.py` — shared RTTTL parsing, frame expansion, and synthesis.
- `tools/evaluate_rtttl_quality.py` — synthetic and opt-in real-excerpt verifier.
- `tests/` — lightweight unit tests that do not require model inference.
- `requirements.txt` — compatible runtime dependency specification.
- `requirements.lock` — exact Python 3.11 dependency resolution.

## Troubleshooting

### The first analysis is slow

The first conversion decodes the selected audio and runs ONNX inference. Re-rendering
the same excerpt with different melody controls should be faster because the GUI can
reuse its in-memory analysis.

### Core ML or TensorFlow packages were installed

Basic Pitch can bring in Core ML or TensorFlow-related packages transitively on some
platforms. This application intentionally selects Basic Pitch's bundled ONNX model
and runs it through `onnxruntime` for consistent inference. The presence of another
backend package does not mean the application is using it.

### The waveform preview or playback fails

Install FFmpeg and ensure it is available on `PATH`, then verify that the local Qt
multimedia backend supports the file's codec. Converting the source to PCM WAV is a
useful compatibility check.

## Audio rights and generated melodies

Only process audio that you are authorized to use. A generated RTTTL melody can still
be a derivative of the source composition; conversion does not grant permission to
publish, distribute, or sell that result. Do not add commercial recordings, private
audio, or audition exports to this repository.

## License status

No open-source license has been granted for this repository yet. Copyright 2026 the
repository owner. All rights reserved. Public visibility does not grant permission to
copy, modify, redistribute, or use the source beyond rights provided by applicable
law or a separate written agreement.
