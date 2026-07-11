# Contributing

Thanks for helping improve MP3 to RTTTL Studio. The most useful contributions make the
melody output more recognizable, keep the desktop workflow responsive, or make the
conversion behavior easier to verify without private audio.

## License status

This repository does not yet have an open-source license. Public visibility does not
grant general permission to copy, modify, or redistribute the code. Contact the
repository owner before beginning a substantial contribution so that contribution and
licensing expectations can be agreed in writing.

Do not submit code, audio, melodies, or other material unless you have the right to
contribute it.

## Set up a development environment

Use Python 3.11. Follow the installation instructions in [README.md](README.md), then
install Ruff if it is not already available:

```bash
python -m pip install ruff
```

`requirements.txt` is appropriate for normal compatible installs.
`requirements.lock` provides the exact Python 3.11 resolution used for reproducible
testing.

## Development workflow

1. Open an issue before a broad behavioral or architectural change.
2. Create a focused branch from the current default branch.
3. Keep code, tests, and documentation changes in the same coherent workstream.
4. Run the checks below.
5. Describe the user-visible effect, validation performed, and remaining limitations
   in the pull request.

## Required lightweight checks

```bash
ruff check .
ruff format --check .
python -m unittest discover -s tests -p "test_*.py" -v
python -m py_compile \
  mp3_to_rtttl.py \
  mp3_to_rtttl_gui.py \
  rtttl_utils.py \
  tools/evaluate_rtttl_quality.py
```

GitHub Actions runs these checks with Python 3.11, NumPy, and Ruff. It intentionally
does not install the complete audio stack or perform model inference.

## Full quality gate

Changes to audio analysis, melody selection, scoring, rhythm, or RTTTL rendering must
also run the synthetic end-to-end verifier in a fully installed environment:

```bash
python tools/evaluate_rtttl_quality.py
```

The default gate generates all audio fixtures in a temporary directory. It must remain
repeatable without local music files or network access.

For a listening-oriented check, use audio you are authorized to process:

```bash
python tools/evaluate_rtttl_quality.py \
  --real-excerpt "Local check" "/absolute/path/to/audio.wav" 10 6 \
  --write-audition-dir audition_exports
```

Never commit the source recording or audition directory unless its license explicitly
allows redistribution and the repository owner has approved it.

## Code and documentation style

- Target Python 3.11 and keep type annotations accurate.
- Run Ruff rather than hand-formatting around the configured style.
- Prefer small functions with names that explain what they do.
- Write comments and docstrings for intent, invariants, scoring choices, threading, and
  non-obvious constraints. Avoid comments that merely repeat the next statement.
- Keep long-running decoding and inference work off the Qt main thread.
- Preserve the separation between expensive analysis and inexpensive rendering.
- Keep `rtttl_utils.py` independent of Qt and the model runtime.
- Update README or architecture documentation when a command, control, dependency, or
  data-flow contract changes.

## Tests

- Add lightweight unit tests for pure parsing, encoding, ranking, or settings logic.
- Use generated waveforms for automated audio tests.
- Keep tests deterministic and independent of user directories, clocks, network
  services, and platform-specific media libraries whenever possible.
- Do not weaken a regression threshold solely to make a change pass; explain and
  justify intentional baseline changes.
- Treat listening checks as supporting evidence, not a replacement for automated
  regression coverage.

## Dependency changes

Keep runtime dependencies in `requirements.txt`. Explain any new dependency, why the
standard library or an existing package is insufficient, and any platform impact.
When the compatible requirements change, regenerate `requirements.lock` using Python
3.11 and include both files in the same review.

## Pull request checklist

- [ ] The change is focused and its user-visible behavior is described.
- [ ] Ruff check and format check pass.
- [ ] Lightweight unit tests and syntax compilation pass.
- [ ] The synthetic quality gate passes when core conversion behavior changed.
- [ ] New behavior has appropriate tests and documentation.
- [ ] No private, commercial, or unlicensed audio is included.
- [ ] No generated audition, cache, virtual-environment, or local output files are
      included.
- [ ] Dependency and lock files are synchronized when dependencies changed.
- [ ] Known limitations or follow-up work are stated clearly.
