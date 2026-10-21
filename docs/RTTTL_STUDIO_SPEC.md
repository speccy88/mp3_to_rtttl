# RTTTL Studio: feature specification and implementation plan

## Problem

Audio conversions often lose the recognizable melody. A separate generative browser
application already provides playback and exports, but cannot use converted notes as
a musical prompt. The user wants one page for conversion, editing and generation.

## Design

The existing `rtttl-generative-transformer/web` application is the shared front end.
This repository supplies a local Python service that serves its built files and
analyzes uploaded excerpts with the existing Basic Pitch ONNX pipeline. The model
for composing and continuing tunes remains in the browser. Both repositories retain
their independent command-line and desktop/training capabilities.

The shared interchange is RTTTL plus a score with `name`, `bpm`, and events containing
`pitch` (MIDI integer or null), `duration` (whole-note denominator), and `dotted`.
Conversion metadata records the excerpt, chosen source mode and candidate settings.

## User behavior

1. Open the local Studio webpage and select an audio file.
2. Choose a short excerpt, optionally supply the known tempo, and select a conversion
   preset and analysis source. Preview the original audio.
3. Analyze once, compare several interpretations and load a result into the shared
   player/library. Evidence scores indicate agreement with the transcription model,
   not measured recognition accuracy.
4. Edit RTTTL, change tempo or transpose as a new library entry.
5. Compose a new melody or explicitly continue the selected score. Continuation
   preserves the supplied events and tempo and samples additional notes.
6. Audition and export RTTTL, WAV or MP3 through the existing browser implementation.

## Boundaries

- No cloud audio upload, accounts, new hosted inference service or model retraining.
- No claim of perfect transcription of full mixes. HPSS separates harmonic and
  percussive content; it does not isolate the singer or lead instrument.
- No silent AI rewriting of a transcription. Continuations are identified separately.
- Preserve existing browser cache keys, licenses, model pins and device fallbacks.
- The public static GitHub Pages app can compose/edit/export independently; audio
  conversion requires running the same front end with this local Python service.

## Acceptance criteria

- The legacy converter and lightweight tests still work.
- Onset-backed short notes and real rests survive cleanup, and evidence scoring does
  not time-stretch an incorrectly timed candidate to make it match.
- Synthetic model-inference quality checks pass without weakening existing gates.
- A single local origin serves the studio and bounded asynchronous conversion API.
- Source uploads are removed after success or failure. At most one analysis runs at
  once; failed or concurrent requests have explicit outcomes.
- Converted and edited scores enter the same library as generated scores. Starting
  generation does not discard earlier work.
- Continuation feeds the prefix into actual model inference, preserves it exactly,
  and rejects unsupported note durations rather than silently coercing them.
- A missing converter never breaks the standalone composition app.
- Unit tests, production build and available end-to-end checks are reported honestly.

## Ordered implementation tasks

1. Inspect both pinned repository revisions and reproduce fidelity defects.
2. Fix evidence-preserving decoding and add deterministic regression coverage.
3. Add the local conversion service, bounded uploads, status API and serving command.
4. Add validated prefix conditioning to the existing browser sampler and worker.
5. Connect audio import, candidates, editing and continuation to the shared workspace.
6. Run unit tests, synthetic inference gate, production build and integration checks.
7. Publish paired review branches/pull requests with setup and validation evidence.

## Validation record

See `STUDIO_VALIDATION.md` for completed checks and remaining limitations.
