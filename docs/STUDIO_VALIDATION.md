# RTTTL Studio verification

## Converter fidelity

The original converter revision was `41f86342c024ad261c0256a50106a641fc492391`.
The comparison runs the original and revised renderers on identical Basic Pitch
ONNX evidence from generated audio. No private or commercial recordings are used.

| Generated case | Original | Revised |
| --- | ---: | ---: |
| Short notes, rests and initial silence | 0.550649 | 1.000000 |
| Short passing notes | 0.800000 | 1.000000 |
| Repeated notes with short rests | 0.800000 | 1.000000 |

Metric: 55% aligned pitch-class accuracy, 25% octave-adjusted pitch accuracy and
20% onset F1, with the verifier's documented alignment tolerance. These are
controlled regression cases, not an estimate of real-song recognition accuracy.

The six-case full ONNX gate passes at **1.000 average**, preserving the previous
0.95 threshold. See [the raw gate output](studio_quality_gate.txt) and
[the reproducible comparison evidence](studio_fidelity_results.json).
The gate's `legacy` row is an older parameter preset on the revised code; the
table above instead compares the actual original revision.

A separate synthetic tempo regression previously scored 60, 120 and 240 BPM
equally at 1.000 against a 120 BPM source. Revised source-evidence scores are
0.2138, 1.0000 and 0.2908. Timing is no longer normalized away before ranking.

The browser's faithful preset uses a 16th-note grid, a 100 ms minimum detected
note duration, a two-step repeated-note threshold, and no octave smoothing.
A bounded four-preset comparison on one known nine-event melody found that
32nd-note timing exposed false re-attacks. The selected preset retained all nine
events exactly in both generated WAV and MP3; the initial 32nd-note/60 ms preset
produced 16 and 19 events respectively. See
[the preset comparison](studio_preset_results.json) and reproduce it with
`python tools/evaluate_studio_presets.py`. This default cannot represent genuine
32nd-note detail, and the 100 ms filter may omit genuinely fast notes.

## Local service

- 27 Python unit/integration tests pass, including real loopback HTTP requests.
- Tests cover bounded options/uploads, active-job conflicts, result polling,
  success/failure cleanup, Host/Origin validation and static asset delivery.
- Ruff lint/format and Python syntax checks pass.
- An actual generated WAV was encoded to MP3 with FFmpeg, uploaded through HTTP,
  transcribed by Basic Pitch ONNX, and returned as parseable candidates. The
  browser entry asset was served and the temporary upload was removed. See
  [the service evidence](studio_service_results.json).

Reproduce the full service check after building the sibling browser application:

```sh
python tools/verify_studio_service.py --output /tmp/studio-service-check.json
```

## Browser application and generation

- All seven JavaScript unit-test files pass, including prefix conditioning,
  service-client errors, tempo/transpose edits and append semantics.
- Root-path production build passes with the existing locked dependency versions.
- Nine focused browser integration cases were added for conversion, fractional
  excerpt timing, editing, continuation, originals, errors, static fallback and a
  narrow viewport.
- Interactive browser smoke verified page rendering, RTTTL import, a transposed
  copy, original preservation after reload, playback controls, and the static
  site's converter setup state.
- Actual published ONNX weights were hash-verified and run through the continuation
  sampler with native CPU inference: eight source events retained and eight new
  events sampled, with 16 inference calls per run. Same-seed output matched.
  Evidence is in the generator's `results/studio_continuation_native.json`.

The internal browser preview uses plain HTTP on a non-localhost name, so Web Crypto
is unavailable and model hash verification cannot run there. The worker now gives
an actionable HTTPS/localhost message. No checksum check was bypassed. Native
inference establishes checkpoint/sampler compatibility; it does not establish the
browser WebGPU/WASM lifecycle. The supported local URL is a secure-context
loopback origin, and the public site uses HTTPS.

The browser regression suite passed on the paired generator pull request in
[GitHub Actions run 39520703116](https://github.com/speccy88/rtttl-generative-transformer/actions/runs/39520703116).
The converter's lightweight CI also passed in
[run 39520741292](https://github.com/speccy88/mp3_to_rtttl/actions/runs/39520741292).
No physical iPhone or user-device performance test has been performed in this
environment. The browser tests use mocked model/API responses where indicated;
the separate ONNX and real-HTTP checks above establish actual engine execution.

## Practical limits

- The model may still follow accompaniment in a dense mix. Harmonic/percussive
  separation is not lead/vocal separation.
- Fine rhythm grids and low note-duration filters can expose false attacks; compare
  the available interpretations and supply known BPM when possible.
- The silence veto identifies near-silence in the whole source. It cannot infer a
  lead's rests while accompaniment is playing.
- AI continuation is a separate creative action. It does not verify or correct the
  audio transcription.
