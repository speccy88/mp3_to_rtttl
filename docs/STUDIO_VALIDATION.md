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

A reference faithful preset uses a 16th-note grid, a 100 ms minimum detected
note duration, a two-step repeated-note threshold, and no octave smoothing.
A bounded four-preset comparison on one known nine-event melody found that
32nd-note timing exposed false re-attacks. The selected preset retained all nine
events exactly in both generated WAV and MP3; the initial 32nd-note/60 ms preset
produced 16 and 19 events respectively. See
[the preset comparison](studio_preset_results.json) and reproduce it with
`python tools/evaluate_studio_presets.py`. This default cannot represent genuine
32nd-note detail, and the 100 ms filter may omit genuinely fast notes.

## Verification scope

The converter core and parser have 22 Python unit tests. The inference gate and
original-revision comparison above validate the Python implementation. The preset
comparison includes actual WAV and MP3 decoding with the same model evidence.
These results must not be copied as measured browser transcription accuracy.

The local HTTP service introduced in the initial draft has been removed after the
user clarified the GitHub Pages-only requirement. Its tests and setup instructions
are not part of the final application.

The independent JavaScript transcription port, interactive selection, variations
and repair are verified in the generator repository. See its
[Studio documentation](https://github.com/speccy88/rtttl-generative-transformer/blob/main/docs/RTTTL_STUDIO.md)
and the paired pull request for browser/native inference evidence and current CI.

## Practical limits

- The model may still follow accompaniment in a dense mix. Harmonic/percussive
  separation is not lead/vocal separation.
- Fine rhythm grids and low note-duration filters can expose false attacks; compare
  the available interpretations and supply known BPM when possible.
- The silence veto identifies near-silence in the whole source. It cannot infer a
  lead's rests while accompaniment is playing.
- AI continuation is a separate creative action. It does not verify or correct the
  audio transcription.
