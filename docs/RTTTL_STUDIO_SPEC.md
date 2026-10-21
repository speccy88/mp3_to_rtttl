# RTTTL Studio: cross-repository contract

The complete studio runs **only on GitHub Pages**, without a local server. The
canonical behavior, architecture, acceptance criteria and implementation plan are
in [the generator repository's Studio specification](https://github.com/speccy88/rtttl-generative-transformer/blob/main/docs/RTTTL_STUDIO.md).

## This repository's responsibility

- Maintain the native desktop application and Python CLI.
- Preserve source timing, supported short attacks and rests during RTTTL rendering.
- Score candidate agreement on the source timeline without hiding tempo errors.
- Keep auditable synthetic regression fixtures and original-revision comparisons.
- Supply reference behavior for the JavaScript port; do not require a Python process
  for the webpage.

## Browser repository's responsibility

- Decode supported audio files on the visitor's device and run real Basic Pitch
  ONNX inference through browser WASM.
- Provide waveform excerpt selection, detected melody selection and interactive
  note/phrase selection in one editor.
- Preserve versions while composing, continuing, editing or varying a score.
- Offer bounded selected-note repair suggestions using surrounding musical context,
  explicitly distinguishing causal candidate search from trained masked infilling.
- Keep source audio on the device and deploy static assets through GitHub Pages.

The earlier loopback HTTP service/API, setup instructions and server-specific tests
have been removed. Converter quality evidence remains applicable to the Python core;
browser inference and user interaction have separate checks in the web repository.
