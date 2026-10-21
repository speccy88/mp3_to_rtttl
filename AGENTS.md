# Development agreements

- Preserve the CLI and native Qt application. The combined browser application is
  implemented in the sibling generator repository and runs on GitHub Pages.
- Keep transcription separate from creative generation. Never present generated notes
  as notes recovered from the recording.
- Preserve supported attacks, rests and elapsed timing. Document intentional pitch or
  rhythm simplification and test it with known synthetic evidence.
- Run expensive analysis once per excerpt and reuse its evidence for candidates.
- Keep source audio local and temporary. Do not commit user recordings, model weights,
  caches, virtual environments or build output.
- Do not introduce a local server dependency for the browser studio. Its audio
  inference must run on the visitor's device through the static webpage.
- Use `docs/RTTTL_STUDIO_SPEC.md` for the cross-repository behavior and acceptance
  criteria. Update the verification record with actual results and limitations.
- Follow the existing lightweight checks and synthetic inference gate in
  `CONTRIBUTING.md` for changes to the converter.
