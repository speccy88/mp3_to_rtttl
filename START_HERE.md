# RTTTL Studio

## Use the combined webpage

Open **[RTTTL Studio](https://speccy88.github.io/rtttl-generative-transformer/)**.
The webpage is hosted on GitHub Pages and performs audio transcription, melody
generation, editing, playback and export on your device. No Python installation,
local server, account or API key is required for the browser studio.

The browser source lives in
[`speccy88/rtttl-generative-transformer/web`](https://github.com/speccy88/rtttl-generative-transformer/tree/main/web).
Its [studio specification](https://github.com/speccy88/rtttl-generative-transformer/blob/main/docs/RTTTL_STUDIO.md)
describes selection, variations, repair and browser-only acceptance criteria.

1. Choose **Convert audio** and select a short source recording in a supported format.
2. Select and audition an excerpt in the waveform. Supply known BPM when possible.
3. Compare melody candidates and choose which one to keep.
4. Select notes in the piano roll to edit, vary, or request repair suggestions.
5. Continue the selected score or compose a new tune with the existing generator.
6. Compare versions and export RTTTL, WAV or MP3.

Browser codec support differs by device. Dense mixes can still confuse melody
selection. Repair suggestions use a causal symbolic model and surrounding notes;
they are not a BERT-trained denoiser and cannot guarantee the original audio notes.

## Use the existing Python tools

This repository retains its native PySide6 editor and command-line converter.
They are useful independently of the webpage. Follow [README.md](README.md) for
Python 3.11 dependencies, `python mp3_to_rtttl_gui.py`, and CLI usage.

The converter improvements preserve supported short notes and rests, align attacks
and quantization, preserve excerpt timing, and avoid stretching away tempo errors
when ranking candidates. Their regression evidence is recorded in
[docs/STUDIO_VALIDATION.md](docs/STUDIO_VALIDATION.md).

The initial local-server integration has been removed in favor of the explicit
GitHub Pages-only requirement. Python is not part of the browser user workflow.
