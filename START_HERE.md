# RTTTL Studio

One local webpage for **audio conversion → audition/editing → AI continuation → export**.

The converter lives here. The shared browser application remains in
[`speccy88/rtttl-generative-transformer`](https://github.com/speccy88/rtttl-generative-transformer).
Use the matching `feat/rtttl-studio` branches in both repositories while reviewing
this feature, then their default branches after the changes are merged.

## 1. Put the repositories beside each other

```text
your-projects/
  mp3_to_rtttl/
  rtttl-generative-transformer/
    web/
```

Keep your existing checkouts if you already have them. A different location is also
supported through `--web-root /path/to/rtttl-generative-transformer/web/dist`.

## 2. Install dependencies once

In `mp3_to_rtttl`, use the Python 3.11 environment from the existing installation
instructions. Activate it, then:

```sh
python -m pip install -r requirements.txt
```

In `rtttl-generative-transformer/web`, with a supported Node.js/npm installation:

```sh
npm ci
```

The browser app does not need the optional native ONNX CUDA download. If `npm ci`
fails in `onnxruntime-node` while contacting NuGet, `npm ci --ignore-scripts` is a
tested fallback for this browser build. The build command still explicitly copies
the required browser runtime files.

## 3. Start the studio

From `mp3_to_rtttl`, with its Python environment activated:

```sh
python rtttl_studio.py --build-web
```

Open **http://127.0.0.1:8765/**. The build flag sets the correct local asset path on
Windows, macOS and Linux without shell-specific environment-variable syntax.
After the first build, `python rtttl_studio.py` starts the saved build directly.
Use `--port 8766` if the default port is busy. Ctrl+C stops the service.

## 4. Work with a recognizable phrase

1. Choose **Convert audio** and load an MP3, WAV, FLAC, M4A or OGG file (up to 32 MiB).
2. Select **0.25–30 seconds**, preferably a clear hook. Use **Listen to excerpt**.
3. Enter a known BPM if you have one. Automatic tempo can be half or double time.
4. Start with **Faithful / detailed**. Try **Full mix** for a clean instrument,
   solo recording or already separated lead/vocal stem. Harmonic mode reduces
   percussion but retains pitched accompaniment.
5. Convert once and compare the distinct interpretations in the shared workspace.
   The percentage is **source evidence**, not an accuracy probability.
6. Open **Edit the selected melody** to correct RTTTL, transpose or change tempo.
   Save a new version to preserve the original.
7. **Continue selected tune** keeps the existing notes, rests, rhythm and tempo,
   then asks the browser transformer for additional events. The length control
   now means additional events. New-song composition remains available separately.
8. Export RTTTL text, WAV or MP3. The latest 64 scores persist in browser storage;
   export important work for a durable backup.

## What runs where

| Component | Execution location |
| --- | --- |
| Decode and Basic Pitch transcription | Local Python process, CPU ONNX Runtime |
| Candidate selection and rhythm cleanup | Local Python process |
| Transformer composition/continuation | Browser worker; existing WebGPU/WASM policy |
| Editing, score library, playback and audio export | Browser |
| Model/runtime downloads | Existing pinned model and package sources |

The HTTP service listens only on loopback. Audio uploads go to that local process,
are stored under random temporary names, and are deleted after success or failure.
One conversion runs at a time. Recent results remain in memory for up to 15 minutes.
The service uses Python's standard library and does not require FastAPI or Qt imports.

The public GitHub Pages version can compose, continue, edit and export. **Audio
conversion requires this local service**, because GitHub Pages cannot execute the
Python transcription pipeline. Starting the local studio gives both engines the
same webpage and origin; no cross-site localhost permissions or iframe are needed.

## Evidence and limits

Read [the verification record](docs/STUDIO_VALIDATION.md) and
[the feature specification](docs/RTTTL_STUDIO_SPEC.md). The changes address observed
timing/cleanup failures and are tested with generated audio and the actual models.
Dense mixes can still select accompaniment instead of the intended lead. No user's
commercial recording was supplied or used as a recognition benchmark.

The generator was trained on symbolic RTTTL. It creates a continuation; it cannot
verify which notes were in the original audio. Imported 64th notes can play, but
continuation explicitly rejects them because the model vocabulary ends at 32nds.
