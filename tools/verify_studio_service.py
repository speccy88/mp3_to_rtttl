#!/usr/bin/env python3
"""Verify built browser assets and real MP3 transcription through the local API.

Requires the normal audio dependencies, FFmpeg, and the sibling browser build.
Synthesizes all input audio in a temporary directory; uses no user recordings.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtttl_studio import StudioServer  # noqa: E402
from rtttl_utils import parse_rtttl_tokens  # noqa: E402
from tools.evaluate_rtttl_quality import FIXTURES, synthesize_fixture  # noqa: E402


def verify(web_root: Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for this MP3 verification.")
    if not (web_root / "index.html").is_file():
        raise RuntimeError("Build the browser app first or provide --web-root.")
    with tempfile.TemporaryDirectory(prefix="studio-http-check-") as directory:
        temporary = Path(directory)
        wav = temporary / "generated.wav"
        mp3 = temporary / "generated.mp3"
        fixture = next(item for item in FIXTURES if item.name == "short_passing_notes")
        synthesize_fixture(fixture, wav)
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", str(mp3)],
            check=True,
        )
        uploads = temporary / "uploads"
        uploads.mkdir()
        server = StudioServer(("127.0.0.1", 0), web_root, uploads)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def request(method: str, path: str, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            headers = {"Content-Type": "application/octet-stream"} if body else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            connection.close()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {data.decode('utf-8')}")
            return response.status, data

        try:
            _, index = request("GET", "/")
            script = re.search(rb'<script[^>]+src="([^"]+)"', index)
            if not script:
                raise RuntimeError("The browser build has no entry script.")
            _, entry_script = request("GET", script[1].decode("utf-8"))
            if not entry_script:
                raise RuntimeError("The browser entry script is empty.")
            _, health_bytes = request("GET", "/api/health")
            health = json.loads(health_bytes)
            if not health["ready"]:
                raise RuntimeError(f"Missing audio dependencies: {health['missing']}")
            parameters = urlencode(
                {
                    "filename": mp3.name,
                    "duration": 8,
                    "bpm": fixture.bpm,
                    "source_mode": "full",
                    "preset": "faithful",
                }
            )
            status, accepted = request(
                "POST", "/api/transcriptions?" + parameters, mp3.read_bytes()
            )
            if status != 202:
                raise RuntimeError("Conversion did not return HTTP 202.")
            job_id = json.loads(accepted)["id"]
            deadline = time.monotonic() + 180
            while True:
                _, snapshot = request("GET", "/api/transcriptions/" + job_id)
                job = json.loads(snapshot)
                if job["status"] == "error":
                    raise RuntimeError(job["error"])
                if job["status"] == "complete":
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Conversion exceeded the verification timeout.")
                time.sleep(0.1)
            result = job["result"]
            if not result["candidates"]:
                raise RuntimeError("No candidates returned.")
            for candidate in result["candidates"]:
                tokens = parse_rtttl_tokens(candidate["rtttl"])
                if not any(token.midi is not None for token in tokens):
                    raise RuntimeError("A candidate contains only rests.")
            # Completion is published just before the job's unconditional cleanup.
            cleanup_deadline = time.monotonic() + 2
            while server.jobs.active and time.monotonic() < cleanup_deadline:
                time.sleep(0.01)
            if list(uploads.iterdir()):
                raise RuntimeError("The uploaded source was not cleaned up.")
            return {
                "status": "PASS",
                "scope": "Real generated WAV -> FFmpeg MP3 -> HTTP upload -> Basic Pitch ONNX -> RTTTL",
                "fixture": fixture.name,
                "static_entry_served": True,
                "health_ready": health["ready"],
                "uploaded_bytes": mp3.stat().st_size,
                "upload_removed": True,
                "result": result,
                "limits": "Synthetic audio; does not measure recognition of real full-mix songs or browser UI behavior.",
            }
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--web-root", type=Path, default=ROOT.parent / "rtttl-generative-transformer/web/dist"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = verify(args.web_root.resolve())
    text = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
