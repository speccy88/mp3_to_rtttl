#!/usr/bin/env python3
"""Serve the shared RTTTL browser studio and a bounded local conversion API.

Build ``rtttl-generative-transformer/web`` with ``VITE_BASE_PATH=/`` first.
The HTTP layer uses only Python's standard library; audio inference uses the
same optional dependencies as the CLI and does not import Qt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import mimetypes
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlsplit

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_EXCERPT_SECONDS = 30.0
MAX_RECENT_JOBS = 8
JOB_TTL_SECONDS = 15 * 60
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
AUDIO_MODULES = ("numpy", "librosa", "soundfile", "basic_pitch", "onnxruntime")
PRESETS = ("faithful", "balanced", "smooth")


@dataclass(frozen=True)
class ConversionRequest:
    filename: str
    start: float = 0.0
    duration: float = 12.0
    bpm: float | None = None
    source_mode: str = "auto"
    preset: str = "faithful"


def parse_conversion_request(query: str) -> ConversionRequest:
    """Validate the public API without loading the audio/model dependencies."""
    values = parse_qs(query, keep_blank_values=True, max_num_fields=8)
    allowed = {"filename", "start", "duration", "bpm", "source_mode", "preset"}
    if set(values) - allowed or any(len(value) != 1 for value in values.values()):
        raise ValueError("Unknown or duplicate conversion options.")
    options = {key: value[0] for key, value in values.items()}
    filename = options.get("filename", "audio.wav")
    if not filename or len(filename) > 256 or "\x00" in filename:
        raise ValueError("Choose an audio filename of at most 256 characters.")
    if Path(filename).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError("Choose an MP3, WAV, FLAC, M4A or OGG file.")

    def number(key: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(options.get(key, str(default)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
        return value

    start = number("start", 0.0, 0.0, 86_400.0)
    duration = number("duration", 12.0, 0.25, MAX_EXCERPT_SECONDS)
    bpm = number("bpm", 120.0, 25.0, 900.0) if options.get("bpm") else None
    source_mode = options.get("source_mode", "auto")
    preset = options.get("preset", "faithful")
    if source_mode not in {"auto", "full", "harmonic"}:
        raise ValueError("source_mode must be auto, full or harmonic.")
    if preset not in PRESETS:
        raise ValueError("preset must be faithful, balanced or smooth.")
    return ConversionRequest(filename, start, duration, bpm, source_mode, preset)


def conversion_presets():
    """Small interpretable settings families; none is an accuracy guarantee."""
    from mp3_to_rtttl import MelodySettings

    balanced = MelodySettings()
    return {
        "faithful": replace(
            balanced,
            grid=16,
            minimum_note_ms=100.0,
            minimum_melody_run_steps=1,
            repeated_note_min_steps=2,
            octave_smoothing=False,
        ),
        "balanced": balanced,
        "smooth": replace(
            balanced,
            minimum_melody_run_steps=3,
            note_change_penalty=0.22,
            sustain_bonus=0.52,
        ),
    }


def transcribe(path: Path, request: ConversionRequest, progress: Callable[[str], None]):
    """Run inference once, then render diverse candidates from the same evidence."""
    import mp3_to_rtttl as converter

    analysis = converter.analyze_audio(
        path,
        start=request.start,
        duration=request.duration,
        bpm_hint=request.bpm,
        source_mode=request.source_mode,
        progress=progress,
    )
    selected = conversion_presets()[request.preset]
    profiles = [
        ("Detailed lead" if request.preset == "faithful" else request.preset.title(), selected),
        ("Upper voice", replace(selected, decoder_mode="upper-events")),
        ("Continuous contour", replace(selected, decoder_mode="contour")),
    ]
    candidates = []
    signatures = set()
    warnings = []
    for label, settings in profiles:
        progress(f"Rendering {label.lower()}...")
        diagnostics = {}
        try:
            rtttl = converter.render_analysis(
                analysis,
                title=Path(request.filename).stem,
                settings=settings,
                bpm_override=request.bpm,
                diagnostics=diagnostics,
            )
            signature = rtttl.split(":", 1)[-1]
            if signature in signatures:
                continue
            signatures.add(signature)
            score = float(converter.score_rtttl_fidelity(analysis, rtttl, grid=settings.grid))
            candidates.append(
                {
                    "name": label,
                    "rtttl": rtttl,
                    "score": score if math.isfinite(score) else 0.0,
                    "settings": asdict(settings),
                    "diagnostics": diagnostics,
                }
            )
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{label}: {exc}")
    if not candidates:
        raise ValueError(
            "No melody survived. Try a clearer lead/vocal stem, another excerpt, "
            "or a different analysis source. " + " ".join(warnings)
        )
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    warnings.append(
        "Source evidence scores measure agreement with detected notes, not recognition "
        "accuracy. Compare the original excerpt by ear."
    )
    if request.bpm is None:
        warnings.append("Automatic tempo can be half or double time. Set the known BPM if needed.")
    if analysis.source_mode == "harmonic":
        warnings.append("Harmonic separation reduces percussion; it does not isolate the lead.")
    return {
        "candidates": candidates,
        "analysis": {
            "estimated_bpm": float(analysis.estimated_bpm),
            "source_mode": analysis.source_mode,
            "start": analysis.start,
            "duration": analysis.duration,
        },
        "warnings": warnings,
    }


class JobStore:
    """One active analysis, bounded recent results, and unconditional upload cleanup."""

    def __init__(self, converter=transcribe):
        self.converter = converter
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.active: str | None = None

    def reserve(self) -> str | None:
        with self.lock:
            if self.active is not None:
                return None
            now = time.monotonic()
            self.jobs = {
                key: job for key, job in self.jobs.items() if now - job["created"] < JOB_TTL_SECONDS
            }
            while len(self.jobs) >= MAX_RECENT_JOBS:
                self.jobs.pop(next(iter(self.jobs)))
            job_id = uuid.uuid4().hex
            self.active = job_id
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "progress": "Receiving audio...",
                "created": now,
            }
            return job_id

    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job_id != self.active and time.monotonic() - job["created"] >= JOB_TTL_SECONDS:
                self.jobs.pop(job_id)
                return None
            return {key: value for key, value in job.items() if key != "created"}

    def update(self, job_id: str, **changes) -> None:
        with self.lock:
            self.jobs[job_id].update(changes)

    def fail_upload(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
            if self.active == job_id:
                self.active = None

    def start(self, job_id: str, path: Path, request: ConversionRequest) -> None:
        threading.Thread(target=self.run, args=(job_id, path, request), daemon=True).start()

    def run(self, job_id: str, path: Path, request: ConversionRequest) -> None:
        try:
            self.update(job_id, status="running", progress="Preparing conversion...")
            result = self.converter(
                path, request, lambda message: self.update(job_id, progress=str(message))
            )
            self.update(job_id, status="complete", progress="Ready to compare.", result=result)
        except Exception as exc:
            self.update(job_id, status="error", progress="Conversion failed.", error=str(exc))
        finally:
            path.unlink(missing_ok=True)
            with self.lock:
                if self.active == job_id:
                    self.active = None


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, web_root: Path, upload_root: Path, jobs: JobStore | None = None):
        self.web_root = web_root.resolve()
        self.upload_root = upload_root
        self.jobs = jobs or JobStore()
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioServer
    server_version = "RTTTLStudio/1"

    def log_message(self, format, *args):
        # Requests include audio filenames; keep them out of ordinary access logs.
        pass

    def local_request(self) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        if host not in allowed:
            self.json_response(HTTPStatus.FORBIDDEN, {"error": "Use the local Studio address."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self.json_response(HTTPStatus.FORBIDDEN, {"error": "Open Studio on this local server."})
            return False
        return True

    def json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self.local_request():
            return
        route = urlsplit(self.path).path
        if route == "/api/health":
            missing = [name for name in AUDIO_MODULES if importlib.util.find_spec(name) is None]
            self.json_response(
                HTTPStatus.OK,
                {
                    "service": "mp3-to-rtttl",
                    "version": 1,
                    "ready": not missing,
                    "missing": missing,
                    "max_upload_bytes": MAX_UPLOAD_BYTES,
                    "max_excerpt_seconds": MAX_EXCERPT_SECONDS,
                    "presets": list(PRESETS),
                },
            )
        elif route.startswith("/api/transcriptions/"):
            result = self.server.jobs.snapshot(route.rsplit("/", 1)[-1])
            self.json_response(
                HTTPStatus.OK if result else HTTPStatus.NOT_FOUND,
                result or {"error": "Conversion was not found or has expired."},
            )
        elif route.startswith("/api/"):
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Unknown API route."})
        else:
            self.serve_file(route)

    def serve_file(self, route: str) -> None:
        # Accept the existing GitHub Pages build as well as a root-path build.
        prefix = "/rtttl-generative-transformer/"
        if route.startswith(prefix):
            route = "/" + route[len(prefix) :]
        path = (self.server.web_root / unquote(route).lstrip("/")).resolve()
        if not path.is_relative_to(self.server.web_root):
            self.json_response(HTTPStatus.FORBIDDEN, {"error": "Invalid asset path."})
            return
        if path.is_dir():
            path = (path / "index.html").resolve()
        if not path.is_relative_to(self.server.web_root):
            self.json_response(HTTPStatus.FORBIDDEN, {"error": "Invalid asset path."})
            return
        if not path.is_file():
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Asset not found."})
            return
        self.send_response(HTTPStatus.OK)
        media_type = "application/wasm" if path.suffix == ".wasm" else mimetypes.guess_type(path)[0]
        if path.suffix in {".js", ".mjs"}:
            media_type = "text/javascript"
        self.send_header("Content-Type", media_type or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with path.open("rb") as source:
                while data := source.read(64 * 1024):
                    self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.local_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path != "/api/transcriptions":
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "Unknown API route."})
            return
        try:
            request = parse_conversion_request(parsed.query)
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Upload with a Content-Length header.")
            content_type = self.headers.get_content_type()
            if content_type != "application/octet-stream" and not content_type.startswith("audio/"):
                raise ValueError("Upload the audio bytes as application/octet-stream.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self.json_response(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "Choose a nonempty audio file no larger than 32 MiB."},
                )
                return
        except (TypeError, ValueError) as exc:
            self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        job_id = self.server.jobs.reserve()
        if job_id is None:
            self.json_response(
                HTTPStatus.CONFLICT,
                {"error": "A conversion is already running. Wait for it to finish."},
            )
            return
        path = self.server.upload_root / (job_id + Path(request.filename).suffix.lower())
        try:
            self.connection.settimeout(30)
            with path.open("xb") as target:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 64 * 1024))
                    if not chunk:
                        raise ValueError("The audio upload was incomplete.")
                    target.write(chunk)
                    remaining -= len(chunk)
            self.server.jobs.start(job_id, path, request)
        except (OSError, ValueError, socket.timeout) as exc:
            path.unlink(missing_ok=True)
            self.server.jobs.fail_upload(job_id)
            self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self.json_response(HTTPStatus.ACCEPTED, {"id": job_id, "status": "queued"})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--web-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rtttl-generative-transformer/web/dist",
        help="Built generator web/dist directory (build with VITE_BASE_PATH=/).",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--build-web",
        action="store_true",
        help="Build the sibling browser app for this local server before starting (npm ci first).",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.build_web:
        web_source = args.web_root.resolve().parent
        npm = shutil.which("npm")
        if not npm or not (web_source / "package.json").is_file():
            parser.error("Install Node.js/npm and point --web-root to the browser app's web/dist.")
        environment = {**os.environ, "VITE_BASE_PATH": "/"}
        result = subprocess.run([npm, "run", "build"], cwd=web_source, env=environment, check=False)
        if result.returncode:
            parser.exit(
                result.returncode, "Browser build failed. Run npm ci in its web directory.\n"
            )
    if not (args.web_root / "index.html").is_file():
        parser.error(
            "Built studio not found. In rtttl-generative-transformer/web run "
            "npm ci, then VITE_BASE_PATH=/ npm run build; or pass --web-root."
        )
    with tempfile.TemporaryDirectory(prefix="rtttl-studio-") as uploads:
        server = StudioServer(("127.0.0.1", args.port), args.web_root, Path(uploads))
        print(f"RTTTL Studio: http://127.0.0.1:{args.port}/", flush=True)
        print("Audio stays on this machine. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
