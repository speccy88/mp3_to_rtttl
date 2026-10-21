"""Exercise real HTTP boundaries and job cleanup without model downloads."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rtttl_studio import (
    MAX_UPLOAD_BYTES,
    ConversionRequest,
    JobStore,
    StudioServer,
    parse_conversion_request,
)


class RequestTests(unittest.TestCase):
    def test_audio_options_are_bounded_and_explicit(self):
        request = parse_conversion_request(
            "filename=hook.MP3&start=1.5&duration=8&bpm=123&source_mode=full&preset=faithful"
        )
        self.assertEqual(request, ConversionRequest("hook.MP3", 1.5, 8, 123, "full", "faithful"))
        self.assertIsNone(parse_conversion_request("filename=hook.wav&bpm=").bpm)
        invalid = [
            "start=nan",
            "duration=inf",
            "duration=31",
            "duration=0",
            "bpm=0",
            "start=-1",
            "preset=magic",
            "source_mode=vocals",
            "filename=notes.txt",
            "duration=4&duration=8",
            "unknown=yes",
        ]
        for query in invalid:
            with self.subTest(query=query), self.assertRaises(ValueError):
                parse_conversion_request(query)

    def test_failed_model_job_cleans_source_and_releases_capacity(self):
        def fail(path, request, progress):
            raise RuntimeError("No notes detected")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"source")
            jobs = JobStore(fail)
            job_id = jobs.reserve()
            self.assertIsNone(jobs.reserve())
            jobs.run(job_id, path, ConversionRequest("audio.wav"))
            self.assertFalse(path.exists())
            self.assertEqual(jobs.snapshot(job_id)["status"], "error")
            self.assertEqual(jobs.snapshot(job_id)["error"], "No notes detected")
            self.assertIsNotNone(jobs.reserve())


class StudioHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.web = self.root / "web"
        self.web.mkdir()
        (self.web / "index.html").write_text("<title>RTTTL Studio</title>", encoding="utf-8")
        (self.web / "model.wasm").write_bytes(b"\x00asm")
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.started = threading.Event()
        self.release = threading.Event()
        self.received = None

        def convert(path, request, progress):
            self.received = (path.read_bytes(), request)
            progress("Testing transcription")
            self.started.set()
            if not self.release.wait(3):
                raise RuntimeError("Test timed out")
            return {"candidates": [{"name": "Lead", "rtttl": "Hook:d=8,o=5,b=120:c,e,g"}]}

        self.jobs = JobStore(convert)
        self.server = StudioServer(("127.0.0.1", 0), self.web, self.uploads, self.jobs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        deadline = time.monotonic() + 2
        while self.jobs.active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        result = (response.status, dict(response.getheaders()), data)
        connection.close()
        return result

    def post(self, body=b"test audio", query="filename=hook.wav&duration=4", headers=None):
        return self.request(
            "POST",
            "/api/transcriptions?" + query,
            body,
            {"Content-Type": "application/octet-stream", **(headers or {})},
        )

    def test_async_conversion_round_trip_preserves_options_and_deletes_upload(self):
        status, _, body = self.post(query="filename=hook.wav&start=2&duration=4&bpm=120")
        self.assertEqual(status, 202)
        job_id = json.loads(body)["id"]
        self.assertTrue(self.started.wait(1))
        self.assertEqual(self.post()[0], 409)
        status, _, body = self.request("GET", f"/api/transcriptions/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "running")
        self.release.set()
        deadline = time.monotonic() + 2
        while self.jobs.active and time.monotonic() < deadline:
            time.sleep(0.01)
        status, _, body = self.request("GET", f"/api/transcriptions/{job_id}")
        result = json.loads(body)
        self.assertEqual(result["status"], "complete")
        self.assertIn("rtttl", result["result"]["candidates"][0])
        self.assertEqual(self.received, (b"test audio", ConversionRequest("hook.wav", 2, 4, 120)))
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_origin_host_size_and_options_are_rejected_before_job_creation(self):
        self.assertEqual(self.post(headers={"Origin": "https://unrelated.invalid"})[0], 403)
        self.assertEqual(self.post(headers={"Host": "unrelated.invalid"})[0], 403)
        self.assertEqual(
            self.post(body=b"", headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1)})[0], 413
        )
        self.assertEqual(self.post(query="filename=hook.wav&duration=nan")[0], 400)
        self.assertIsNone(self.jobs.active)
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_static_assets_health_and_unknown_job(self):
        status, _, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"RTTTL Studio", data)
        status, headers, _ = self.request("GET", "/model.wasm")
        self.assertEqual(headers["Content-Type"], "application/wasm")
        with patch("rtttl_studio.importlib.util.find_spec", return_value=None):
            status, _, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["ready"])
        self.assertIn("basic_pitch", json.loads(body)["missing"])
        self.assertEqual(self.request("GET", "/api/transcriptions/missing")[0], 404)
        self.assertEqual(self.request("GET", "/%2e%2e/private.txt")[0], 403)


if __name__ == "__main__":
    unittest.main()
