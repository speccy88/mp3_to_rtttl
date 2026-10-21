# Local Studio API v1

`python rtttl_studio.py` serves the built browser app and API from
`http://127.0.0.1:8765/`. This is a local application service, not a production
multiuser host. Requests with an unrelated Host or Origin are rejected. There is
no permissive CORS policy. The sibling Pages build path and root-path builds work.

## Health

`GET /api/health`

```json
{
  "service": "mp3-to-rtttl",
  "version": 1,
  "ready": true,
  "missing": [],
  "max_upload_bytes": 33554432,
  "max_excerpt_seconds": 30,
  "presets": ["faithful", "balanced", "smooth"]
}
```

`ready` reports dependency availability. Model initialization happens on the first
conversion and a runtime/model initialization failure is returned as a job error.

## Submit a conversion

`POST /api/transcriptions?filename=hook.mp3&start=12&duration=8&bpm=120&source_mode=auto&preset=faithful`

The request body is the raw audio file, **not multipart form data**. Send
`Content-Type: application/octet-stream` and `Content-Length`. No client-provided
path is used to save uploads.

| Parameter | Contract |
| --- | --- |
| `filename` | Supported extension, up to 256 characters; default `audio.wav` |
| `start` | Finite seconds, 0–86,400; default 0 |
| `duration` | Finite seconds, 0.25–30; default 12 |
| `bpm` | Optional finite 25–900; blank/absent estimates tempo |
| `source_mode` | `auto`, `full`, `harmonic`; default `auto` |
| `preset` | `faithful`, `balanced`, `smooth`; default `faithful` |

Successful response: HTTP 202, `{"id":"opaque-id","status":"queued"}`.
Another active conversion returns HTTP 409; invalid options return 400; empty or
oversized uploads return 413. The maximum upload is 32 MiB.

## Poll

`GET /api/transcriptions/{id}`

The job has `id`, `status` (`queued`, `running`, `complete`, `error`) and a human
readable `progress` string. On completion it includes:

```json
{
  "result": {
    "candidates": [
      {
        "name": "Detailed lead",
        "rtttl": "Hook:d=4,o=5,b=120:8c5,8e5,4g5",
        "score": 0.82,
        "settings": {"decoder_mode": "events", "grid": 16},
        "diagnostics": {"source_offset_seconds": 0, "duration_seconds": 1}
      }
    ],
    "analysis": {"estimated_bpm": 120, "source_mode": "full", "start": 12, "duration": 1},
    "warnings": ["Source evidence scores are not recognition accuracy."]
  }
}
```

The example score is illustrative. Actual `settings` contains the full rendering
configuration. Up to three distinct decoder outputs are returned, ordered by
source evidence. Duplicate interpretations collapse to one entry. The selected
preset applies to each decoder. Errors have an `error` string instead of a result.

The server retains at most eight recent jobs for 15 minutes. Unknown/expired jobs
return 404. Closing the browser stops polling; it does not interrupt active model
inference. The source upload is removed after the job finishes or fails.
