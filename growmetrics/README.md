# Growmetrics AI Chemistry Video Request Service

A FastAPI backend that generates short educational chemistry videos using:
- **Manim** (Community Edition) — vector animation engine
- **Edge-TTS** — Microsoft Neural TTS for zero-cost narration audio
- **FFmpeg** — audio-video merging and duration inspection
- **LLM Agent** — Gemini 1.5 Flash / GPT-4o-mini for scene specification generation

---

## Setup

### 1. System Dependencies

Install FFmpeg (required for audio-video merging and FFprobe inspection):
```powershell
winget install ffmpeg
```

Manim also requires **Cairo** on Windows. Install via the Manim Community docs:
```
https://docs.manim.community/en/stable/installation/windows.html
```

### 2. Python Environment

```powershell
cd growmetrics
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the `growmetrics/` directory:
```ini
# Provide at least one of these:
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional overrides (defaults shown):
LLM_MODEL=gemini-1.5-flash
MAX_RETRIES=3
MIN_SCRIPT_WORDS=70
MIN_AUDIO_SECONDS=25.0
MAX_AUDIO_SECONDS=90.0
TTS_VOICE=en-US-JennyNeural
MANIM_QUALITY=l
```

---

## Running the Service

```powershell
uvicorn app.main:app --reload --port 8000
```

Interactive API documentation: **http://localhost:8000/docs**

---

## API Reference

### POST /api/v1/videos/request
Submit a chemistry concept query.

```powershell
curl -X POST "http://localhost:8000/api/v1/videos/request" `
  -H "Content-Type: application/json" `
  -d '{"query": "How does the pH scale work?"}'
```
Response:
```json
{"job_id": "job_a3f2c1d8e0", "status": "PENDING", "message": "Video generation job queued successfully."}
```

### GET /api/v1/videos
List all jobs:
```powershell
curl "http://localhost:8000/api/v1/videos"
```

### GET /api/v1/videos/{job_id}
Poll job status and stage history:
```powershell
curl "http://localhost:8000/api/v1/videos/job_a3f2c1d8e0"
```

### GET /api/v1/videos/{job_id}/artifact
Download the generated MP4 (only when status = COMPLETED):
```powershell
curl -O "http://localhost:8000/api/v1/videos/job_a3f2c1d8e0/artifact"
```

---

## Required Chemistry Queries

The three required end-to-end concepts:

| # | Query |
|---|-------|
| 1 | How does the pH scale work? |
| 2 | Why do atoms form covalent bonds? |
| 3 | What is the difference between ionic and covalent bonding? |

---

## Architecture Notes

### Job Lifecycle

```
PENDING → ANALYZING → GENERATING_AUDIO → GENERATING_CODE → RENDERING → VALIDATING → COMPLETED
                                                    ↑              |
                                                    └── (retry) ───┘
                                                              ↓ (if max retries hit)
                                                        Parametric Fallback
                                                              ↓
                                                       GENERATING_CODE → RENDERING → VALIDATING → COMPLETED
```

### 4-Stage Quality Gate Pipeline

| Gate | What Is Validated | Failure Code |
|------|------------------|--------------|
| **Gate 1** | LLM JSON output parses & contains required fields | `ERR_JSON_PARSE_FAILED`, `ERR_JSON_SCHEMA_INVALID` |
| **Gate 2** | Script ≥ 70 words; Edge-TTS audio 25–90 seconds | `ERR_SCRIPT_TOO_SHORT`, `ERR_AUDIO_TOO_SHORT`, `ERR_AUDIO_TOO_LONG` |
| **Gate 3** | AST safety linting + Manim subprocess dry-run | `ERR_CODE_SYNTAX`, `ERR_CODE_UNSAFE`, `ERR_RENDER_FAILED` |
| **Gate 4** | FFprobe file integrity & audio-video sync (≤0.5s) | `ERR_ARTIFACT_MISSING`, `ERR_AV_SYNC` |

### Persistence & Artifact Boundary
- **Job state:** in-memory (`JobRepository`) — clean interface, swap for Redis/PostgreSQL by replacing the class.
- **Video artifacts:** filesystem under `growmetrics/artifacts/{job_id}/video.mp4`.

### AI/Video-Generation Boundary
- `app/generator/agent.py` owns all LLM interaction.
- `app/generator/manim_engine.py` owns all Manim subprocess calls.
- `app/generator/tts_engine.py` owns all Edge-TTS calls.
- Swap any provider by implementing the same function signatures.

### Cost Notes
- **LLM (Gemini 1.5 Flash):** ~$0.0003 per generation call. 3 retries = max ~$0.0009.
- **Edge-TTS:** $0 (Microsoft Neural TTS, no API key required).
- **Manim rendering:** $0 (local CPU/GPU).
- **Total per video:** approximately **$0.001–$0.005**.

---

## Testing

```powershell
# Unit tests (gates & API, no live LLM/render required)
pytest tests/test_quality_gates.py tests/test_api.py -v

# 5x repeated-run consistency benchmark (requires live server on :8000)
pytest tests/test_consistency.py -v -s --timeout=600
```
