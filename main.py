"""
MeetingMind — AI Meeting-to-Action Agent
Backend: FastAPI + AssemblyAI + Grok (xAI)
"""

import os
import json
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
GROK_API_KEY       = os.getenv("GROK_API_KEY", "")
HOST               = os.getenv("HOST", "0.0.0.0")
PORT               = int(os.getenv("PORT", 8000))
ALLOWED_ORIGINS    = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meetingmind")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MeetingMind API",
    description="AI agent that turns meeting audio/transcripts into structured action plans",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend — static/ lives next to backend/, not inside it
STATIC_DIR = Path(__file__).parent.parent / "static"
if not STATIC_DIR.exists():
    # Fallback: try a static/ folder inside backend/ (local dev override)
    STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic Models ────────────────────────────────────────────────────────────
class TextAnalyseRequest(BaseModel):
    transcript: str
    context: Optional[str] = ""


class AnalysisResponse(BaseModel):
    transcript: str
    analysis: dict


# ── AssemblyAI helpers ─────────────────────────────────────────────────────────
ASSEMBLY_BASE = "https://api.assemblyai.com/v2"

async def upload_audio_to_assemblyai(file_bytes: bytes, api_key: str) -> str:
    """Upload raw audio bytes to AssemblyAI and return the hosted URL."""
    headers = {"authorization": api_key}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{ASSEMBLY_BASE}/upload",
            headers=headers,
            content=file_bytes,
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"AssemblyAI upload failed: {resp.text[:200]}")
    return resp.json()["upload_url"]


async def request_transcription(audio_url: str, api_key: str) -> str:
    """Submit a transcription job and return the job ID."""
    headers = {"authorization": api_key, "content-type": "application/json"}
    payload = {"audio_url": audio_url, "speaker_labels": True}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{ASSEMBLY_BASE}/transcript", headers=headers, json=payload)
    if resp.status_code != 200:
        raise HTTPException(502, f"AssemblyAI transcription request failed: {resp.text[:200]}")
    return resp.json()["id"]


async def poll_transcription(job_id: str, api_key: str) -> str:
    """Poll AssemblyAI until transcript is ready. Returns plain text."""
    headers = {"authorization": api_key}
    url = f"{ASSEMBLY_BASE}/transcript/{job_id}"
    for attempt in range(120):          # max ~6 minutes
        await asyncio.sleep(3)
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=headers)
        data = resp.json()
        status = data.get("status")
        log.info(f"Transcription poll #{attempt+1}: {status}")
        if status == "completed":
            # Use speaker-labelled utterances if available
            utterances = data.get("utterances") or []
            if utterances:
                return "\n".join(f"Speaker {u['speaker']}: {u['text']}" for u in utterances)
            return data.get("text", "")
        if status == "error":
            raise HTTPException(502, f"AssemblyAI error: {data.get('error')}")
    raise HTTPException(504, "Transcription timed out after 6 minutes")


async def transcribe_audio(file_bytes: bytes, api_key: str) -> str:
    """Full AssemblyAI transcription pipeline."""
    log.info("Uploading audio to AssemblyAI…")
    audio_url = await upload_audio_to_assemblyai(file_bytes, api_key)
    log.info("Requesting transcription…")
    job_id = await request_transcription(audio_url, api_key)
    log.info(f"Polling job {job_id}…")
    return await poll_transcription(job_id, api_key)


# ── Grok / xAI helpers ────────────────────────────────────────────────────────
GROK_URL = "https://api.groq.com/openai/v1/chat/completions"
GROK_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert meeting analyst AI agent. Extract structured information from meeting transcripts with precision.

Respond ONLY with valid JSON — no markdown fences, no explanation, just raw JSON.

Return this exact structure:
{
  "decisions": [
    { "text": "decision description", "context": "brief why/background" }
  ],
  "action_items": [
    {
      "task": "what needs to be done",
      "owner": "person name or UNKNOWN",
      "deadline": "deadline string or null",
      "priority": "high|medium|low",
      "flagged": true or false,
      "flag_reason": "reason if flagged, else null"
    }
  ],
  "unresolved": [
    { "issue": "unresolved topic", "context": "why it's unresolved" }
  ],
  "escalations": [
    {
      "issue": "what needs human attention",
      "reason": "why the agent cannot resolve this",
      "suggested_action": "what a human should do next"
    }
  ],
  "summary": "2-3 sentence summary of what the meeting was about and what was discussed, even if nothing was decided",
  "discussion_topics": [
    { "topic": "topic that was discussed", "outcome": "what came out of it — decision, no conclusion, deferred, informational, etc." }
  ],
  "meeting_health": "on-track|at-risk|critical|informational",
  "health_reason": "one sentence reason"
}

CRITICAL RULES FOR EMPTY RESULTS:
- If NO decisions were made, return decisions as [] but still fill summary and discussion_topics with what was actually talked about.
- If NO action items exist, return action_items as [] but explain in summary what the meeting was for.
- If the meeting was purely informational (status update, briefing, catch-up), set meeting_health to "informational".
- NEVER return an empty summary. Even if nothing was decided, describe what was discussed, who spoke, and what the purpose of the meeting appeared to be.
- discussion_topics must ALWAYS be filled — list every topic that came up, even casually mentioned ones.

ESCALATION RULES — always escalate when:
- Action item has no clear owner (owner = UNKNOWN)
- Assigned owner is unavailable (on leave, quit, etc.)
- Deadline is missing or ambiguous on a high-priority task
- Two people are assigned the same task (conflict)
- Decision was made without key stakeholders present

PRIORITY RULES:
- high: production bugs, client-facing deliverables, compliance/board deadlines
- medium: internal tasks with clear deadlines
- low: informational, no deadline, nice-to-have

Extract EVERY action item — including implicit ones (e.g. "someone should look into X").
Be thorough. Real meetings are messy; handle them."""

async def analyse_with_grok(transcript: str, context: str, api_key: str) -> dict:
    """Send transcript to Grok and parse the JSON response."""
    user_content = f"{'Meeting context: ' + context + chr(10)*2 if context else ''}Transcript:\n{transcript}"
    payload = {
        "model": GROK_MODEL,
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    log.info("Sending transcript to Grok for analysis…")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GROK_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(502, f"Grok API error {resp.status_code}: {resp.text[:300]}")

    raw = resp.json()["choices"][0]["message"]["content"]
    # Strip any accidental markdown fences
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed. Raw response:\n{clean[:500]}")
        raise HTTPException(502, f"Grok returned invalid JSON: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    # Also check backend/static as fallback
    fallback = Path(__file__).parent / "static" / "index.html"
    if fallback.exists():
        return FileResponse(str(fallback))
    return {"message": "MeetingMind API is running. See /docs for API reference."}


@app.get("/health")
async def health():
    """Health check — used by Render/Railway to confirm the server is up."""
    return {
        "status": "ok",
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "grok_configured": bool(GROK_API_KEY),
    }


@app.post("/api/analyse/audio", response_model=AnalysisResponse)
async def analyse_audio(
    file: UploadFile = File(..., description="Audio or video file (MP3, MP4, WAV, M4A, WEBM)"),
    context: str = Form("", description="Optional meeting context"),
):
    """
    Upload an audio/video file.
    The backend transcribes it via AssemblyAI, then analyses with Grok.
    Returns structured decisions, action items, escalations.
    """
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY not set in server environment")
    if not GROK_API_KEY:
        raise HTTPException(500, "GROK_API_KEY not set in server environment")

    # Validate file type
    allowed_types = {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
        "audio/mp4", "video/mp4", "audio/m4a", "audio/x-m4a",
        "audio/webm", "video/webm", "audio/ogg",
    }
    content_type = file.content_type or ""
    if content_type and content_type not in allowed_types:
        log.warning(f"Received content-type: {content_type} — proceeding anyway")

    log.info(f"Received file: {file.filename} ({content_type})")
    file_bytes = await file.read()

    if len(file_bytes) > 200 * 1024 * 1024:   # 200 MB
        raise HTTPException(413, "File too large. Maximum size is 200 MB.")

    transcript = await transcribe_audio(file_bytes, ASSEMBLYAI_API_KEY)
    log.info(f"Transcript length: {len(transcript)} chars")

    analysis = await analyse_with_grok(transcript, context, GROK_API_KEY)
    log.info("Analysis complete ✓")

    return AnalysisResponse(transcript=transcript, analysis=analysis)


@app.post("/api/analyse/text", response_model=AnalysisResponse)
async def analyse_text(body: TextAnalyseRequest):
    """
    Accept a plain-text transcript (already transcribed).
    Skips AssemblyAI — goes straight to Grok analysis.
    """
    if not GROK_API_KEY:
        raise HTTPException(500, "GROK_API_KEY not set in server environment")
    if not body.transcript.strip():
        raise HTTPException(400, "Transcript cannot be empty")

    log.info(f"Text analyse request — {len(body.transcript)} chars")
    analysis = await analyse_with_grok(body.transcript, body.context or "", GROK_API_KEY)
    log.info("Analysis complete ✓")

    return AnalysisResponse(transcript=body.transcript, analysis=analysis)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    display_host = "localhost" if HOST in ("0.0.0.0", "::") else HOST
    log.info(f"Starting MeetingMind on http://{HOST}:{PORT}")
    log.info(f"Open in browser  →  http://{display_host}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)