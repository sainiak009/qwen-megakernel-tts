"""
app.py — Minimal streaming TTS server.

Text in → chunked PCM audio out via Server-Sent Events.

Endpoints:
  POST /tts           body: {"text": "..."} → SSE stream of base64 PCM chunks
  POST /tts/wav       body: {"text": "..."} → complete WAV file (non-streaming)
  GET  /health

Run:
  uvicorn server.app:app --host 0.0.0.0 --port 8080

Or standalone:
  python server/app.py
"""

import asyncio
import base64
import json
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from server.qwen_tts_engine import QwenTTSEngine, SAMPLE_RATE

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Qwen3-TTS Megakernel Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single engine instance (loaded lazily on first request)
_engine: QwenTTSEngine | None = None
_engine_lock = asyncio.Lock()


def _get_engine() -> QwenTTSEngine:
    global _engine
    if _engine is None:
        _engine = QwenTTSEngine(
            model_id=os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            use_megakernel=os.environ.get("USE_MEGAKERNEL", "1") == "1",
            chunk_codes=int(os.environ.get("CHUNK_CODES", "12")),
            verbose=True,
        )
        _engine.load()
    return _engine


# ── Request models ────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    chunk_codes: int | None = None  # override default chunk size


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-load model on startup (avoids cold start on first request)."""
    preload = os.environ.get("PRELOAD_MODEL", "1") == "1"
    if preload:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_engine)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "backend": "megakernel" if os.environ.get("USE_MEGAKERNEL", "1") == "1" else "hf_baseline",
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/tts")
async def tts_stream(req: TTSRequest):
    """
    Streaming TTS via Server-Sent Events.

    Each SSE event carries a JSON payload:
      {"type": "chunk", "pcm_b64": "<base64 PCM-16>", "sample_rate": 24000}
      {"type": "done",  "elapsed_ms": 123, "ttfc_ms": 45}
      {"type": "error", "message": "..."}

    PCM format: signed 16-bit, mono, 24000 Hz.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    async def event_stream():
        engine = _get_engine()
        if req.chunk_codes:
            engine.chunk_codes = req.chunk_codes

        t_start = time.perf_counter()
        ttfc_sent = False
        try:
            async for pcm_chunk in engine.stream(req.text):
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                payload = {
                    "type":        "chunk",
                    "pcm_b64":     base64.b64encode(pcm_chunk).decode(),
                    "sample_rate": SAMPLE_RATE,
                    "elapsed_ms":  round(elapsed_ms, 1),
                }
                if not ttfc_sent:
                    payload["ttfc_ms"] = round(elapsed_ms, 1)
                    ttfc_sent = True
                yield f"data: {json.dumps(payload)}\n\n"

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            yield f"data: {json.dumps({'type': 'done', 'elapsed_ms': round(elapsed_ms, 1)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'trace': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tts/wav")
async def tts_wav(req: TTSRequest):
    """
    Non-streaming: return complete WAV file.
    Useful for testing; not suitable for real-time use.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    loop = asyncio.get_event_loop()
    engine = _get_engine()
    wav_bytes = await loop.run_in_executor(None, engine.generate_wav, req.text)
    return Response(content=wav_bytes, media_type="audio/wav")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
