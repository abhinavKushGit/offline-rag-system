import sys
import asyncio
import threading
import tempfile
from pathlib import Path

# ── Path setup MUST come before any local imports ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Third-party imports ──────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# ── Local imports (python-api/) ──────────────────────────────────────────────
from session import state, clear_pipeline, MODALITY_MAP
from ingest import run_ingestion
from query import make_sse_stream

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="OmniRAG API", version="6.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── STATUS ───────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    return {
        "pipeline_ready":   state["pipeline_ready"],
        "current_file":     state["current_file"],
        "current_modality": state["current_modality"],
        "processing":       state["processing"],
        "error":            state.get("error"),        # surface errors to frontend
    }


# ── SESSION CLEAR ─────────────────────────────────────────────────────────────
@app.delete("/session")
def delete_session():
    clear_pipeline()
    return {"message": "Session cleared."}


# ── background worker ─────────────────────────────────────────────────────────
def _ingest_worker(dest: Path, modality: str, filename: str):
    """Runs in a daemon thread — updates state directly, never touches HTTP."""
    try:
        docs = run_ingestion(str(dest), modality, filename)
        state["current_file"]     = filename
        state["current_modality"] = modality
        state["pipeline_ready"]   = True
        state["error"]            = None
        print(f"[Ingest] Done — {len(docs)} chunks indexed.")
    except Exception as exc:
        clear_pipeline()
        state["error"] = str(exc)
        print(f"[Ingest] ERROR: {exc}")
    finally:
        state["processing"] = False
        if dest.exists():
            dest.unlink()


# ── INGEST (returns 202 immediately) ─────────────────────────────────────────
@app.post("/ingest", status_code=202)
async def ingest_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    modality = MODALITY_MAP.get(ext)
    if modality is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: "
                   f"{', '.join(MODALITY_MAP.keys())}",
        )
    if state["processing"]:
        raise HTTPException(
            status_code=409,
            detail="Already processing a file. Wait or clear session.",
        )

    clear_pipeline()
    state["processing"] = True
    state["error"]      = None

    # Save the upload synchronously — this is fast
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # Kick off heavy processing in a background thread
    t = threading.Thread(target=_ingest_worker, args=(dest, modality, file.filename), daemon=True)
    t.start()

    # Return immediately — frontend must poll /status
    return {
        "message":  "Ingestion started.",
        "filename": file.filename,
        "modality": modality,
    }


# ── QUERY (SSE streaming) ─────────────────────────────────────────────────────
@app.get("/query")
async def query_sse(q: str):
    if not state["pipeline_ready"] or state["pipeline"] is None:
        raise HTTPException(
            status_code=400,
            detail="No file ingested. Upload a file first via POST /ingest",
        )
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        make_sse_stream(state["pipeline"], q),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)