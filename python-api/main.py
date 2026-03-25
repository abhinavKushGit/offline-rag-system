import sys
import asyncio
import tempfile
from pathlib import Path

# ── Path setup MUST come before any local imports ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))                    # gives access to src/
sys.path.insert(0, str(Path(__file__).resolve().parent)) # gives access to python-api/

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
app = FastAPI(title="OmniRAG API", version="6.0")

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
    }


# ── SESSION CLEAR ─────────────────────────────────────────────────────────────
@app.delete("/session")
def delete_session():
    clear_pipeline()
    return {"message": "Session cleared."}


# ── INGEST ────────────────────────────────────────────────────────────────────
@app.post("/ingest")
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

    # Clear previous session before starting new one
    clear_pipeline()
    state["processing"] = True

    dest = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        dest.write_bytes(content)

        # Run blocking ingestion in thread pool so FastAPI stays responsive
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, run_ingestion, str(dest), modality, file.filename
        )

        state["current_file"]     = file.filename
        state["current_modality"] = modality
        state["pipeline_ready"]   = True

        return {
            "message":   f"Ingested {len(docs)} document chunks.",
            "filename":  file.filename,
            "modality":  modality,
            "doc_count": len(docs),
        }

    except Exception as exc:
        clear_pipeline()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        state["processing"] = False
        if dest.exists():
            dest.unlink()  # clean up uploaded temp file


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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)