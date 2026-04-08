# python-api/session.py
import gc
import torch

state = {
    "pipeline":         None,
    "current_file":     None,
    "current_modality": None,
    "pipeline_ready":   False,
    "processing":       False,
    "error":            None,
}

MODALITY_MAP = {
    ".txt":  "text",
    ".pdf":  "pdf",
    ".png":  "image", ".jpg": "image", ".jpeg": "image",
    ".webp": "image", ".bmp": "image",
    ".mp3":  "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".mp4":  "video", ".mkv": "video", ".avi": "video", ".mov": "video",
}


def _unload_model(obj, attr: str):
    """Delete a model attribute and clear its CUDA memory safely."""
    m = getattr(obj, attr, None)
    if m is None:
        return
    try:
        # Move to CPU first — forces CUDA allocator to release the pages
        if hasattr(m, "to"):
            m.to("cpu")
    except Exception:
        pass
    try:
        delattr(obj, attr)
    except Exception:
        pass
    del m


def clear_pipeline():
    """
    Fully unload every model held by the current RAGPipeline before
    discarding it, so VRAM is clean for the next ingestion.
    """
    pipeline = state.get("pipeline")

    if pipeline is not None:
        # ── Generator (Phi-3 / llama-cpp) ────────────────────────────────
        gen = getattr(pipeline, "generator", None)
        if gen is not None:
            # llama-cpp model
            llm = getattr(gen, "llm", None)
            if llm is not None:
                try:
                    del llm
                except Exception:
                    pass
                gen.llm = None
            # HF pipeline wrapper
            pipe = getattr(gen, "pipe", None)
            if pipe is not None:
                _unload_model(pipe, "model")
                try:
                    del pipe
                except Exception:
                    pass
                gen.pipe = None
            try:
                del gen
            except Exception:
                pass
            pipeline.generator = None

        # ── Text embedder (SentenceTransformer) ──────────────────────────
        te = getattr(pipeline, "text_embedder", None)
        if te is not None:
            _unload_model(te, "model")        # SentenceTransformer stores .model
            _unload_model(te, "_model")       # some wrappers use _model
            try:
                del te
            except Exception:
                pass
            pipeline.text_embedder = None

        # ── Image embedder (CLIP via open_clip) ──────────────────────────
        ie = getattr(pipeline, "image_embedder", None)
        if ie is not None:
            _unload_model(ie, "model")
            try:
                del ie
            except Exception:
                pass
            pipeline.image_embedder = None

        # ── Vector stores (FAISS — CPU, but release the reference) ───────
        pipeline.text_vectorstore = None
        pipeline.image_retriever  = None

        try:
            del pipeline
        except Exception:
            pass

    state["pipeline"]         = None
    state["pipeline_ready"]   = False
    state["current_file"]     = None
    state["current_modality"] = None
    state["error"]            = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[Session] VRAM freed. "
              f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB  "
              f"Reserved:  {torch.cuda.memory_reserved()  / 1024**2:.1f} MB")