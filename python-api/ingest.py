# python-api/ingest.py
import sys
import gc
import shutil
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from session import state


def _free_phi3():
    """
    Unconditionally unload Phi-3 from VRAM before heavy ingestion models load.
    Called at the start of every ingest regardless of modality.
    Phi-3 lazy-reloads automatically on the next query().
    """
    if state.get("pipeline") is None:
        return
    pipeline = state["pipeline"]
    if hasattr(pipeline, "generator") and pipeline.generator is not None:
        try:
            del pipeline.generator.llm
        except Exception:
            pass
        try:
            del pipeline.generator
        except Exception:
            pass
        pipeline.generator = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[Ingest] Phi-3 unloaded — VRAM freed for ingestion models.")


def run_ingestion(file_path: str, modality: str, filename: str) -> list:
    """
    Runs full ingestion for a single uploaded file.
    Always frees Phi-3 first, then loads ingestion models sequentially.
    """
    # ── ALWAYS free Phi-3 first ───────────────────────────────────────────
    _free_phi3()

    from src.rag_pipeline import RAGPipeline

    path      = Path(file_path)
    documents = []

    # ── TEXT ─────────────────────────────────────────────────────────────
    if modality == "text":
        from src.ingestion.text_loader import TextLoader
        tmp = _make_tmp(path, "txt_tmp")
        documents = TextLoader(str(tmp)).load()
        shutil.rmtree(tmp)

    # ── PDF ──────────────────────────────────────────────────────────────
    elif modality == "pdf":
        from src.ingestion.pdf_loader import PDFLoader
        tmp = _make_tmp(path, "pdf_tmp")
        documents = PDFLoader(str(tmp)).load()
        shutil.rmtree(tmp)

    # ── IMAGE ─────────────────────────────────────────────────────────────
    elif modality == "image":
        from src.ingestion.image_captioner import ImageCaptioner
        captioner = ImageCaptioner()
        documents = captioner.caption_file(str(path))
        captioner.unload()
        gc.collect()
        torch.cuda.empty_cache()
        print("[Ingest] Qwen2-VL freed.")

    # ── AUDIO ─────────────────────────────────────────────────────────────
    elif modality == "audio":
        from src.ingestion.audio_transcriber import AudioTranscriber
        tmp = _make_tmp(path, "audio_tmp")
        transcriber = AudioTranscriber(model_size="small", device="cuda")
        documents = transcriber.transcribe(str(tmp))
        del transcriber
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(tmp)
        print("[Ingest] Whisper freed.")

    # ── VIDEO ─────────────────────────────────────────────────────────────
    elif modality == "video":
        from src.ingestion.video_processor import VideoProcessor
        from src.ingestion.video_captioner import VideoCaptioner

        tmp = _make_tmp(path, "video_tmp")

        processor = VideoProcessor(keyframe_interval=2, device="cuda")
        transcript_docs, keyframe_images, keyframe_sources = processor.process(str(tmp))
        processor.unload()
        gc.collect()
        torch.cuda.empty_cache()
        print("[Ingest] Whisper freed.")

        captioner = VideoCaptioner()
        keyframe_docs = captioner.caption_frames(keyframe_images, keyframe_sources)
        captioner.unload()
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(tmp)
        print("[Ingest] Qwen2-VL freed.")

        documents = transcript_docs + keyframe_docs

    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not documents:
        raise ValueError("No documents extracted from file.")

    # ── Build index AFTER all heavy models freed ───────────────────────────
    print(f"[Ingest] Building RAG index for {len(documents)} documents...")
    pipeline = RAGPipeline()
    pipeline.ingest(documents, source_dir=str(path.parent))
    state["pipeline"] = pipeline
    print("[Ingest] Pipeline ready.")
    return documents


def _make_tmp(src_path: Path, dirname: str) -> Path:
    """Create a temp dir containing one file — loaders expect a directory."""
    tmp = src_path.parent / dirname
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, tmp / src_path.name)
    return tmp