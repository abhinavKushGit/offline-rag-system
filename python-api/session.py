# python-api/session.py
import gc
import torch

state = {
    "pipeline": None,
    "current_file": None,
    "current_modality": None,
    "pipeline_ready": False,
    "processing": False,
}

MODALITY_MAP = {
    ".txt":  "text",
    ".pdf":  "pdf",
    ".png":  "image", ".jpg": "image", ".jpeg": "image",
    ".webp": "image", ".bmp": "image",
    ".mp3":  "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".mp4":  "video", ".mkv": "video", ".avi": "video", ".mov": "video",
}

def clear_pipeline():
    if state["pipeline"] is not None:
        del state["pipeline"]
        state["pipeline"] = None
    gc.collect()
    torch.cuda.empty_cache()
    state["pipeline_ready"] = False
    state["current_file"] = None
    state["current_modality"] = None