# src/ingestion/audio_transcriber.py
import os
import whisper
from pathlib import Path
from src.schema import Document


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}


class AudioTranscriber:
    """
    Transcribes audio files in a directory using OpenAI Whisper.
    Returns a list of Document objects (modality='audio') with
    per-segment start_time metadata — mirrors VideoProcessor's transcript output.
    """

    def __init__(self, model_size: str = "small", device: str = "cuda"):
        print(f"[AudioTranscriber] Loading Whisper '{model_size}' on {device}...")
        self.device = device
        self.model = whisper.load_model(model_size, device=device)
        print("[AudioTranscriber] Whisper loaded.")

    def transcribe(self, directory: str) -> list:
        """
        Transcribes all supported audio files found in `directory`.

        Args:
            directory: Path to a folder containing one or more audio files.

        Returns:
            List of Document objects, one per Whisper segment.
        """
        dir_path = Path(directory)
        audio_files = sorted([
            f for f in dir_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not audio_files:
            raise ValueError(f"[AudioTranscriber] No supported audio files found in: {directory}")

        documents = []

        for audio_path in audio_files:
            print(f"[AudioTranscriber] Transcribing: {audio_path.name}")
            result = self.model.transcribe(str(audio_path), verbose=False)

            segments = result.get("segments", [])

            if not segments:
                # Fallback: treat the full transcript as one document
                full_text = result.get("text", "").strip()
                if full_text:
                    documents.append(Document(
                        text=full_text,
                        source=str(audio_path),
                        modality="audio",
                        metadata={"start_time": 0.0},
                    ))
                continue

            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                documents.append(Document(
                    text=text,
                    source=str(audio_path),
                    modality="audio",
                    metadata={"start_time": seg.get("start", 0.0)},
                ))

            print(f"[AudioTranscriber] {len(segments)} segments from '{audio_path.name}'.")

        print(f"[AudioTranscriber] Total documents produced: {len(documents)}")
        return documents

    def transcribe_file(self, file_path: str) -> list:
        """
        Transcribes a single audio/video file directly (no directory wrapping needed).
        Alias used by callers that pass a file path instead of a directory.
        """
        audio_path = Path(file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"[AudioTranscriber] File not found: {file_path}")

        print(f"[AudioTranscriber] Transcribing file: {audio_path.name}")
        result = self.model.transcribe(str(audio_path), verbose=False)
        segments = result.get("segments", [])

        documents = []
        if not segments:
            full_text = result.get("text", "").strip()
            if full_text:
                documents.append(Document(
                    text=full_text,
                    source=str(audio_path),
                    modality="audio",
                    metadata={"start_time": 0.0},
                ))
        else:
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                documents.append(Document(
                    text=text,
                    source=str(audio_path),
                    modality="audio",
                    metadata={"start_time": seg.get("start", 0.0)},
                ))
            print(f"[AudioTranscriber] {len(segments)} segments from '{audio_path.name}'.")

        print(f"[AudioTranscriber] Total documents produced: {len(documents)}")
        return documents

    def unload(self):
        """Explicitly free Whisper from memory (optional, ingest.py uses del instead)."""
        del self.model
        self.model = None