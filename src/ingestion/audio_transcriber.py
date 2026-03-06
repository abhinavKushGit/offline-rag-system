import whisper
import torch
from pathlib import Path
from src.schema import Document

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

class AudioTranscriber:
    def __init__(self, model_size: str = "small", device: str = "cuda", language: str = "en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AudioTranscriber] Loading Whisper '{model_size}' on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)
        self.language = language

    def transcribe(self, audio_dir: str) -> list[Document]:
        """Transcribe all audio files in a directory"""
        docs = []
        for file in sorted(Path(audio_dir).iterdir()):
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            docs.extend(self.transcribe_file(str(file)))
        return docs

    def transcribe_file(self, file_path: str) -> list[Document]:
        """Transcribe a single audio file"""
        file = Path(file_path)
        docs = []
        print(f"[AudioTranscriber] Transcribing {file.name}...")
        result = self.model.transcribe(str(file), language=self.language, verbose=False)
        for segment in result["segments"]:
            doc = Document(
                text=segment["text"].strip(),
                source=str(file),
                modality="audio",
                timestamp=segment["start"],
                metadata={
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "audio_file": file.name
                }
            )
            docs.append(doc)
        print(f"[AudioTranscriber] {file.name} → {len(result['segments'])} segments")
        return docs