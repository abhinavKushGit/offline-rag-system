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
        file = Path(file_path)
        print(f"[AudioTranscriber] Transcribing {file.name}...")
        result = self.model.transcribe(str(file), language=self.language, verbose=False)

        segments = result["segments"]
        docs = []
        GROUP_SIZE = 8   # merge 8 Whisper segments into one Document

        for i in range(0, len(segments), GROUP_SIZE):
            group = segments[i : i + GROUP_SIZE]
            combined_text = " ".join(s["text"].strip() for s in group)
            start_time    = group[0]["start"]
            end_time      = group[-1]["end"]

            docs.append(Document(
                text=combined_text,
                source=str(file),
                modality="audio",
                timestamp=start_time,
                metadata={
                    "start_time": start_time,
                    "end_time":   end_time,
                    "audio_file": file.name,
                }
            ))

        print(f"[AudioTranscriber] {file.name} → {len(segments)} segments → {len(docs)} grouped docs")
        return docs