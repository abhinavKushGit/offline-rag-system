import cv2
import subprocess
import torch
from pathlib import Path
from PIL import Image
from src.schema import Document
from src.ingestion.audio_transcriber import AudioTranscriber
from src.embeddings.image_embedder import ImageEmbedder

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}

class VideoProcessor:
    def __init__(self, keyframe_interval: int = 30, device: str = "cuda"):
        self.keyframe_interval = keyframe_interval
        self.device = device  # respect what was passed in
        self.transcriber = AudioTranscriber(model_size="small", device=self.device)
        self.image_embedder = ImageEmbedder(device=self.device)

    def process(self, video_dir: str) -> tuple[list[Document], list[Document]]:
        transcript_docs = []
        keyframe_docs = []

        for file in sorted(Path(video_dir).iterdir()):
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            print(f"[VideoProcessor] Processing {file.name}")

            # Audio → transcript
            audio_out = Path("data/audio") / f"{file.stem}_extracted.wav"
            self._extract_audio(file, audio_out)
            segments = self.transcriber.transcribe_file(str(audio_out))
            for doc in segments:
                doc.metadata["video_file"] = file.name
                doc.modality = "video"
            transcript_docs.extend(segments)

            # Keyframes
            frames = self._extract_keyframes(file)
            for timestamp, frame_img in frames:
                doc = Document(
                    text=f"Video frame from {file.name} at {timestamp:.1f}s",
                    source=str(file),
                    modality="video",
                    timestamp=timestamp,
                    metadata={"video_file": file.name, "frame_time": timestamp,
                              "_pil_image": frame_img}
                )
                keyframe_docs.append(doc)

        print(f"[VideoProcessor] Transcripts: {len(transcript_docs)}, Keyframes: {len(keyframe_docs)}")
        return transcript_docs, keyframe_docs

    def _extract_audio(self, video_path: Path, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(out_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _extract_keyframes(self, video_path: Path) -> list[tuple[float, Image.Image]]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        interval_frames = int(fps * self.keyframe_interval)
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / fps
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append((timestamp, img))
            frame_idx += 1
        cap.release()
        return frames