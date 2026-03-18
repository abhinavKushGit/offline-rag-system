import cv2
import subprocess
import torch
import gc
import shutil
from pathlib import Path
from PIL import Image
from src.schema import Document
from src.ingestion.audio_transcriber import AudioTranscriber
from src.retrieval.temporal_attention import TemporalAttention

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


class VideoProcessor:
    def __init__(self, keyframe_interval: int = 2, device: str = "cuda"):
        self.keyframe_interval = keyframe_interval
        self.device = device
        self.transcriber = AudioTranscriber(model_size="small", device=self.device)
        self.temporal_attn = TemporalAttention(embed_dim=512, num_heads=8)
        self.temporal_attn.eval()
        # dedicated temp dir — never overlaps with data/audio/
        self._audio_tmp = Path("outputs/video_audio_tmp")
        self._audio_tmp.mkdir(parents=True, exist_ok=True)

    def process(self, video_dir: str) -> tuple[list[Document], list[Image.Image], list[str]]:
        transcript_docs = []
        keyframe_images = []
        keyframe_sources = []

        for file in sorted(Path(video_dir).iterdir()):
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            print(f"[VideoProcessor] Processing {file.name}")

            # extract audio to isolated tmp dir
            audio_out = self._audio_tmp / f"{file.stem}_extracted.wav"
            self._extract_audio(file, audio_out)
            segments = self.transcriber.transcribe_file(str(audio_out))
            for doc in segments:
                doc.metadata["video_file"] = file.name
                doc.modality = "video"
            transcript_docs.extend(segments)

            frames = self._extract_keyframes(file)
            for timestamp, frame_img in frames:
                keyframe_images.append(frame_img)
                keyframe_sources.append(f"{file.name}::frame_{timestamp:.1f}s")

        print(f"[VideoProcessor] Transcripts: {len(transcript_docs)}, "
              f"Keyframes: {len(keyframe_images)}")
        return transcript_docs, keyframe_images, keyframe_sources

    def _extract_audio(self, video_path: Path, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ac", "1", "-ar", "16000", str(out_path)],
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

    def apply_temporal_attention(self, vectors):
        print(f"[VideoProcessor] Applying temporal attention over "
              f"{vectors.shape[0]} frames...")
        attended = self.temporal_attn.attend(vectors)
        print(f"[VideoProcessor] Temporal attention applied.")
        return attended

    def unload(self):
        print("[VideoProcessor] Unloading Whisper...")
        del self.transcriber
        self.transcriber = None
        # clean up temp audio files after transcription done
        if self._audio_tmp.exists():
            shutil.rmtree(self._audio_tmp)
            print(f"[VideoProcessor] Cleaned up temp audio files.")
        gc.collect()
        torch.cuda.empty_cache()
        print("[VideoProcessor] Whisper unloaded.")