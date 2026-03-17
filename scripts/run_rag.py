import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.text_loader import TextLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.image_loader import ImageLoader
from src.ingestion.audio_transcriber import AudioTranscriber
from src.ingestion.video_processor import VideoProcessor


def main():
    print("=== MULTIMODAL RAG SYSTEM ===")
    print("Choose source:")
    print("  1. text")
    print("  2. pdf")
    print("  3. image")
    print("  4. audio")
    print("  5. video")

    source = input("Source: ").strip().lower()
    if source not in {"text", "pdf", "image", "audio", "video"}:
        print("❌ Invalid source.")
        return

    path = input("Enter directory path: ").strip()
    if not Path(path).exists():
        print("❌ Path does not exist.")
        return

    documents = []

    if source == "text":
        documents = TextLoader(path).load()

    elif source == "pdf":
        documents = PDFLoader(path).load()

    elif source == "image":
        from src.ingestion.image_captioner import ImageCaptioner
        captioner = ImageCaptioner()
        documents = captioner.caption_dir(path)
        captioner.unload()                  # VRAM freed before RAGPipeline()
        print(f"[INFO] LLaVA unloaded, VRAM freed.")

    elif source == "audio":
        transcriber = AudioTranscriber(model_size="small", device="cuda")
        documents = transcriber.transcribe(path)
        # Free VRAM before Phi-3 loads
        del transcriber
        torch.cuda.empty_cache()
        print("[INFO] Whisper unloaded, VRAM freed.")

    elif source == "video":
        processor = VideoProcessor(keyframe_interval=30, device="cpu")
        transcript_docs, keyframe_docs = processor.process(path)
        documents = transcript_docs + keyframe_docs
        del processor
        torch.cuda.empty_cache()
        print("[INFO] VideoProcessor unloaded, VRAM freed.")

    if not documents:
        print("❌ No documents found.")
        return

    print(f"[INFO] Loaded {len(documents)} documents")

    rag = RAGPipeline()  # moved here — after transcription, VRAM is free
    rag.ingest(documents, source_dir=path)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("Exiting RAG session.")
            break
        answer = rag.query(query)
        print("\n=== ANSWER ===")
        print(answer)


if __name__ == "__main__":
    main()