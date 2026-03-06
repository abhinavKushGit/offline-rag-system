from src.ingestion.video_processor import VideoProcessor
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.image_retriever import ImageRetriever

VIDEO_DIR = "data/video"

processor = VideoProcessor(keyframe_interval=30, device="cuda")
transcript_docs, keyframe_docs = processor.process(VIDEO_DIR)

if not transcript_docs and not keyframe_docs:
    print("No video files found. Add .mp4 files to data/video/")
    exit()

print(f"Transcript segments: {len(transcript_docs)}")
print(f"Keyframes extracted: {len(keyframe_docs)}")

# --- TEXT SIDE (transcript) ---
text_embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [c.text for c in transcript_docs]
vectors = text_embedder.embed(texts)

metadatas = [
    {
        "text": c.text,
        "source": c.source,
        "modality": c.modality,
        "start_time": c.metadata.get("start_time", 0),
        "end_time": c.metadata.get("end_time", 0),
        "video_file": c.metadata.get("video_file", "")
    }
    for c in transcript_docs
]
text_store = FAISSStore(dim=384)
text_store.add(vectors, metadatas)
text_retriever = TextRetriever(text_embedder, text_store, top_k=3)

# --- IMAGE SIDE (keyframes) ---
image_embedder = ImageEmbedder(device="cuda")
image_retriever = ImageRetriever(image_embedder, index_dir="outputs/indexes/video_frames")
image_retriever.build_index(keyframe_docs)

# --- QUERY BOTH ---
query = "what is happening in the video"
print(f"\n--- Transcript results for: '{query}' ---")
text_results = text_retriever.retrieve(query)
for r in text_results:
    ts = r.get("start_time", 0)
    print(f"[{ts:.1f}s] {r['text'][:200]}")

print(f"\n--- Keyframe results for: '{query}' ---")
frame_results = image_retriever.retrieve(query, top_k=3)
for r in frame_results:
    print(f"[{r.timestamp:.1f}s] {r.text}  (score: {r.metadata['clip_score']:.3f})")