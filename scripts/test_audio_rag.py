from src.ingestion.audio_transcriber import AudioTranscriber
from src.chunking.token_chunker import TokenChunker
from src.embeddings.text_embedder import TextEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.text_retriever import TextRetriever

AUDIO_DIR = "data/audio"

# Step 1 — Transcribe
transcriber = AudioTranscriber(model_size="small", device="cuda")
docs = transcriber.transcribe(AUDIO_DIR)

if not docs:
    print("No audio files found. Add .mp3 or .wav files to data/audio/")
    exit()

print(f"Total segments: {len(docs)}")

# Step 2 — Chunk
chunker = TokenChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=300,
    overlap=50
)
chunks = chunker.chunk(docs)
print(f"Total chunks: {len(chunks)}")

# Step 3 — Embed
embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [c.text for c in chunks]
vectors = embedder.embed(texts)

# Step 4 — Build FAISS store
metadatas = [
    {
        "text": c.text,
        "source": c.source,
        "modality": c.modality,
        "start_time": c.metadata.get("start_time", 0),
        "end_time": c.metadata.get("end_time", 0)
    }
    for c in chunks
]
store = FAISSStore(dim=384)
store.add(vectors, metadatas)

# Step 5 — Retrieve
retriever = TextRetriever(embedder, store, top_k=4)
query = "what does the speaker say about python"
results = retriever.retrieve(query)

print(f"\nResults for: '{query}'")
for r in results:
    ts = r.get("start_time", 0)
    print(f"\n[{ts:.1f}s] {r['text'][:200]}")