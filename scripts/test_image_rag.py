from src.ingestion.image_loader import ImageLoader
from src.embeddings.image_embedder import ImageEmbedder
from src.retrieval.image_retriever import ImageRetriever

IMAGE_DIR = "data/images"

loader = ImageLoader(IMAGE_DIR)
docs = loader.load()

if not docs:
    print("No images found. Add .jpg or .png files to data/images/")
    exit()

embedder = ImageEmbedder(device="cuda")
retriever = ImageRetriever(embedder)
retriever.build_index(docs)

queries = [
    "a cup of coffee",
    "coffee plantation field",
    "roasted coffee beans",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = retriever.retrieve(q, top_k=3)
    for r in results:
        print(f"  → {r.source}  (score: {r.metadata['clip_score']:.3f})")