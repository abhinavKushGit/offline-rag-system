import faiss
import numpy as np
import pickle
from pathlib import Path
from src.schema import Document
from src.embeddings.image_embedder import ImageEmbedder

class ImageRetriever:
    def __init__(self, embedder: ImageEmbedder, index_dir: str = "outputs/indexes/images"):
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.documents = []

    def build_index(self, documents: list[Document]):
        self.documents = documents

        # If docs have PIL images in metadata (video keyframes) use those
        # Otherwise use file paths (regular images from Phase 4a)
        pil_images = [doc.metadata.get("_pil_image") for doc in documents]

        if all(img is not None for img in pil_images):
            print(f"[ImageRetriever] Encoding {len(pil_images)} PIL keyframes with CLIP...")
            vectors = self.embedder.encode_pil_images(pil_images)
        else:
            image_paths = [doc.source for doc in documents]
            print(f"[ImageRetriever] Encoding {len(image_paths)} images with CLIP...")
            vectors = self.embedder.encode_images(image_paths)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self._save()
        print(f"[ImageRetriever] Index built: {len(documents)} images, dim={dim}")

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        if self.index is None:
            self._load()
        query_vec = self.embedder.encode_text(query)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            doc.metadata["clip_score"] = float(score)
            results.append(doc)
        return results

    def _save(self):
        # Strip PIL images before pickling — not serializable
        docs_to_save = []
        for doc in self.documents:
            clean_meta = {k: v for k, v in doc.metadata.items() if k != "_pil_image"}
            docs_to_save.append(Document(
                text=doc.text,
                source=doc.source,
                modality=doc.modality,
                timestamp=doc.timestamp,
                metadata=clean_meta
            ))
        faiss.write_index(self.index, str(self.index_dir / "image.index"))
        with open(self.index_dir / "image_docs.pkl", "wb") as f:
            pickle.dump(docs_to_save, f)

    def _load(self):
        index_path = self.index_dir / "image.index"
        docs_path = self.index_dir / "image_docs.pkl"
        if index_path.exists() and docs_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print(f"[ImageRetriever] Loaded index: {len(self.documents)} images")
        else:
            raise FileNotFoundError("No image index found. Run build_index() first.")