import pickle
import faiss
import numpy as np


class FAISSStore:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int, threshold: float = 2.0) -> list[dict]:
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and dist < threshold:
                results.append({
                    **self.metadata[idx],
                    "score": float(dist),   # lower = more similar (L2)
                })

        results.sort(key=lambda x: x["score"])
        return results

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Index saved → {index_path}")

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FAISSStore":
        store = cls.__new__(cls)
        store.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
        return store