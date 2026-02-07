import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int):
        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
