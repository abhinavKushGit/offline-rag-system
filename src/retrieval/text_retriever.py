class TextRetriever:
    def __init__(self, embedder, vectorstore, top_k: int):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str):
        query_vec = self.embedder.embed([query])
        results = self.vectorstore.search(query_vec, self.top_k)
        return results
