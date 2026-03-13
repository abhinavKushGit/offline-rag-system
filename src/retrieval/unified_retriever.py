from src.retrieval.text_retriever import TextRetriever
from src.retrieval.image_retriever import ImageRetriever
from src.schema import Document


class UnifiedRetriever:
    def __init__(self, text_retriever: TextRetriever, image_retriever: ImageRetriever = None):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever

    def retrieve(self, query: str) -> list[dict]:
        results = []

        # Text / Audio / Video transcript results
        if self.text_retriever is not None:
            try:
                text_results = self.text_retriever.retrieve(query)
                results.extend(text_results)
            except Exception as e:
                print(f"[UnifiedRetriever] Text retrieval failed: {e}")

        # Image / Keyframe results — convert Document to dict
        if self.image_retriever is not None:
            try:
                image_results = self.image_retriever.retrieve(query, top_k=3)
                for doc in image_results:
                    results.append({
                        "text": doc.text,
                        "source": doc.source,
                        "modality": doc.modality,
                        "score": 1 - doc.metadata.get("clip_score", 0),
                        "section": doc.metadata.get("video_file", "image"),
                        "page": None,
                        "clip_score": doc.metadata.get("clip_score", 0)
                    })
            except Exception as e:
                print(f"[UnifiedRetriever] Image retrieval failed: {e}")

        # Sort by score — lower is better
        results.sort(key=lambda r: r.get("score", 999))
        return results