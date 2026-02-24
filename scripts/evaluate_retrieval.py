import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.pdf_loader import PDFLoader
from src.retrieval.text_retriever import TextRetriever
from src.evaluation.retrieval_metrics import recall_at_k
from scripts.retrieval_ground_truth import EVAL_SET


def main():
    rag = RAGPipeline()
    loader = PDFLoader("data/pdf")

    documents = loader.load()
    rag.ingest(documents)

    retriever = TextRetriever(
        rag.embedder,
        rag.vectorstore,
        rag.config["retrieval"]["top_k"]
    )

    for item in EVAL_SET:
        query = item["query"]
        relevant_phrases = item["relevant_phrases"]

        retrieved = retriever.retrieve(query)
        retrieved_chunks = [r["text"] for r in retrieved]

        recall = recall_at_k(
            retrieved_chunks,
            relevant_phrases,
            k=rag.config["retrieval"]["top_k"]
        )

        print("\nQUERY:", query)
        print("Recall@k:", round(recall, 2))


if __name__ == "__main__":
    main()
