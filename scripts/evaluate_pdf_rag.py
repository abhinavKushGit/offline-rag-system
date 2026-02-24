# scripts/evaluate_pdf_rag.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.pdf_loader import PDFLoader
from src.evaluation.faithfulness import faithfulness_score
from src.evaluation.latency import measure_latency
from src.retrieval.text_retriever import TextRetriever


def main():
    # 1. Setup pipeline
    rag = RAGPipeline()

    # 2. Load PDF documents
    loader = PDFLoader("data/pdf")
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} PDF documents")

    # 3. Ingest
    rag.ingest(documents)

    # 4. Query
    query = "What are the course objectives?"

    answer, latency = measure_latency(rag.query, query)

    # 5. Retrieve contexts again for evaluation
    retriever = TextRetriever(
        rag.embedder,
        rag.vectorstore,
        rag.config["retrieval"]["top_k"]
    )
    retrieved = retriever.retrieve(query)
    contexts = [r["text"] for r in retrieved]

    # 6. Faithfulness
    faithfulness = faithfulness_score(answer, contexts)

    # 7. Print results
    print("\n=== QUERY ===")
    print(query)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== EVALUATION ===")
    print(f"Latency: {latency:.3f} seconds")
    print(f"Faithfulness score: {faithfulness:.2f}")

    print("\n=== CONTEXT PREVIEW (Top 2) ===")
    for i, ctx in enumerate(contexts[:2], 1):
        print(f"\n--- Context {i} ---")
        print(ctx[:500])


if __name__ == "__main__":
    main()
