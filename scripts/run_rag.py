import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.text_loader import TextLoader
from src.ingestion.pdf_loader import PDFLoader


def main():
    parser = argparse.ArgumentParser(description="Run RAG Pipeline")
    parser.add_argument(
        "--source",
        choices=["text", "pdf"],
        default="text",
        help="Choose ingestion source: text or pdf",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to ask the RAG system",
    )

    args = parser.parse_args()

    rag = RAGPipeline()

    if args.source == "text":
        loader = TextLoader("data/text")
    else:
        loader = PDFLoader("data/pdf")

    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} documents from {args.source} source")

    rag.ingest(documents)

    answer = rag.query(args.query)
    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
