import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.text_loader import TextLoader
from src.ingestion.pdf_loader import PDFLoader


def main():
    print("=== RAG SYSTEM ===")

    source = input("Choose source (text/pdf): ").strip().lower()
    if source not in {"text", "pdf"}:
        print("❌ Invalid source. Choose 'text' or 'pdf'.")
        return

    # ----------------------------
    # Ask for directory path
    # ----------------------------
    path = input("Enter directory path: ").strip()
    if not Path(path).exists():
        print("❌ Path does not exist.")
        return

    # ----------------------------
    # Initialize pipeline
    # ----------------------------
    rag = RAGPipeline()

    # ----------------------------
    # Load documents
    # ----------------------------
    if source == "text":
        loader = TextLoader(path)
    else:
        loader = PDFLoader(path)

    documents = loader.load()

    if not documents:
        print("❌ No documents found.")
        return

    print(f"[INFO] Loaded {len(documents)} sections")

    # ----------------------------
    # Ingest into vector store
    # ----------------------------
    print("[INFO] Ingesting documents...")
    rag.ingest(documents)
    print("[INFO] Ingestion complete.")

    # ----------------------------
    # Interactive query loop
    # ----------------------------
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("Exiting RAG session.")
            break

        answer = rag.query(query)

        print("\n=== ANSWER ===")
        print(answer)


if __name__ == "__main__":
    main()
