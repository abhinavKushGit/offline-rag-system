import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.pdf_loader import PDFLoader


loader = PDFLoader("data/pdf")
docs = loader.load()

print("Number of PDFs loaded:", len(docs))
print("\n--- Preview ---\n")
print(docs[0][:500])
