import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.ingestion.pdf_loader import PDFLoader
from src.evaluation.faithfulness import faithfulness_score
from src.evaluation.latency import measure_latency

rag = RAGPipeline()
loader = PDFLoader("data/pdf")

documents = loader.load()
rag.ingest(documents)

query = "What are the course objectives?"

answer, latency = measure_latency(rag.query, query)

retriever = rag.vectorstore  
print("\nAnswer:\n", answer)
print(f"\nLatency: {latency:.3f} seconds")
