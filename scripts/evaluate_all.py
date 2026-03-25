"""
OmniRAG — Comprehensive Evaluation Script
==========================================
Runs across all 5 modalities and produces:
  - Ingestion time per modality
  - Query latency (avg of N queries)
  - Faithfulness score (ROUGE-L answer vs context)
  - Recall@k (phrase-based)
  - A printable results table for synopsis

Usage:
    cd ~/Desktop/RAGproject
    source rag-env/bin/activate
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python scripts/evaluate_all.py
"""

import sys
import time
import json
import gc
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import recall_at_k
from src.evaluation.faithfulness import faithfulness_score

# ─────────────────────────────────────────────────────────────────────────────
# EVAL SETS — matched to actual files in data/
#
# text/sample.txt  → coffee article (origins, Ethiopia, Arabica/Robusta,
#                    roasting, brewing, health effects, global industry)
# pdf/testpdf.pdf  → another coffee document
# images/          → airplane.png, coffee.png, coffee2.png,
#                    testimage.png (girl with fruits),
#                    testimage2.png (girl with frisbee)
# audio/           → TEDx talk "A one minute TEDx Talk for the digital age"
# video/           → samplevid1.mp4 — animals video
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SETS = {
    "text": {
        "data_path": "data/text",
        "queries": [
            {
                "question": "Where did coffee originate and how was it discovered?",
                "relevant_phrases": ["ethiopia", "kaldi", "goat", "berries", "arabian"],
            },
            {
                "question": "What is the difference between Arabica and Robusta coffee?",
                "relevant_phrases": ["arabica", "robusta", "altitude", "caffeine", "taste"],
            },
            {
                "question": "How is coffee roasted and what are the different roast levels?",
                "relevant_phrases": ["roast", "light", "medium", "dark", "temperature"],
            },
            {
                "question": "What are the health effects of drinking coffee?",
                "relevant_phrases": ["caffeine", "diabetes", "parkinson", "liver", "blood pressure"],
            },
        ],
    },

    "pdf": {
        "data_path": "data/pdf",
        "queries": [
            {
                "question": "What does this document say about coffee production?",
                "relevant_phrases": ["coffee", "production", "bean", "farm", "grow"],
            },
            {
                "question": "Which countries are the largest coffee producers?",
                "relevant_phrases": ["brazil", "vietnam", "colombia", "producer", "supply"],
            },
        ],
    },

    "image": {
        "data_path": "data/images",
        "queries": [
            {
                "question": "Is there an airplane in any of the images?",
                "relevant_phrases": ["airplane", "aircraft", "jet", "sky", "flight", "white"],
            },
            {
                "question": "Describe any images showing people or human subjects.",
                "relevant_phrases": ["woman", "girl", "person", "holding", "wearing", "hand"],
            },
            {
                "question": "Are there images showing food, drinks or fruit?",
                "relevant_phrases": ["coffee", "cup", "latte", "fruit", "apple", "orange", "drink"],
            },
        ],
    },

    "audio": {
        "data_path": "data/audio",
        "queries": [
            {
                "question": "What is the main message of the TEDx talk?",
                "relevant_phrases": ["digital", "age", "talk", "message", "minute"],
            },
            {
                "question": "What does the speaker say about technology or the modern world?",
                "relevant_phrases": ["digital", "technology", "today", "world", "future", "change"],
            },
        ],
    },

    "video": {
        "data_path": "data/video",
        "queries": [
            {
                "question": "What animals are shown in the video?",
                "relevant_phrases": ["animal", "wildlife", "bird", "mammal", "creature", "species"],
            },
            {
                "question": "What is the environment or setting visible in the video?",
                "relevant_phrases": ["background", "nature", "forest", "grass", "outdoor", "environment"],
            },
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def ingest_modality(modality: str, data_path: str):
    """
    Runs full ingestion for a modality.
    Returns (RAGPipeline, ingestion_time_seconds, doc_count, chunk_count).
    """
    from src.rag_pipeline import RAGPipeline

    t0 = time.time()

    if modality == "text":
        from src.ingestion.text_loader import TextLoader
        docs = TextLoader(data_path).load()

    elif modality == "pdf":
        from src.ingestion.pdf_loader import PDFLoader
        docs = PDFLoader(data_path).load()

    elif modality == "image":
        from src.ingestion.image_captioner import ImageCaptioner
        captioner = ImageCaptioner()
        docs = captioner.caption_dir(data_path)
        captioner.unload()
        free_vram()

    elif modality == "audio":
        from src.ingestion.audio_transcriber import AudioTranscriber
        transcriber = AudioTranscriber(model_size="small", device="cuda")
        docs = transcriber.transcribe(data_path)
        del transcriber
        free_vram()

    elif modality == "video":
        from src.ingestion.video_processor import VideoProcessor
        from src.ingestion.video_captioner import VideoCaptioner
        processor = VideoProcessor(keyframe_interval=2, device="cuda")
        transcript_docs, keyframe_images, keyframe_sources = processor.process(data_path)
        processor.unload()
        free_vram()
        captioner = VideoCaptioner()
        keyframe_docs = captioner.caption_frames(keyframe_images, keyframe_sources)
        captioner.unload()
        free_vram()
        docs = transcript_docs + keyframe_docs

    else:
        raise ValueError(f"Unknown modality: {modality}")

    doc_count = len(docs)

    pipeline = RAGPipeline()
    pipeline.ingest(docs, source_dir=data_path)

    ingestion_time = time.time() - t0

    chunk_count = 0
    if pipeline.text_vectorstore is not None:
        try:
            chunk_count = len(pipeline.text_vectorstore.metadata)
        except Exception:
            chunk_count = -1

    return pipeline, ingestion_time, doc_count, chunk_count


def evaluate_pipeline(pipeline, queries: list[dict], top_k: int = 10) -> dict:
    """
    Runs all queries against a pipeline.
    Returns aggregate metrics dict.
    """
    from src.retrieval.text_retriever import TextRetriever

    latencies           = []
    faithfulness_scores = []
    recall_scores       = []

    for item in queries:
        question         = item["question"]
        relevant_phrases = item["relevant_phrases"]

        # ── Query + latency ────────────────────────────────────────
        t0 = time.time()
        answer = pipeline.query(question)
        query_latency = time.time() - t0
        latencies.append(query_latency)

        # ── Recall@k ───────────────────────────────────────────────
        if pipeline.text_vectorstore is not None:
            retriever = TextRetriever(
                pipeline.text_embedder,
                pipeline.text_vectorstore,
                top_k,
            )
            retrieved = retriever.retrieve(question)
            chunks = [r["text"] for r in retrieved]
        else:
            chunks = []

        recall = recall_at_k(chunks, relevant_phrases, k=top_k)
        recall_scores.append(recall)

        # ── Faithfulness (ROUGE-L answer vs context) ───────────────
        faith = faithfulness_score(answer, chunks[:5])
        faithfulness_scores.append(faith)

        print(f"  Q: {question}")
        print(f"     Latency={query_latency:.2f}s  "
              f"Recall@{top_k}={recall:.2f}  "
              f"Faithfulness={faith:.3f}")
        print(f"     A: {answer[:150]}\n")

    return {
        "avg_latency":      round(sum(latencies) / len(latencies), 2),
        "max_latency":      round(max(latencies), 2),
        "avg_recall":       round(sum(recall_scores) / len(recall_scores), 3),
        "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "n_queries":        len(queries),
        "per_query": [
            {
                "question":     q["question"],
                "latency":      round(latencies[i], 2),
                "recall":       round(recall_scores[i], 3),
                "faithfulness": round(faithfulness_scores[i], 4),
            }
            for i, q in enumerate(queries)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    top_k = 10
    results = {}

    modalities_to_run = []
    for modality, cfg in EVAL_SETS.items():
        path = Path(cfg["data_path"])
        if path.exists() and any(path.iterdir()):
            modalities_to_run.append(modality)
        else:
            print(f"[SKIP] {modality} — no files found at {cfg['data_path']}")

    print(f"\n{'='*60}")
    print(f"  OmniRAG Evaluation — {len(modalities_to_run)} modalities")
    print(f"{'='*60}\n")

    for modality in modalities_to_run:
        cfg = EVAL_SETS[modality]
        print(f"\n{'─'*50}")
        print(f"  MODALITY: {modality.upper()}")
        print(f"{'─'*50}")

        print(f"[1] Ingesting {cfg['data_path']} …")
        try:
            pipeline, ingest_time, doc_count, chunk_count = ingest_modality(
                modality, cfg["data_path"]
            )
            print(f"    → {doc_count} docs | {chunk_count} chunks | {ingest_time:.1f}s")
        except Exception as e:
            print(f"    ✗ Ingestion failed: {e}")
            results[modality] = {"error": str(e)}
            continue

        print(f"[2] Running {len(cfg['queries'])} queries …\n")
        try:
            metrics = evaluate_pipeline(pipeline, cfg["queries"], top_k=top_k)
        except Exception as e:
            print(f"    ✗ Query failed: {e}")
            metrics = {"error": str(e)}

        results[modality] = {
            "ingestion_time_s": round(ingest_time, 1),
            "doc_count":        doc_count,
            "chunk_count":      chunk_count,
            **metrics,
        }

        del pipeline
        free_vram()

    # ── Results table ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Modality':<10} {'Ingest(s)':<12} {'Docs':<8} {'Chunks':<10} "
          f"{'Recall@'+str(top_k):<12} {'Faithfulness':<14} {'Avg Latency(s)'}")
    print(f"{'─'*70}")

    for modality in modalities_to_run:
        r = results.get(modality, {})
        if "error" in r:
            print(f"{modality:<10}  ERROR: {r['error']}")
            continue
        print(
            f"{modality:<10} "
            f"{r.get('ingestion_time_s', '-'):<12} "
            f"{r.get('doc_count', '-'):<8} "
            f"{r.get('chunk_count', '-'):<10} "
            f"{r.get('avg_recall', '-'):<12} "
            f"{r.get('avg_faithfulness', '-'):<14} "
            f"{r.get('avg_latency', '-')}"
        )
    print(f"{'='*70}\n")

    # ── Save JSON for plotting ─────────────────────────────────────
    out_path = Path("outputs/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved → {out_path}")
    print("Run  python scripts/plot_results.py  to generate charts.\n")


if __name__ == "__main__":
    main()