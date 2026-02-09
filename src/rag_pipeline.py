# src/rag_pipeline.py

import yaml
from transformers import AutoTokenizer

from src.chunking.fixed_chunker import FixedChunker
from src.chunking.token_chunker import TokenChunker
from src.embeddings.text_embedder import TextEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.text_retriever import TextRetriever
from src.generation.prompt_templates import build_prompt
from src.generation.generator import Generator


class RAGPipeline:
    def __init__(self):
        # ----------------------------
        # Load configuration files
        # ----------------------------
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)

        with open("config/models.yaml") as f:
            self.models = yaml.safe_load(f)

        # ----------------------------
        # Initialize chunker
        # ----------------------------
        chunk_cfg = self.config["chunking"]

        if chunk_cfg["type"] == "token":
            self.chunker = TokenChunker(
                model_name=self.models["offline_llm"]["model_name"],
                max_tokens=chunk_cfg["max_tokens"],
                overlap=chunk_cfg["overlap"],
            )
        else:
            self.chunker = FixedChunker(
                chunk_cfg["chunk_size"],
                chunk_cfg["overlap"],
                tokenizer_name=self.models["embedding_model"],
            )


        # ----------------------------
        # Initialize embedder
        # ----------------------------
        self.embedder = TextEmbedder(self.models["embedding_model"])

        # ----------------------------
        # Vector store (lazy init)
        # ----------------------------
        self.vectorstore = None

        # ----------------------------
        # Generator (offline LLM)
        # ----------------------------
        self.generator = Generator(
            self.models["offline_llm"]["model_name"],
            self.config["llm"]["temperature"],
            self.config["llm"]["max_tokens"],
        )

        # ----------------------------
        # Tokenizer (prompt-aware budgeting)
        # ----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.models["offline_llm"]["model_name"]
        )

    # ============================================================
    # INGESTION (TEXT + PDF SAFE, SECTION-AWARE INTERNALLY)
    # ============================================================
    def ingest(self, documents):
        """
        Accepts BOTH:
        - List[str]            (plain text documents)
        - List[dict]           (section-aware PDF documents)

        Internally normalizes everything to:
        {
            "section": "...",
            "text": "..."
        }
        """
        all_chunks = []
        metadata = []

        for doc in documents:
            # ----------------------------
            # Normalize input format
            # ----------------------------
            if isinstance(doc, dict):
                section = doc.get("section", "General")
                text = doc["text"]
            else:
                # Backward compatibility for Phase-1 text RAG
                section = "General"
                text = doc

            # ----------------------------
            # Chunking (section-safe)
            # ----------------------------
            chunks = self.chunker.chunk(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "text": chunk,
                    "section": section
                })

        # ----------------------------
        # Embedding + indexing
        # ----------------------------
        embeddings = self.embedder.embed(all_chunks)

        if self.vectorstore is None:
            self.vectorstore = FAISSStore(embeddings.shape[1])

        self.vectorstore.add(embeddings, metadata)

    # ============================================================
    # QUERY (PROMPT-AWARE + SOFT SECTION BIAS)
    # ============================================================
    def query(self, question: str):
        """
        Prompt-aware RAG query:
        - retrieve
        - soft section bias (NO filtering)
        - remove junk
        - prioritize dense chunks
        - token-safe context assembly
        """
        # ----------------------------
        # Retrieval
        # ----------------------------
        retriever = TextRetriever(
            self.embedder,
            self.vectorstore,
            self.config["retrieval"]["top_k"],
        )

        results = retriever.retrieve(question)

        # ----------------------------
        # SOFT SECTION BIAS (SAFE & OPTIONAL)
        # ----------------------------
        q = question.lower()

        if "objective" in q:
            results = sorted(
                results,
                key=lambda r: r.get("section", "").lower() != "course objectives"
            )
        elif "outcome" in q:
            results = sorted(
                results,
                key=lambda r: r.get("section", "").lower() != "course learning outcomes"
            )
        elif "prerequisite" in q:
            results = sorted(
                results,
                key=lambda r: r.get("section", "").lower() != "pre-requisites"
            )

        # ----------------------------
        # Extract contexts
        # ----------------------------
        contexts = [r["text"] for r in results]

        # ----------------------------
        # Remove very low-info chunks
        # ----------------------------
        contexts = [c for c in contexts if len(c.split()) > 20]

        # ----------------------------
        # Prefer shorter, denser chunks
        # ----------------------------
        contexts = sorted(contexts, key=lambda x: len(x.split()))

        # ----------------------------
        # PROMPT-AWARE TOKEN BUDGETING
        # ----------------------------
        safe_contexts = []
        TOKEN_LIMIT = 450  # total prompt budget

        for ctx in contexts:
            test_prompt = build_prompt(safe_contexts + [ctx], question)
            token_len = len(self.tokenizer.encode(test_prompt))

            if token_len > TOKEN_LIMIT:
                used_tokens = len(
                    self.tokenizer.encode(
                        build_prompt(safe_contexts, question)
                    )
                )
                available = TOKEN_LIMIT - used_tokens

                if available > 0:
                    truncated = self.tokenizer.decode(
                        self.tokenizer.encode(ctx)[:available]
                    )
                    safe_contexts.append(truncated)
                break

            safe_contexts.append(
                f"[CONTEXT CHUNK {len(safe_contexts)+1}]\n{ctx}"
            )

        # ----------------------------
        # Final prompt + generation
        # ----------------------------
        prompt = build_prompt(safe_contexts, question)
        return self.generator.generate(prompt)
