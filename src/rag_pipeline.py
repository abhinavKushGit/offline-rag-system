# src/rag_pipeline.py

import yaml
from transformers import AutoTokenizer

from src.schema import Document
from src.chunking.fixed_chunker import FixedChunker
from src.chunking.token_chunker import TokenChunker
from src.embeddings.text_embedder import TextEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.text_retriever import TextRetriever
from src.generation.prompt_templates import build_prompt
from src.generation.generator import Generator
from src.utils.cache import compute_dir_hash, get_cache_paths, cache_exists


class RAGPipeline:

    def __init__(self):
        # ----------------------------------------
        # Load config
        # ----------------------------------------
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)
        with open("config/models.yaml") as f:
            self.models = yaml.safe_load(f)

        # ----------------------------------------
        # Chunker
        # ----------------------------------------
        chunk_cfg = self.config["chunking"]
        if chunk_cfg["type"] == "token":
            self.chunker = TokenChunker(
                model_name=self.models["embedding_model"],
                max_tokens=chunk_cfg["max_tokens"],
                overlap=chunk_cfg["overlap"],
            )
        else:
            self.chunker = FixedChunker(
                chunk_cfg["chunk_size"],
                chunk_cfg["overlap"],
                tokenizer_name=self.models["embedding_model"],
            )

        # ----------------------------------------
        # Embedder
        # ----------------------------------------
        self.embedder = TextEmbedder(self.models["embedding_model"])

        # ----------------------------------------
        # Vectorstore (built during ingest)
        # ----------------------------------------
        self.vectorstore = None

        # ----------------------------------------
        # Generator (pluggable backend)
        # ----------------------------------------
        self.generator = Generator(
            model_config=self.models["offline_llm"],
            temperature=self.config["llm"]["temperature"],
            max_new_tokens=self.config["llm"]["max_new_tokens"],
        )

        # ----------------------------------------
        # Tokenizer for prompt budget counting
        # model_max_length set high to avoid warnings —
        # we manage the budget ourselves in query()
        # ----------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.models["embedding_model"],
            model_max_length=4096,
        )

    # ==========================================================
    # INGESTION
    # ==========================================================

    def ingest(self, documents: list, source_dir: str = None):
        cache_dir = "outputs/indexes"

        # -- Cache check --
        if source_dir is not None:
            source_hash = compute_dir_hash(source_dir)
            index_path, meta_path = get_cache_paths(cache_dir, source_hash)

            if cache_exists(cache_dir, source_hash):
                print(f"[INFO] Cache hit ({source_hash[:8]}...) — loading index.")
                self.vectorstore = FAISSStore.load(index_path, meta_path)
                print(f"[INFO] {len(self.vectorstore.metadata)} chunks loaded.")
                return

        # -- Build from scratch --
        all_chunks = []
        metadata = []

        for doc in documents:
            if isinstance(doc, Document):
                section = doc.section or "General"
                text = doc.text
                source = doc.source
                page = doc.page
            elif isinstance(doc, dict):
                section = doc.get("section", "General")
                text = doc["text"]
                source = doc.get("source", "unknown")
                page = doc.get("page", None)
            else:
                section = "General"
                text = str(doc)
                source = "unknown"
                page = None

            chunks = self.chunker.chunk(text)
            print(f"[INFO] {source} — {len(chunks)} chunks")

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "text": chunk,
                    "section": section,
                    "source": source,
                    "page": page,
                })

        if not all_chunks:
            print("[WARN] No chunks produced. Check your documents.")
            return

        embeddings = self.embedder.embed(all_chunks)

        if self.vectorstore is None:
            self.vectorstore = FAISSStore(embeddings.shape[1])

        self.vectorstore.add(embeddings, metadata)
        print(f"[INFO] Indexed {len(all_chunks)} chunks total.")

        # -- Save cache --
        if source_dir is not None:
            self.vectorstore.save(index_path, meta_path)
            print(f"[INFO] Index cached ({source_hash[:8]}...).")

    # ==========================================================
    # QUERY
    # ==========================================================

    def query(self, question: str) -> str:
        if self.vectorstore is None:
            return "No documents have been ingested yet."

        # -- Retrieve --
        retriever = TextRetriever(
            self.embedder,
            self.vectorstore,
            self.config["retrieval"]["top_k"],
        )
        results = retriever.retrieve(question)

        if not results:
            return "I could not find relevant information in the documents."

        # -- Dynamic section bias --
        q_words = set(question.lower().split())

        def section_bias(r):
            section_words = set(r.get("section", "").lower().split())
            overlap = len(q_words & section_words)
            return r.get("score", 1.0) - (overlap * 0.1)

        results = sorted(results, key=section_bias)

        # -- Extract + filter --
        contexts = [r["text"] for r in results]
        contexts = [c for c in contexts if len(c.split()) > 15]

        if not contexts:
            return "The retrieved content was too sparse to answer the question."

        # -- Token-budget-aware context assembly --
        TOKEN_LIMIT = self.config.get("token_budget", {}).get("total", 3500)
        safe_contexts = []

        for ctx in contexts:
            test_prompt = build_prompt(safe_contexts + [ctx], question)
            token_len = len(self.tokenizer.encode(
                test_prompt, add_special_tokens=False
            ))

            if token_len > TOKEN_LIMIT:
                used = len(self.tokenizer.encode(
                    build_prompt(safe_contexts, question),
                    add_special_tokens=False,
                ))
                available = TOKEN_LIMIT - used
                if available > 50:
                    truncated = self.tokenizer.decode(
                        self.tokenizer.encode(
                            ctx, add_special_tokens=False
                        )[:available]
                    )
                    safe_contexts.append(
                        f"[CONTEXT CHUNK {len(safe_contexts)+1}]\n{truncated}"
                    )
                break

            safe_contexts.append(
                f"[CONTEXT CHUNK {len(safe_contexts)+1}]\n{ctx}"
            )

        prompt = build_prompt(safe_contexts, question)
        return self.generator.generate(prompt)