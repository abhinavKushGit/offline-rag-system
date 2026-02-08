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

    def ingest(self, documents: list[str]):
        """
        Ingest a list of raw text documents into the vector store.
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        embeddings = self.embedder.embed(all_chunks)

        if self.vectorstore is None:
            self.vectorstore = FAISSStore(embeddings.shape[1])

        metadata = [{"text": chunk} for chunk in all_chunks]
        self.vectorstore.add(embeddings, metadata)

    def query(self, question: str):
        """
        Prompt-aware RAG query:
        - retrieve
        - filter junk
        - prioritize dense chunks
        - build prompt incrementally
        - truncate safely instead of dropping
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
        TOKEN_LIMIT = 450  # budget for FULL prompt

        for ctx in contexts:
            test_prompt = build_prompt(safe_contexts + [ctx], question)
            token_len = len(self.tokenizer.encode(test_prompt))

            if token_len > TOKEN_LIMIT:
                # 🔑 SAFETY TRUNCATION (NOT DROP)
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

            safe_contexts.append(ctx)

        # ----------------------------
        # Final prompt + generation
        # ----------------------------
        prompt = build_prompt(safe_contexts, question)
        return self.generator.generate(prompt)
