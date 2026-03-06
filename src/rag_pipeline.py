import yaml
from transformers import AutoTokenizer

from src.schema import Document
from src.chunking.token_chunker import TokenChunker
from src.chunking.fixed_chunker import FixedChunker
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.image_retriever import ImageRetriever
from src.retrieval.unified_retriever import UnifiedRetriever
from src.generation.prompt_templates import build_prompt
from src.generation.generator import Generator
from src.utils.cache import compute_dir_hash, get_cache_paths, cache_exists


class RAGPipeline:

    def __init__(self):
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)
        with open("config/models.yaml") as f:
            self.models = yaml.safe_load(f)

        # Chunker
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

        # Embedders
        self.text_embedder = TextEmbedder(self.models["embedding_model"])
        self.image_embedder = None

        # Stores
        self.text_vectorstore = None
        self.image_retriever = None

        # Generator
        self.generator = Generator(
            model_config=self.models["offline_llm"],
            temperature=self.config["llm"]["temperature"],
            max_new_tokens=self.config["llm"]["max_new_tokens"],
        )

        # Tokenizer for budget
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.models["embedding_model"],
            model_max_length=4096,
        )

    # ==========================================================
    # INGESTION
    # ==========================================================

    def ingest(self, documents: list[Document], source_dir: str = None):
        cache_dir = "outputs/indexes"

        # Separate by modality
        text_docs = [d for d in documents if d.modality in ("text", "pdf", "audio", "video")]
        image_docs = [d for d in documents if d.modality == "image"]

        # --- TEXT/AUDIO/VIDEO TRANSCRIPTS ---
        if text_docs:
            if source_dir is not None:
                source_hash = compute_dir_hash(source_dir)
                index_path, meta_path = get_cache_paths(cache_dir, source_hash)
                if cache_exists(cache_dir, source_hash):
                    print(f"[INFO] Cache hit — loading text index.")
                    self.text_vectorstore = FAISSStore.load(index_path, meta_path)
                    print(f"[INFO] {len(self.text_vectorstore.metadata)} chunks loaded.")
                    return

            chunks = self.chunker.chunk(text_docs)
            print(f"[INFO] Total chunks: {len(chunks)}")

            texts = [c.text for c in chunks]
            embeddings = self.text_embedder.embed(texts)

            metadata = [
                {
                    "text": c.text,
                    "section": c.section or "General",
                    "source": c.source,
                    "page": c.page,
                    "modality": c.modality,
                    "start_time": c.metadata.get("start_time"),
                }
                for c in chunks
            ]

            self.text_vectorstore = FAISSStore(embeddings.shape[1])
            self.text_vectorstore.add(embeddings, metadata)
            print(f"[INFO] Indexed {len(chunks)} text chunks.")

            if source_dir is not None:
                source_hash = compute_dir_hash(source_dir)
                index_path, meta_path = get_cache_paths(cache_dir, source_hash)
                self.text_vectorstore.save(index_path, meta_path)

        # --- IMAGES / KEYFRAMES ---
        if image_docs:
            if self.image_embedder is None:
                self.image_embedder = ImageEmbedder(
                model_name=self.models["clip"]["model"],
                pretrained=self.models["clip"]["pretrained"],
                device=self.models["clip"]["device"]
            )
        self.image_retriever = ImageRetriever(
            self.image_embedder,
            index_dir="outputs/indexes/images"
        )
        self.image_retriever.build_index(image_docs)
        print(f"[INFO] Indexed {len(image_docs)} images.")

    # ==========================================================
    # QUERY
    # ==========================================================

    def query(self, question: str) -> str:
        if self.text_vectorstore is None and self.image_retriever is None:
            return "No documents have been ingested yet."

        # Build unified retriever
        text_retriever = TextRetriever(
            self.text_embedder,
            self.text_vectorstore,
            self.config["retrieval"]["top_k"],
        ) if self.text_vectorstore else None

        retriever = UnifiedRetriever(text_retriever, self.image_retriever)
        results = retriever.retrieve(question)

        if not results:
            return "I could not find relevant information in the documents."

        # Section bias
        q_words = set(question.lower().split())
        def section_bias(r):
            section_words = set(r.get("section", "").lower().split())
            overlap = len(q_words & section_words)
            return r.get("score", 1.0) - (overlap * 0.1)
        results = sorted(results, key=section_bias)

        # Filter short
        contexts = [r["text"] for r in results]
        contexts = [c for c in contexts if len(c.split()) > 5]

        if not contexts:
            return "The retrieved content was too sparse to answer the question."

        # Token budget
        TOKEN_LIMIT = self.config.get("token_budget", {}).get("total", 3500)
        safe_contexts = []
        for ctx in contexts:
            test_prompt = build_prompt(safe_contexts + [ctx], question)
            token_len = len(self.tokenizer.encode(test_prompt, add_special_tokens=False))
            if token_len > TOKEN_LIMIT:
                used = len(self.tokenizer.encode(
                    build_prompt(safe_contexts, question), add_special_tokens=False
                ))
                available = TOKEN_LIMIT - used
                if available > 50:
                    truncated = self.tokenizer.decode(
                        self.tokenizer.encode(ctx, add_special_tokens=False)[:available]
                    )
                    safe_contexts.append(f"[CONTEXT CHUNK {len(safe_contexts)+1}]\n{truncated}")
                break
            safe_contexts.append(f"[CONTEXT CHUNK {len(safe_contexts)+1}]\n{ctx}")

        prompt = build_prompt(safe_contexts, question)
        return self.generator.generate(prompt)