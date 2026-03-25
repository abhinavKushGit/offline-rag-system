import os
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

        # Text embedder — always needed
        self.text_embedder = TextEmbedder(self.models["embedding_model"])

        # Image embedder — lazy loaded only when images present
        self.image_embedder = None

        # Stores
        self.text_vectorstore = None
        self.image_retriever = None

        # Generator — lazy loaded only when query() is called
        self.generator = None
        self._model_config = self.models["offline_llm"]

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

        # video keyframe docs have _pil_image — route to image retriever
        # video transcript docs have no _pil_image — route to text retriever
        text_docs = [d for d in documents if d.modality in ("text", "pdf", "audio")]
        text_docs += [d for d in documents if d.modality == "video"
                      and "_pil_image" not in d.metadata]
        image_docs = [d for d in documents if d.modality == "image"]
        image_docs += [d for d in documents if d.modality == "video"
                       and "_pil_image" in d.metadata]

        # Inject image captions into text pipeline as plain text docs
        caption_docs = [
            Document(
                text=doc.text,
                source=doc.source,
                modality="image",
                metadata={**doc.metadata, "is_caption": True},
            )
            for doc in image_docs
            if doc.text and len(doc.text.strip()) > 5
        ]
        text_docs = text_docs + caption_docs

        # --- TEXT / AUDIO / VIDEO TRANSCRIPTS + IMAGE CAPTIONS ---
        if text_docs:
            if source_dir is not None:
                source_hash = compute_dir_hash(source_dir)
                index_path, meta_path = get_cache_paths(cache_dir, source_hash)
                if cache_exists(cache_dir, source_hash):
                    print(f"[INFO] Cache hit — loading text index.")
                    self.text_vectorstore = FAISSStore.load(index_path, meta_path)
                    print(f"[INFO] {len(self.text_vectorstore.metadata)} chunks loaded.")
                    # still need to build image retriever even on cache hit
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

            # check if these are video keyframes — apply temporal attention if so
            is_video = any(d.modality == "video" for d in image_docs)
            if is_video and len(image_docs) > 1:
                from src.retrieval.temporal_attention import TemporalAttention

                class _AttnWrapper:
                    def __init__(self):
                        self.temporal_attn = TemporalAttention(512, 8)
                        self.temporal_attn.eval()

                    def apply_temporal_attention(self, v):
                        return self.temporal_attn.attend(v)

                self.image_retriever.build_index(
                    image_docs,
                    apply_temporal_attention=True,
                    temporal_attn=_AttnWrapper()
                )
            else:
                self.image_retriever.build_index(image_docs)

            print(f"[INFO] Indexed {len(image_docs)} images"
                  f"{' with temporal attention' if is_video else ''}.")

    # ==========================================================
    # SHARED RETRIEVAL — used by query() and query_stream()
    # ==========================================================

    def _retrieve_context(self, question: str) -> tuple[list[str], list]:
        """
        Runs full retrieval + reranking + token-budget assembly.
        Returns:
            safe_contexts : list[str]  — budget-trimmed context chunks
            results       : list[dict] — raw ranked retrieval results
        """
        if self.text_vectorstore is None and self.image_retriever is None:
            return [], []

        # Build retrievers — only what exists
        text_retriever = TextRetriever(
            self.text_embedder,
            self.text_vectorstore,
            self.config["retrieval"]["top_k"],
        ) if self.text_vectorstore is not None else None

        retriever = UnifiedRetriever(text_retriever, self.image_retriever)
        results = retriever.retrieve(question)

        # Filename pre-filter — ONLY if user explicitly mentions a filename
        # Falls back to full semantic retrieval if no filename found in query
        query_lower = question.lower()
        matched_source = None

        for r in results:
            src = r.get("source", "")
            fname = os.path.basename(src).lower()
            fname_stem = os.path.splitext(fname)[0].lower()
            # handle frame sources like "samplevid.mp4::frame_5.0s"
            if "::" in fname:
                fname = fname.split("::")[0]
                fname_stem = os.path.splitext(fname)[0]
            if fname in query_lower or fname_stem in query_lower:
                matched_source = src.split("::")[0]
                break

        if matched_source:
            filtered = [r for r in results
                        if r.get("source", "").startswith(matched_source)]
            if filtered:
                results = filtered

        if not results:
            return [], []

        # Section bias reranking
        q_words = set(question.lower().split())

        def section_bias(r):
            section_words = set(r.get("section", "").lower().split())
            overlap = len(q_words & section_words)
            return r.get("score", 1.0) - (overlap * 0.1)

        results = sorted(results, key=section_bias)

        # Filter too-short chunks
        contexts = [r["text"] for r in results]
        contexts = [c for c in contexts if len(c.split()) > 2]

        if not contexts:
            return [], results

        # Token budget assembly
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
                    safe_contexts.append(f"[CONTEXT CHUNK {len(safe_contexts) + 1}]\n{truncated}")
                break
            safe_contexts.append(f"[CONTEXT CHUNK {len(safe_contexts) + 1}]\n{ctx}")

        return safe_contexts, results

    # ==========================================================
    # QUERY — lazy loads generator, returns full string
    # ==========================================================

    def _ensure_generator(self):
        """Lazy-load Phi-3 on first call. Safe to call multiple times."""
        if self.generator is None:
            self.generator = Generator(
                model_config=self._model_config,
                temperature=self.config["llm"]["temperature"],
                max_new_tokens=self.config["llm"]["max_new_tokens"],
            )

    def query(self, question: str) -> str:
        """Run a full RAG query and return the complete answer as a string."""
        self._ensure_generator()

        if self.text_vectorstore is None and self.image_retriever is None:
            return "No documents have been ingested yet."

        safe_contexts, results = self._retrieve_context(question)

        if not safe_contexts:
            if not results:
                return "I could not find relevant information in the documents."
            return "The retrieved content was too sparse to answer the question."

        prompt = build_prompt(safe_contexts, question)
        return self.generator.generate(prompt)

    def query_stream(self, question: str):
        """
        Generator — yields string tokens one by one for SSE streaming.
        Follows the same retrieval path as query().
        """
        self._ensure_generator()

        if self.text_vectorstore is None and self.image_retriever is None:
            yield "No documents have been ingested yet."
            return

        safe_contexts, results = self._retrieve_context(question)

        if not safe_contexts:
            if not results:
                yield "I could not find relevant information in the documents."
            else:
                yield "The retrieved content was too sparse to answer the question."
            return

        prompt = build_prompt(safe_contexts, question)
        yield from self.generator.generate_stream(prompt)