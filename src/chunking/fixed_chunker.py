from typing import List
from transformers import AutoTokenizer


class FixedChunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 512,
    ):
        self.chunk_size = chunk_size          # target tokens per chunk
        self.overlap = overlap                # token overlap
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            # 🔒 HARD SAFETY CHECK
            if len(chunk_tokens) > self.max_tokens:
                chunk_tokens = chunk_tokens[:self.max_tokens]

            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start = end - self.overlap
            if start < 0:
                start = 0

        return chunks
