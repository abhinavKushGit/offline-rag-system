from typing import List
from transformers import AutoTokenizer


class TokenChunker:

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        overlap: int = 50,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []

        current_tokens = []
        current_len = 0

        for para in paragraphs:
            para_tokens = self.tokenizer.encode(
                para, add_special_tokens=False
            )

            if len(para_tokens) > self.max_tokens:
                for i in range(0, len(para_tokens), self.max_tokens):
                    sub = para_tokens[i:i + self.max_tokens]
                    chunks.append(
                        self.tokenizer.decode(sub, skip_special_tokens=True)
                    )
                current_tokens = []
                current_len = 0
                continue

            if current_len + len(para_tokens) > self.max_tokens:
                chunks.append(
                    self.tokenizer.decode(
                        current_tokens,
                        skip_special_tokens=True
                    )
                )

                overlap_tokens = current_tokens[-self.overlap:] if self.overlap > 0 else []
                current_tokens = overlap_tokens + para_tokens
                current_len = len(current_tokens)
            else:
                current_tokens.extend(para_tokens)
                current_len += len(para_tokens)

        if current_tokens:
            chunks.append(
                self.tokenizer.decode(
                    current_tokens,
                    skip_special_tokens=True
                )
            )

        return chunks
