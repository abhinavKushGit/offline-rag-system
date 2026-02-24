# src/ingestion/text_loader.py

from pathlib import Path
from typing import List
from src.schema import Document


class TextLoader:

    def __init__(self, text_dir: str):
        self.text_dir = Path(text_dir)

    def load(self) -> List[Document]:
        documents = []

        for txt_path in self.text_dir.glob("*.txt"):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                cleaned = self._clean_text(text)
                if cleaned.strip():
                    documents.append(Document(
                        text=cleaned,
                        source=str(txt_path),
                        modality="text",
                        section="General",
                    ))

        return documents

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text