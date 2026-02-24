from typing import List
from pathlib import Path
from src.schema import Document
import pdfplumber


class PDFLoader:

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)

    def load(self) -> List[Document]:
        documents = []
        for pdf_path in self.pdf_dir.glob("*.pdf"):
            sections = self._extract_sections(pdf_path)
            documents.extend(sections)
        return documents

    def _extract_sections(self, pdf_path: Path) -> List[Document]:
        documents = []
        current_heading = "General"
        current_text = []
        current_page = 1

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    if self._is_heading(line):
                        # Flush previous section
                        if current_text:
                            joined = " ".join(current_text).strip()
                            if joined:
                                documents.append(Document(
                                    text=joined,
                                    source=str(pdf_path),
                                    modality="pdf",
                                    section=current_heading,
                                    page=current_page,
                                ))
                        current_heading = line
                        current_text = []
                        current_page = page_num
                    else:
                        current_text.append(line)

        # Flush final section
        if current_text:
            joined = " ".join(current_text).strip()
            if joined:
                documents.append(Document(
                    text=joined,
                    source=str(pdf_path),
                    modality="pdf",
                    section=current_heading,
                    page=current_page,
                ))

        return documents

    def _is_heading(self, line: str) -> bool:
        """
        Generic heuristic — works on any PDF, not hardcoded to one file.
        A heading is: short, mostly title-cased, no sentence-ending punctuation.
        """
        line = line.strip()
        if not line or len(line) > 100:
            return False
        if line.endswith(".") or line.endswith(",") or line.endswith(":"):
            return False
        words = line.split()
        if len(words) == 0 or len(words) > 12:
            return False
        capitalized = sum(1 for w in words if w and w[0].isupper())
        return (capitalized / len(words)) > 0.6