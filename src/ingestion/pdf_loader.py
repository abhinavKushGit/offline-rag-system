from typing import List
from pathlib import Path
from pypdf import PdfReader
import re


class PDFLoader:

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)

    def load(self) -> List[str]:
        documents = []

        for pdf_path in self.pdf_dir.glob("*.pdf"):
            text = self._extract_text(pdf_path)
            if text.strip():
                documents.append(text)

        return documents

    def _extract_text(self, pdf_path: Path) -> str:
        reader = PdfReader(pdf_path)
        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if not page_text:
                continue

            page_text = self._normalize_sections(page_text)
            pages_text.append(page_text)

        return "\n".join(pages_text)

    def _normalize_sections(self, text: str) -> str:
        
        headings = [
            "Course Objectives",
            "Course Learning Outcomes",
            "Course Contents",
            "Course Curriculum",
            "Pedagogy for Course Delivery",
            "Course Description",
        ]

        for h in headings:
            text = re.sub(
                rf"\s*{h}\s*",
                f"\n\n{h}\n\n",
                text,
                flags=re.IGNORECASE
            )

        return text
