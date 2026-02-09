from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader
import re


class PDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)

    def load(self) -> List[Dict]:
        documents = []

        for pdf_path in self.pdf_dir.glob("*.pdf"):
            sections = self._extract_sections(pdf_path)
            documents.extend(sections)

        return documents

    def _extract_sections(self, pdf_path: Path) -> List[Dict]:
        reader = PdfReader(pdf_path)

        raw_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text.append(text)

        full_text = "\n".join(raw_text)

        heading_pattern = re.compile(
            r"(Course Objectives|Course Learning Outcomes|Pre-Requisites|Course Contents|Course Curriculum|Course Description)",
            re.IGNORECASE
        )

        splits = heading_pattern.split(full_text)

        sections = []
        current_heading = "General"

        for part in splits:
            part = part.strip()
            if not part:
                continue

            if heading_pattern.fullmatch(part):
                current_heading = part
            else:
                sections.append({
                    "section": current_heading,
                    "text": part
                })

        return sections
