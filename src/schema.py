from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    text: str
    source: str                          # file path
    modality: str = "text"               # text | pdf | image | audio | video
    section: Optional[str] = None
    page: Optional[int] = None
    timestamp: Optional[float] = None    # for audio/video chunks
    metadata: dict = field(default_factory=dict)