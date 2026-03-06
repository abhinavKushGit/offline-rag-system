from pathlib import Path
from PIL import Image
from src.schema import Document

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ImageLoader:
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)

    def load(self) -> list[Document]:
        docs = []
        for file in sorted(self.image_dir.iterdir()):
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                img = Image.open(file).convert("RGB")
                caption = file.stem.replace("_", " ").replace("-", " ")
                doc = Document(
                    text=caption,
                    source=str(file),
                    modality="image",
                    metadata={"width": img.width, "height": img.height}
                )
                docs.append(doc)
            except Exception as e:
                print(f"[ImageLoader] Failed to load {file}: {e}")
        print(f"[ImageLoader] Loaded {len(docs)} images from {self.image_dir}")
        return docs