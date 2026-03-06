import torch
import open_clip
from PIL import Image
import numpy as np

class ImageEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ImageEmbedder] Loading CLIP on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """Encode from file paths — used by Phase 4a (image files)"""
        vectors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            vectors.append(self._encode_pil(img))
        return np.vstack(vectors).astype("float32")

    def encode_pil_images(self, pil_images: list) -> np.ndarray:
        """Encode from PIL images directly — used by Phase 4c (video keyframes)"""
        vectors = [self._encode_pil(img) for img in pil_images]
        return np.vstack(vectors).astype("float32")

    def _encode_pil(self, img: Image.Image) -> np.ndarray:
        """Shared internal encoding logic"""
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy()

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query into CLIP vector space"""
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            vec = self.model.encode_text(tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().astype("float32")