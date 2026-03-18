import torch
import gc
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from src.schema import Document

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CAPTION_PROMPT = """Describe this image in complete detail. Include:
- First, state whether this image is in color or black and white / grayscale
- Every visible object (if color: include exact colors; if black and white: describe tones only, do not guess colors)
- What people are doing, their appearance and clothing
- Any text or writing visible anywhere
- The background, setting, and lighting conditions
- Spatial relationships between objects
- Any actions or events taking place
Be specific and thorough."""

VIDEO_FRAME_PROMPT = """This is a frame from a video. Describe what is happening in this video frame in complete detail. Include:
- First, state whether this frame is in color or black and white / grayscale
- Every visible object and animal (if color: include exact colors; if black and white: describe tones only)
- What people or animals are doing, their appearance
- Any text or writing visible anywhere in the frame
- The background, setting, environment and lighting
- Spatial relationships between subjects
- Any actions, motion or events visible
Describe this as a video scene, not a static image."""


class ImageCaptioner:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is not None:
            return

        print(f"[ImageCaptioner] Loading {self.model_id} with 4-bit quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[ImageCaptioner] Qwen2-VL loaded.")

    def _caption_pil_with_prompt(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        caption = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        del inputs, output_ids, generated, image_inputs
        torch.cuda.empty_cache()

        return caption

    def _caption_pil(self, image: Image.Image) -> str:
        return self._caption_pil_with_prompt(image, CAPTION_PROMPT)

    def caption_dir(self, image_dir: str) -> list[Document]:
        self._load()
        docs = []
        for path in sorted(Path(image_dir).iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            docs.extend(self.caption_file(str(path)))
        return docs

    def caption_file(self, file_path: str) -> list[Document]:
        self._load()
        path = Path(file_path)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[ImageCaptioner] Could not open {path.name}: {e}")
            return []

        print(f"[ImageCaptioner] Captioning {path.name}...")
        caption = self._caption_pil(img)
        print(f"  → {caption[:120]}{'...' if len(caption) > 120 else ''}")

        return [Document(
            text=caption,
            source=str(path),
            modality="image",
            metadata={
                "image_file": path.name,
                "_pil_image": img,
                "caption_model": self.model_id,
            },
        )]

    def caption_pil_list(
        self, pil_images: list, sources: list[str]
    ) -> list[Document]:
        self._load()
        docs = []
        for img, src in zip(pil_images, sources):
            print(f"[ImageCaptioner] Captioning frame: {src}")
            caption = self._caption_pil_with_prompt(img, VIDEO_FRAME_PROMPT)
            docs.append(Document(
                text=caption,
                source=src,
                modality="image",
                metadata={
                    "_pil_image": img,
                    "caption_model": self.model_id,
                    "source_type": "keyframe_caption",
                },
            ))
        return docs

    def unload(self):
        print("[ImageCaptioner] Unloading Qwen2-VL...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[ImageCaptioner] VRAM freed.")