import json
import hashlib
from pathlib import Path
from src.ingestion.image_captioner import ImageCaptioner
from src.schema import Document
from PIL import Image


class VideoCaptioner:
    def __init__(self, cache_dir: str = "outputs/indexes/video_captions"):
        self.captioner = ImageCaptioner()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _sources_hash(self, sources: list[str]) -> str:
        key = "|".join(sorted(sources))
        return hashlib.md5(key.encode()).hexdigest()

    def caption_frames(
        self,
        pil_images: list,
        sources: list[str],
    ) -> list[Document]:
        cache_hash = self._sources_hash(sources)
        cache_path = self.cache_dir / f"{cache_hash}.json"

        # cache hit — skip Qwen2-VL entirely
        if cache_path.exists():
            print(f"[VideoCaptioner] Cache hit — loading captions from disk.")
            with open(cache_path) as f:
                cached = json.load(f)
            docs = []
            for entry, img in zip(cached, pil_images):
                docs.append(Document(
                    text=entry["text"],
                    source=entry["source"],
                    modality="video",
                    metadata={
                        "_pil_image": img,
                        "caption_model": entry["caption_model"],
                        "source_type": "keyframe_caption",
                    },
                ))
            return docs

        # cache miss — run Qwen2-VL
        raw_docs = self.captioner.caption_pil_list(pil_images, sources)

        docs = []
        cache_data = []
        for doc, img in zip(raw_docs, pil_images):
            src = doc.source
            if "::" in src:
                video_file = src.split("::")[0]
                frame_part = src.split("::")[1]
                timestamp = frame_part.replace("frame_", "").replace("s", "")
                video_name = Path(video_file).stem
                prefix = f"Video file '{video_name}' at {timestamp} seconds: "
            else:
                prefix = "Video frame: "

            enriched_text = prefix + doc.text

            docs.append(Document(
                text=enriched_text,
                source=doc.source,
                modality="video",
                metadata={
                    "_pil_image": img,
                    "caption_model": doc.metadata.get("caption_model", ""),
                    "source_type": "keyframe_caption",
                },
            ))
            cache_data.append({
                "text": enriched_text,
                "source": doc.source,
                "caption_model": doc.metadata.get("caption_model", ""),
            })

        # save to cache
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"[VideoCaptioner] Captions cached → {cache_path}")

        return docs

    def unload(self):
        self.captioner.unload()
