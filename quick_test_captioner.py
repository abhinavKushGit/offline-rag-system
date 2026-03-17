import sys
sys.path.append(".")
from src.ingestion.image_captioner import ImageCaptioner

c = ImageCaptioner()
docs = c.caption_file("data/images/testimage2.png")
for d in docs:
    print(d.text)
c.unload()
