# debug_captions.py
import pickle, sys
sys.path.append(".")

with open("outputs/indexes/3a489cc1ce7612b2ddfdbbd3ceeea818/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

for i, m in enumerate(metadata):
    if "video" in m.get("modality", "") or "frame" in m.get("source", ""):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {m.get('source')}")
        print(f"Text: {m.get('text', '')[:300]}")