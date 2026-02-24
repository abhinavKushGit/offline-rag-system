import hashlib
import json
from pathlib import Path


def compute_dir_hash(directory: str) -> str:
    """
    Stable hash of a directory based on filenames + sizes + mtimes.
    Same files = same hash. Any change = different hash.
    """
    dir_path = Path(directory)
    entries = []

    for f in sorted(dir_path.rglob("*")):
        if f.is_file():
            stat = f.stat()
            entries.append({
                "path": str(f.relative_to(dir_path)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    hash_input = json.dumps(entries, sort_keys=True).encode()
    return hashlib.md5(hash_input).hexdigest()


def get_cache_paths(cache_dir: str, source_hash: str):
    """
    Returns (index_path, meta_path) for a given source hash.
    Creates the directory if it doesn't exist.
    """
    cache_path = Path(cache_dir) / source_hash
    cache_path.mkdir(parents=True, exist_ok=True)
    return (
        str(cache_path / "index.faiss"),
        str(cache_path / "metadata.pkl"),
    )


def cache_exists(cache_dir: str, source_hash: str) -> bool:
    index_path, meta_path = get_cache_paths(cache_dir, source_hash)
    return Path(index_path).exists() and Path(meta_path).exists()