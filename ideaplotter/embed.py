"""Embed documents with a sentence-transformers model (GPU + fp16 when available)."""
import hashlib
import json
import threading

import numpy as np

from . import config


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, max_tokens: int = config.EMBED_MAX_TOKENS):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from .space import compute_dtype

        # not the checkpoint's bfloat16, which most CPUs and older GPUs run slowly
        model_kwargs = {"dtype": compute_dtype(self.device)}
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_tokens
        self.name = model_name
        self._lock = threading.Lock()  # encode() is called from worker threads

    def encode(self, texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
        with self._lock:
            vecs = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=progress,
            )
        return vecs.astype(np.float32)


def embed_corpus(df, model_name: str = config.EMBED_MODEL, batch_size: int = 32, chunk: int = 8192) -> np.ndarray:
    """Embed every game's doc into data/space/emb.npy (float16, row-aligned with games.parquet).

    Work is saved in chunks under data/space/emb_parts/, so an interrupted run resumes.
    """
    config.SPACE.mkdir(parents=True, exist_ok=True)
    meta_path, emb_path = config.SPACE / "emb_meta.json", config.SPACE / "emb.npy"
    appids = [int(a) for a in df["appid"]]
    if emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("appids") == appids and meta.get("model") == model_name:
            print("embeddings: up to date")
            return np.load(emb_path).astype(np.float32)

    parts_dir = config.SPACE / "emb_parts"
    parts_dir.mkdir(exist_ok=True)
    docs = df["doc"].tolist()
    embedder, parts = None, []
    n_chunks = (len(docs) + chunk - 1) // chunk
    for ci, s in enumerate(range(0, len(docs), chunk)):
        key = hashlib.sha1((model_name + "\n" + "\n".join(docs[s : s + chunk])).encode("utf-8")).hexdigest()[:12]
        path = parts_dir / f"part_{ci:04d}_{key}.npy"
        if path.exists():
            parts.append(np.load(path))
            continue
        if embedder is None:
            embedder = Embedder(model_name)
            print(f"embeddings: encoding {len(docs):,} docs with {model_name} on {embedder.device}")
        parts.append(embedder.encode(docs[s : s + chunk], batch_size=batch_size).astype(np.float16))
        np.save(path, parts[-1])
        print(f"  chunk {ci + 1}/{n_chunks} done")
    E = np.concatenate(parts)
    np.save(emb_path, E)
    meta_path.write_text(json.dumps({"model": model_name, "dim": int(E.shape[1]), "appids": appids}))
    for p in parts_dir.glob("part_*.npy"):
        p.unlink()
    return E.astype(np.float32)
