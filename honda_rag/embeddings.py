"""OpenRouter embeddings: batching, javob tartibi nazorati va L2 normalization."""
import numpy as np

from . import config as c
from .openrouter import OpenRouterClient

QUERY_INSTRUCTION = "Given a question about a Honda vehicle manual, retrieve relevant passages that answer the question"


def input_text(text, query, model):
    if model.startswith("qwen/qwen3-embedding") and query:
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"
    if model.startswith("intfloat/") and "e5" in model:
        return ("query: " if query else "passage: ") + text
    return text


class Embedder:
    def __init__(self):
        self.client = OpenRouterClient()
        self.model = c.EMBEDDING_MODEL
        self.dimensions = c.EMBEDDING_DIMENSIONS or None

    def encode(self, texts, query=False, batch_size=None, progress=False):
        batch_size = batch_size or c.EMBEDDING_BATCH_SIZE
        if batch_size < 1:
            raise ValueError("Embedding batch_size musbat bo'lishi kerak.")
        result = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if any(not isinstance(text, str) or not text.strip() for text in batch):
                raise ValueError("Embedding uchun bo'sh matn yuborilmaydi.")
            payload = {"model": self.model, "input": [input_text(x, query, self.model) for x in batch],
                       "encoding_format": "float"}
            if c.EMBEDDING_DIMENSIONS:
                payload["dimensions"] = c.EMBEDDING_DIMENSIONS
            data = self.client.post("embeddings", payload).get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise RuntimeError("OpenRouter to'liq embedding natijasini qaytarmadi.")
            indices = [item.get("index") for item in data if isinstance(item, dict)]
            if (len(indices) != len(batch) or any(type(i) is not int for i in indices)
                    or sorted(indices) != list(range(len(batch)))):
                raise RuntimeError("Embedding javobidagi index qiymatlari noto'g'ri yoki takrorlangan.")
            try:
                vectors = np.asarray([x["embedding"] for x in sorted(data, key=lambda x: x["index"])], dtype=np.float32)
            except (KeyError, ValueError, TypeError) as exc:
                raise RuntimeError("OpenRouter embedding vektori noto'g'ri shaklda.") from exc
            if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise RuntimeError("Embeddingda bo'sh, NaN yoki cheksiz qiymat mavjud.")
            if self.dimensions is not None and vectors.shape[1] != self.dimensions:
                raise RuntimeError("Embedding o'lchami sozlama yoki oldingi batch bilan mos kelmadi.")
            self.dimensions = vectors.shape[1]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                raise RuntimeError("OpenRouter nol embedding vektorini qaytardi.")
            result.append(vectors / norms)
            if progress:
                print(f"  OpenRouter embedding: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(result) if result else np.empty((0, self.dimensions or 0), dtype=np.float32)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
