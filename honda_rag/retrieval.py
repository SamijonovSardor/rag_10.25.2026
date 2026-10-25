"""Doimiy NumPy vektor indeksi + BM25 orqali gibrid qidiruv."""
from collections import Counter
import hashlib
import json
import re

import numpy as np

from . import config as c
from .embeddings import Embedder
from .query import expand_query


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature():
    return {"version": c.INDEX_VERSION, "pdf_sha256": fingerprint(c.PDF),
            "embedding_provider": "openrouter", "embedding_model": c.EMBEDDING_MODEL,
            "embedding_dimensions_requested": c.EMBEDDING_DIMENSIONS, "input_format_version": 1,
            "chunk_words": c.CHUNK_WORDS, "chunk_overlap": c.CHUNK_OVERLAP}


def build_index(force=False):
    if not c.PDF.exists():
        raise RuntimeError("Honda PDF topilmadi. Avval: python main.py setup")
    expected = signature()
    meta_path = c.INDEX / "metadata.json"
    if not force and all((c.INDEX / x).exists() for x in ("metadata.json", "chunks.json", "vectors.npy", "pages.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(metadata.get(k) == v for k, v in expected.items()):
            print(f"Indeks tayyor: {metadata['pages']} sahifa, {metadata['chunks']} bo'lak.")
            return metadata
    # Kalit xatosi PDFni yuzlab sahifa tahlil qilishdan oldin ko'rsatiladi.
    c.require_api_key()
    from .documents import extract_document
    chunks = pages = None
    try:
        cached = json.loads((c.DOCUMENT_CACHE / "metadata.json").read_text(encoding="utf-8"))
        if all(cached.get(k) == expected[k] for k in ("pdf_sha256", "chunk_words", "chunk_overlap")):
            chunks = json.loads((c.DOCUMENT_CACHE / "chunks.json").read_text(encoding="utf-8"))
            pages = json.loads((c.DOCUMENT_CACHE / "pages.json").read_text(encoding="utf-8"))
            print(f"Tayyor PDF matni ishlatilmoqda: {len(pages)} sahifa.", flush=True)
    except (OSError, ValueError, KeyError):
        pass
    if not chunks or not pages:
        chunks, pages = extract_document()
        c.DOCUMENT_CACHE.mkdir(parents=True, exist_ok=True)
        cache_metadata = {key: expected[key] for key in ("pdf_sha256", "chunk_words", "chunk_overlap")}
        for name, value in (("chunks.json", chunks), ("pages.json", pages), ("metadata.json", cache_metadata)):
            (c.DOCUMENT_CACHE / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    with Embedder() as embedder:
        vectors = embedder.encode([x["text"] for x in chunks], progress=True)
    c.INDEX.mkdir(parents=True, exist_ok=True)
    # Metadata oxirida yoziladi: yarim yaratilgan indeks tayyor deb olinmaydi.
    if meta_path.exists():
        meta_path.unlink()
    np.save(c.INDEX / "vectors.npy", vectors, allow_pickle=False)
    (c.INDEX / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (c.INDEX / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    metadata = expected | {"pages": len(pages), "chunks": len(chunks),
                           "dimensions": int(vectors.shape[1]), "source_url": c.PDF_URL, "title": c.TITLE}
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indeks saqlandi: {len(pages)} sahifa, {len(chunks)} bo'lak, {vectors.shape[1]} o'lcham.")
    return metadata


STOP = set("the a an of for to in is are do does how what which i my it and with can should "
           "honda civic 2023 nima qanday qancha qaysi uchun kerak haqida bilan bor".split())


def tokenize(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP and len(word) > 1]


class Retriever:
    def __init__(self):
        try:
            self.metadata = json.loads((c.INDEX / "metadata.json").read_text(encoding="utf-8"))
            self.chunks = json.loads((c.INDEX / "chunks.json").read_text(encoding="utf-8"))
            self.pages = {x["pdf_page"]: x for x in json.loads((c.INDEX / "pages.json").read_text(encoding="utf-8"))}
            self.vectors = np.load(c.INDEX / "vectors.npy", allow_pickle=False)
        except FileNotFoundError as exc:
            raise RuntimeError("Indeks yo'q. Avval: python main.py setup") from exc
        if any(self.metadata.get(k) != v for k, v in signature().items()):
            raise RuntimeError("PDF yoki indeks sozlamasi o'zgargan. Bajaring: python main.py ingest --force")
        if (self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks)
                or self.vectors.shape[1] != self.metadata.get("dimensions")
                or not np.isfinite(self.vectors).all()):
            raise RuntimeError("Indeks buzilgan. Bajaring: python main.py ingest --force")
        self.embedder = Embedder()
        self.counts = [Counter(tokenize(x["text"])) for x in self.chunks]
        self.lengths = np.array([sum(x.values()) for x in self.counts], dtype=np.float32)
        self.avg_length = max(float(self.lengths.mean()), 1)
        self.df = Counter(word for doc in self.counts for word in doc)

    def bm25(self, question):
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for word in set(tokenize(question)):
            frequency = np.array([doc.get(word, 0) for doc in self.counts], dtype=np.float32)
            df = self.df.get(word, 0)
            idf = np.log(1 + (len(self.chunks) - df + .5) / (df + .5))
            scores += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * self.lengths / self.avg_length))
        return scores

    def search(self, question, top_k=5):
        if not question.strip() or len(question) > 2000:
            raise ValueError("Savol 1-2000 ta belgidan iborat bo'lishi kerak.")
        if not 1 <= top_k <= 8:
            raise ValueError("top-k 1 dan 8 gacha bo'lishi kerak.")
        expanded = expand_query(question)
        # Ko'p tilli embedding asl savolni ko'radi; lug'at faqat BM25 uchun.
        query = self.embedder.encode([question], query=True)[0]
        if query.shape[0] != self.vectors.shape[1]:
            raise RuntimeError("So'rov vektori indeks o'lchamiga mos emas. python main.py ingest --force bajaring.")
        semantic = self.vectors @ query
        lexical = self.bm25(expanded)
        scaled_lexical = lexical / max(float(lexical.max()), 1e-8)
        combined = semantic + .10 * scaled_lexical
        candidates = np.argsort(-combined)
        results, seen_pages = [], set()
        for i in candidates:
            chunk = self.chunks[int(i)]
            # Mundarija/alfabetik indeksdagi nuqtali ro'yxat dalil sifatida olinmaydi.
            if len(re.findall(r"\.{5,}", chunk["text"])) > 4:
                continue
            if chunk["pdf_page"] in seen_pages:
                continue
            seen_pages.add(chunk["pdf_page"])
            # Topilgan bo'lakning oldidagi sarlavha va jadval izohi yo'qolmasin.
            # 5 * 320 so'z odatiy kontekstni 4096 token chegarasida saqlaydi.
            page_words = self.pages[chunk["pdf_page"]]["text"].split()
            if len(page_words) <= 320:
                page_context = " ".join(page_words)
            else:
                offset = int(chunk["id"].split("-")[-1])
                if offset < 160:
                    page_context = " ".join(page_words[:320])
                else:
                    page_context = " ".join(page_words[:120]) + " [...] " + " ".join(page_words[max(120, offset - 30):offset + 170])
            results.append(chunk | {"reference": len(results) + 1,
                                   "matched_text": chunk["text"], "text": page_context,
                                   "semantic_score": round(float(semantic[i]), 4),
                                   "bm25_score": round(float(lexical[i]), 4),
                                   "score": round(float(combined[i]), 4), "search_query": question,
                                   "lexical_query": expanded})
            if len(results) == top_k:
                break
        return results

    def close(self):
        self.embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
