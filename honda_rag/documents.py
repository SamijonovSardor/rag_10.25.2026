"""PDF -> sahifalar -> bir sahifa ichidagi ustma-ust matn bo'laklari."""
import re
from collections import Counter

from pypdf import PdfReader

from . import config as c


def clean_text(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"\d{1,4}", line):
            continue
        if re.match(r"\d{2} .*OM-.*\.book", line) or "Page " in line and "Friday," in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def printed_page(text):
    # Honda matnidagi alohida satrdagi raqam; PDF sahifasi alohida saqlanadi.
    numbers = re.findall(r"(?m)^\s*(\d{1,4})\s*$", text)
    return int(numbers[0]) if numbers else None


def chunk_page(text, pdf_page, book_page=None, words=c.CHUNK_WORDS, overlap=c.CHUNK_OVERLAP):
    if words <= overlap or overlap < 0:
        raise ValueError("Chunk hajmi overlapdan katta bo'lishi kerak.")
    tokens = text.split()
    chunks = []
    for start in range(0, len(tokens), words - overlap):
        piece = tokens[start:start + words]
        if len(piece) < 12:
            break
        chunks.append({"id": f"p{pdf_page:04d}-{start:04d}", "pdf_page": pdf_page,
                       "book_page": book_page, "text": " ".join(piece),
                       "source": c.PDF.name, "title": c.TITLE})
        if start + words >= len(tokens):
            break
    return chunks


def extract_document(path=c.PDF):
    reader = PdfReader(path)
    chunks = []
    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        number = printed_page(raw)
        pages.append({"pdf_page": i + 1, "book_page": number, "text": text})
        # Kitob oxiridagi alfabetik indeks javob mazmunini bermaydi.
        if i > len(reader.pages) * .95 and text.startswith("Index"):
            continue
        chunks.extend(chunk_page(text, i + 1, number))
        if (i + 1) % 100 == 0:
            print(f"  PDF: {i + 1}/{len(reader.pages)} sahifa", flush=True)
    if not chunks:
        raise ValueError("PDFda o'qiladigan matn topilmadi. Skanerlangan PDF uchun OCR kerak.")
    # Alohida jadval raqami sahifa raqami deb olinmasin. Kitobdagi umumiy
    # PDF/print offsetiga zid bo'lgan kam sonli nomzodlar ko'rsatilmaydi.
    offsets = Counter(x["pdf_page"] - x["book_page"] for x in pages if x["book_page"] is not None)
    if offsets:
        common, count = offsets.most_common(1)[0]
        if count / sum(offsets.values()) > .9:
            for item in pages + chunks:
                if item["book_page"] is not None and item["pdf_page"] - item["book_page"] != common:
                    item["book_page"] = None
    return chunks, pages
