"""Honda PDFni olish va OpenRouter orqali indeks yaratish."""
import os
import subprocess

import requests

from . import config as c


def download_pdf():
    if c.PDF.exists():
        with c.PDF.open("rb") as existing:
            if existing.read(5) == b"%PDF-":
                return
    c.DATA.mkdir(parents=True, exist_ok=True)
    temp = c.PDF.with_suffix(".download")
    try:
        with requests.get(c.PDF_URL, timeout=(15, 180), stream=True) as response:
            response.raise_for_status()
            with temp.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
    except requests.exceptions.SSLError:
        if os.name != "nt":
            raise
        # Windows curl tizimning sertifikat omboridan foydalanadi, TLS saqlanadi.
        subprocess.run(["curl.exe", "-L", "--fail", "--max-time", "180",
                        "-o", str(temp), c.PDF_URL], check=True)
    with temp.open("rb") as downloaded:
        if downloaded.read(5) != b"%PDF-":
            raise ValueError("Yuklangan hujjat PDF emas.")
    temp.replace(c.PDF)


def check_models():
    """Ommaviy katalog tekshiruvi; pullik inference so'rovi yuborilmaydi."""
    catalogs = [("models", c.CHAT_MODEL, "Javob"),
                ("embeddings/models", c.EMBEDDING_MODEL, "Embedding")]
    for endpoint, model, label in catalogs:
        try:
            response = requests.get(f"{c.OPENROUTER_BASE_URL}/{endpoint}", timeout=(15, 30))
            response.raise_for_status()
            entries = response.json()["data"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise RuntimeError("OpenRouter modellar katalogini tekshirib bo'lmadi.") from exc
        if not any(item.get("id") == model for item in entries):
            raise RuntimeError(f"{label} modeli OpenRouter katalogida topilmadi: {model}. .env ni tekshiring.")
        print(f"{label} modeli katalogda mavjud: {model}", flush=True)
    print("Bu tekshiruv API kalitining huquqi yoki balansini tekshirmaydi.")


def setup():
    c.require_api_key()
    print("1/3 OpenRouter kaliti va model nomlarini tekshirish...", flush=True)
    from .openrouter import OpenRouterClient
    client = OpenRouterClient()
    try:
        client.validate_api_key()
    finally:
        client.close()
    check_models()
    print("2/3 Honda rasmiy qo'llanmasi...", flush=True)
    download_pdf()
    print("3/3 OpenRouter embeddings bilan indeks yaratish...", flush=True)
    from .retrieval import build_index
    build_index()
