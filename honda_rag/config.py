"""OpenRouter sozlamalari. Shell o'zgaruvchilari .env dan ustun."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env", override=False)
INDEX = ROOT / "index" / "openrouter"
DOCUMENT_CACHE = DATA / "cache"
PDF = DATA / "honda_civic_2023_manual.pdf"
PDF_URL = "https://techinfo.honda.com/rjanisis/pubs/OM/AH/AT202323OMEN/enu/AT202323OMEN.PDF"
TITLE = "2023 Honda Civic Sedan Owner's Manual"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-6-luna").strip()
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b").strip()


def env_integer(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} butun son bo'lishi kerak.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} {minimum}..{maximum} oralig'ida bo'lishi kerak.")
    return value


EMBEDDING_DIMENSIONS = env_integer("OPENROUTER_EMBEDDING_DIMENSIONS", 0, 0, 65536)
EMBEDDING_BATCH_SIZE = env_integer("OPENROUTER_EMBEDDING_BATCH_SIZE", 32, 1, 256)
REQUEST_TIMEOUT = env_integer("OPENROUTER_TIMEOUT_SECONDS", 120, 1, 600)


def require_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key.lower() in ("your_api_key", "your_openrouter_api_key", "sk-or-..."):
        raise RuntimeError("OpenRouter API kaliti qo'yilmagan. .env faylidagi "
                           "OPENROUTER_API_KEY= qatoriga o'z kalitingizni yozing.")
    return key


CHUNK_WORDS = 170
CHUNK_OVERLAP = 35
INDEX_VERSION = 2
