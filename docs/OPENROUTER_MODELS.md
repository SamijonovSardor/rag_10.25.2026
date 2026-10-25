# OpenRouter modellarini tanlash

2026-10-25 kuni OpenRouter'ning ommaviy `/models` va `/embeddings/models`
kataloglaridan model IDlari tekshirildi. Foydalanuvchi so'ragan chat modeli:
**`openai/gpt-6-luna`**. So'rovda `reasoning.effort=none` va JSON Schema structured output ishlatiladi.

Embedding variantlari (ko'rsatilgan narxlar 1 million input token uchun, tekshirilgan paytdagi katalog narxi):

| Model | Narx | Ushbu loyiha uchun baholash |
|---|---|---|
| `qwen/qwen3-embedding-8b` | $0.01 | Standart tanlov: ko'p tilli retrieval, query instruction, 4096 gacha o'lcham |
| `openai/text-embedding-3-small` | $0.02 | Oddiy va arzon muqobil, `.env` orqali tanlash mumkin |
| `openai/text-embedding-3-large` | $0.13 | Inglizcha va boshqa tillardagi embedding uchun muqobil |
| `google/gemini-embedding-001` | $0.15 | Ko'p tilli umumiy embedding uchun muqobil |

Qwen3-Embedding-8B model kartasi 100 dan ortiq til, query instruction va 32..4096
chiqish o'lchamlarini ko'rsatadi. Ushbu xususiyatlar va katalog narxi sababli u
o'zbekcha savol bilan inglizcha kitobdan qidirish uchun ishchi tanlov qilindi.
Bu model Honda to'plamida boshqa modellardan yaxshiroq chiqdi degan o'lchov emas:
haqiqiy API kaliti qo'yilgach retrieval sifatini `evaluate.py` orqali o'lchash kerak.

`OPENROUTER_EMBEDDING_DIMENSIONS=0` bo'lsa dimensions parametri yuborilmaydi;
API qaytargan haqiqiy o'lcham indeksda saqlanadi. Aniq o'lcham so'ralsa, model va
provider uni qo'llashi kerak. Noto'g'ri yoki bir-biriga mos kelmagan o'lchamlar rad etiladi.

Model yoki o'lcham o'zgarsa indeksni `python main.py ingest --force` bilan yangilang.
Chat modeli o'zgarsa embeddinglarni qayta yaratish shart emas.

Manbalar:

- https://openrouter.ai/openai/gpt-6-luna
- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://openrouter.ai/qwen/qwen3-embedding-8b
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
- https://openrouter.ai/openai/text-embedding-3-small
- https://openrouter.ai/openai/text-embedding-3-large
- https://openrouter.ai/google/gemini-embedding-001
- https://openrouter.ai/docs/api_reference/embeddings
- https://openrouter.ai/docs/guides/features/structured-outputs
