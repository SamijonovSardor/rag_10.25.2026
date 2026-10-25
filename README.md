# Honda RAG — OpenRouter bilan terminal loyihasi

Manba: [2023 Honda Civic Sedan Owner's Manual](https://techinfo.honda.com/rjanisis/pubs/OM/AH/AT202323OMEN/enu/AT202323OMEN.PDF),
780 sahifali rasmiy Honda qo'llanmasi. Savol terminalda beriladi, javob manba sahifalari bilan chiqadi.

## Boshlash

Repository papkasida PowerShell orqali o'rnating:

```powershell
.\install.bat
```

O'rnatuvchi virtual muhit va `.env.example` asosida `.env` faylini yaratadi.
Mavjud `.env` saqlanadi. [OpenRouter API Keys](https://openrouter.ai/settings/keys) sahifasidan
**oddiy API kaliti** yarating va `.env` ga yozing. Management/Provisioning kaliti
model so'rovlari uchun ishlamaydi.

```dotenv
OPENROUTER_API_KEY=bu_yerga_haqiqiy_kalitingiz
OPENROUTER_CHAT_MODEL=openai/gpt-6-luna
OPENROUTER_EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

So'ng:

```powershell
.\start.bat auth
.\start.bat setup
.\start.bat
```

`setup` PDF matnini OpenRouter orqali embedding qilib, `index/openrouter/` ichiga indeks saqlaydi.
Birinchi indekslashdan keyin faqat savol embeddingi va javob yaratish uchun API chaqiriladi.
O'zgarmagan indeks `setup` qayta bajarilganda qayta embedding qilinmaydi.

Internet va OpenRouter hisobida yetarli kredit kerak. So'rovlar OpenRouter tarifiga ko'ra hisoblanadi.
PDF bo'laklari indekslashda, savol va tegishli matnlar esa javob vaqtida OpenRouter'ga yuboriladi.
GPU va katta model yuklash talab qilinmaydi.

## Boshqa kompyuterda

Python 3.10+ kerak. Git repositoryga kod, testlar va `.env.example` kiritiladi.
Kalitlar, PDF, uning matn keshi va embedding indeksi lokal saqlanadi.
`setup` kitobni rasmiy Honda manbasidan yuklab, indeksni yaratadi.

Windowsda qo'lda o'rnatish ham mumkin:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` ni to'ldirgach `start.bat setup`, so'ng `start.bat` ishlating.
Quyidagi `python main.py ...` namunalarini virtual muhit Python'i bilan bajarish uchun
`python` o'rniga `.\.venv\Scripts\python.exe` yozing yoki `start.bat` orqali buyruq bering.

## Modellar

| Vazifa | Standart model | Endpoint |
|---|---|---|
| Javob yaratish | `openai/gpt-6-luna` | OpenRouter `/chat/completions` |
| Vektor yaratish | `qwen/qwen3-embedding-8b` | OpenRouter `/embeddings` |

Embedding tanlovi va muqobillar: [docs/OPENROUTER_MODELS.md](docs/OPENROUTER_MODELS.md).

`.env` sozlamalari:

| Sozlama | Qiymati / vazifasi |
|---|---|
| `OPENROUTER_API_KEY` | O'zingiz qo'yadigan maxfiy kalit |
| `OPENROUTER_CHAT_MODEL` | `openai/gpt-6-luna` |
| `OPENROUTER_EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` |
| `OPENROUTER_EMBEDDING_DIMENSIONS` | `0`: modelning standart o'lchami; istalsa model qo'llaydigan aniq o'lcham |
| `OPENROUTER_EMBEDDING_BATCH_SIZE` | `32`: bitta embedding so'rovidagi bo'laklar soni |
| `OPENROUTER_TIMEOUT_SECONDS` | `120`: API javobini kutish chegarasi |

Shell muhitidagi qiymatlar `.env` dan ustun. Sozlamani almashtirgach dasturni qayta oching.
Embedding modeli yoki o'lchamini o'zgartirsangiz **`python main.py ingest --force`** bajaring.
Faqat chat modelini o'zgartirish kitobni qayta embedding qilishni talab qilmaydi.
API kaliti `.gitignore` bilan Gitdan chiqarilgan va xato xabarlariga yozilmaydi.

## Buyruqlar va namunalar

```powershell
python main.py info
python main.py auth
python main.py check
python main.py ask "Qanday motor moyi tavsiya etiladi?"
python main.py ask "Benzin bakining sig'imi qancha?" --show-context
python main.py ask "What is TPMS?" --language en --json
python main.py search "How often should tire pressure be checked?"
python main.py ingest --force
```

`check` ommaviy katalogdan ikkala model IDsi mavjudligini tekshiradi; inference xarajati yo'q.
U hisob kalitining haqiqiyligi, balans yoki modelga kirish ruxsatini tasdiqlamaydi.
`info` API kalitining o'zini chiqarmaydi, faqat qo'yilgan/qo'yilmaganini ko'rsatadi.
`auth` bepul `/key` endpointi orqali kalit turini tekshiradi. `setup` ham bu tekshiruvni
embedding yaratishdan oldin bajaradi. Kalit almashtirilgach dasturni qayta ishga tushiring.

Suhbat buyruqlari: `/help`, `/sources`, `/language uz`, `/language en`, `/exit`.
Har savol mustaqil: oldingi javobga ishora qilish o'rniga to'liq savol yozing.

## RAG oqimi

```text
Honda PDF -> pypdf -> 170 so'zli bo'laklar (35 so'z overlap)
          -> OpenRouter Qwen3 Embedding -> NumPy vektor indeksi + JSON manbalar

Savol -> OpenRouter embedding -> cosine similarity + BM25
      -> 5 ta turli sahifadan kontekst -> OpenRouter GPT-6 Luna
      -> JSON javob va manba IDlarini tekshirish -> javob + PDF sahifalari
```

Embeddingga asl savol yuboriladi. Qwen3 uchun query instruction qo'shiladi;
kitob bo'laklariga shu query instruction qo'shilmaydi. O'zbekcha texnik atamalar
lug'ati faqat BM25 kalit so'zlarini to'ldiradi. Model javoblari lug'atda saqlanmaydi.
Embeddinglar L2 bo'yicha normallanadi; cosine similarity normallangan vektorlarning
skalyar ko'paytmasidir. Yakuniy qidiruv bahosi `cosine + 0.10 * normalized_BM25`.

Bir sahifa takrorlanmaydi. Topilgan bo'lak atrofidagi matn ham olinadi (taxminan 320 so'zgacha),
shunda sarlavha va jadval izohlari saqlanadi. Javob `supported`, `answer`, `sources` JSON sxemasida
olinadi. Mavjud bo'lmagan manba raqamiga havola qilgan javob qabul qilinmaydi.
Bu havola mavjudligini tekshiradi; har bir jumlaning semantik to'g'riligini isbotlamaydi.

PDF SHA256, embedding model nomi, o'lcham sozlamasi va chunking parametrlari indeksga yoziladi.
Vektorlar aynan tanlangan OpenRouter embedding modeli bilan yaratiladi. Mavjud PDF matn keshi
mos kelsa, undan matn qayta olinadi. Model yoki o'lcham o'zgarganda yangi embeddinglar kerak bo'ladi.

## Fayllar

| Fayl | Vazifasi |
|---|---|
| `main.py` | Terminal buyruqlari va suhbat |
| `honda_rag/config.py` | `.env` va model sozlamalari |
| `honda_rag/openrouter.py` | Auth, HTTP, timeout va qayta urinishlar |
| `honda_rag/embeddings.py` | Embedding API, batching, vektorlarni tekshirish |
| `honda_rag/llm.py` | GPT-6 Luna prompti va structured output |
| `honda_rag/documents.py` | PDF parsing va sahifali chunking |
| `honda_rag/retrieval.py` | Indeks, cosine similarity va BM25 |
| `honda_rag/pipeline.py` | RAG oqimi va manba tekshiruvi |
| `data/SOURCE.md` | Kitob manbasi va huquqlari |
| `docs/HIMOYA.md` | Ustozga tushuntirish |
| `docs/TEKSHIRUV.md` | Qaysi sinovlar bajarilgani va cheklovlar |
| `.github/workflows/tests.yml` | Python 3.10 va 3.13 uchun GitHub Actions testlari |

## Tekshirish

```powershell
python -m unittest discover -s tests -v
python evaluate.py
python evaluate.py --generate
```

Unit testlar API so'rovlarini taqlid qiladi va haqiqiy kalitsiz ishlaydi.
GitHub Actions push va pull requestlarda shu testlarni ishga tushiradi; CI uchun API kaliti kerak emas.
`evaluate.py` haqiqiy API chaqiradi: avval kalit qo'yib indeks yaratish kerak.
`--generate` 4 ta javob namunasi uchun GPT-6 Luna'ni ham chaqiradi. Natija model nomlari bilan
`openrouter-test-results.json` fayliga yoziladi. Bu kichik development set, mustaqil aniqlik benchmarki emas.

## Muammolar

- Kalit bo'sh: `.env` dagi `OPENROUTER_API_KEY` ni to'ldiring.
- Management/Provisioning kaliti: oddiy API kaliti yarating, `.env` da almashtiring va `python main.py auth` bajaring.
- HTTP 401: kalitni; 402: balansni; 403: hisob/model ruxsatini tekshiring.
- HTTP 429: limit tugagan; dastur qisqa qayta urinishdan keyin tushunarli xabar chiqaradi.
- Indeks yo'q yoki model o'zgargan: `python main.py setup` yoki `ingest --force`.
- Diagramma va murakkab jadvallarning bog'lanishlari matn ekstraksiyasida yo'qolishi mumkin.
  Model bunday variantlarni taxmin qilmaslikka yo'naltirilgan. Aniq faktni manba sahifasida tekshiring.

Kitobning mualliflik huquqi Honda Motor Co., Ltd.ga tegishli. Loyihada tanlangan kitob uchun fine-tuning qilinmaydi;
yangi bilim hujjat indeksidan olinadi.
