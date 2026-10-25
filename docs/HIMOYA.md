# Loyihani ustozga tushuntirish

**Mavzu:** Honda kompaniyasining foydalanish qo'llanmasi asosida terminal RAG tizimi.

## Masala

Foydalanuvchi 780 sahifali 2023 Honda Civic Sedan kitobidan kerakli ma'lumotni
tabiiy tilda so'raydi. Tizim mos parchalarni topadi va javobni manba sahifalari bilan beradi.

## RAG bosqichlari

1. **Retrieval:** savol va kitob bo'laklarining embeddinglari bilan tegishli matn qidiriladi.
2. **Augmentation:** topilgan matnlar va sahifa raqamlari savol bilan promptga qo'shiladi.
3. **Generation:** GPT-6 Luna ushbu matnlarga tayangan javob yaratadi.

| Qism | Tanlov | Sababi |
|---|---|---|
| Kitob | Honda Civic 2023 rasmiy PDF | Aniq kompaniya va modelga oid bilim bazasi |
| Chunking | 170 so'z, 35 so'z overlap | Matnni kichik, o'zaro tutash qismlarga bo'lish |
| Sahifa chegarasi | Bir chunk bitta sahifadan | Manba havolasini saqlash |
| Embedding | OpenRouter `qwen/qwen3-embedding-8b` | Ko'p tilli semantik qidiruv |
| Qidiruv | Cosine similarity + BM25 | Ma'no va aniq texnik atamalar |
| Indeks | NumPy + JSON, lokal diskda | Bitta kitob uchun sodda saqlash |
| Javob modeli | OpenRouter `openai/gpt-6-luna` | Hujjat kontekstidan javob yaratish |
| Natija | JSON + manba IDlarini tekshirish | Javob va sahifa havolalarini bog'lash |

Embedding modeli matnni vektorga aylantiradi. Chat modeli vektor yaratmaydi,
topilgan matnni o'qib javob tuzadi. Bu loyihada model Honda kitobiga fine-tuning qilinmagan.
Savolga yangi bilim kitob indeksidan beriladi.

## Namoyish

Avval `.env` ga kalit qo'yib `python main.py setup` bajaring. Keyin:

```powershell
python main.py info
python main.py search "recommended engine oil"
python main.py
```

Savollar:

```text
Qanday motor moyi tavsiya etiladi?
Benzin bakining sig'imi qancha?
Shinalar bosimini qachon tekshirish kerak?
What is the exact retail price of a 2030 Honda Civic in Uzbekistan?
```

Oxirgi savol kitob doirasidan tashqarida. Tizim bunday fakt topilmasa javob bermaslikka
yo'naltirilgan. `/sources` bilan modelga berilgan matnni ko'rsatish mumkin.

## Baholash va cheklov

Unit testlar API chegaralari va manba tekshiruvini kalitsiz sinaydi.
`python evaluate.py --generate` haqiqiy API bilan 12 ta retrieval va 4 ta generation
namunasini tekshiradi; bu buyruq kalit va kredit talab qiladi. Hit@5 kutilgan sahifa va
fakt dastlabki beshta natijada borligini ko'rsatadi, barcha jumlalarning aniqligini o'lchamaydi.

Havola tekshiruvi har bir gapning to'g'riligini kafolatlamaydi. PDF diagrammalari va
murakkab jadvallardagi munosabatlarni matnli ekstraksiya to'liq saqlamasligi mumkin.
Natija aynan shu yil va qo'llanmadagi variantlarga tegishli.
