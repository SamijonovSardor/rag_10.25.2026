# OpenRouter versiyasi tekshiruvi

Sana: 2026-10-25. Muhit: Windows x64, Python 3.13.5.

| Tekshiruv | Holat |
|---|---|
| `openai/gpt-6-luna` ommaviy OpenRouter katalogida | Tasdiqlandi |
| `qwen/qwen3-embedding-8b` embeddings katalogida | Tasdiqlandi |
| Kalitsiz unit/integratsiya testlari | 29 / 29 o'tdi; HTTP javoblari taqlid qilingan |
| PDF matni | Mavjud 780 sahifali kitob, 1 194 bo'lak; qayta foydalanish uchun matn keshi |
| Haqiqiy OpenRouter embedding indeksi | Lokal indeksda Qwen3 Embedding 8B, 4096 o'lcham, 1 194 bo'lak mavjud |
| GPT-6 Luna avtomatik API baholashi | To'liq hisobot uchun `python evaluate.py --generate` bajariladi |

Testlar quyidagilarni qamrab oladi:

- Kalit bo'sh bo'lsa HTTP so'rovi yubormaslik.
- Management/Provisioning kalitini bepul `/key` endpointi bilan aniqlash va aniq xato qaytarish.
- Bearer autentifikatsiyasi, chat va embedding endpointlari.
- 401 xatosida maxfiy kalitni chiqarmaslik, 429 da qisqa qayta urinish.
- Timeoutdan keyin takroriy pullik so'rovni avtomatik yubormaslik.
- Tartibi almashgan embedding javoblarini index bo'yicha to'g'ri tartiblash.
- NaN, nol vektor, yetishmayotgan natija va turli o'lchamlarni rad etish.
- Faqat Qwen querylari uchun instruction qo'shish.
- GPT-6 Luna model IDsi va strict JSON Schema so'rov shartnomasi.
- Kesilgan yoki yaroqsiz chat javobini manbali javob sifatida ko'rsatmaslik.
- Matn keshi -> yangi vektorlar -> retrieval -> chat -> haqiqiy sahifa IDsi oqimi.
- Embedding modeli/o'lchami o'zgarsa indeksni rad etish; chat modeli o'zgarsa qayta embedding qilmaslik.
- O'zgarmagan indeksni API chaqirmasdan qayta ishlatish.
- Noto'g'ri manba raqamlari va sahifa chegaralari.

Haqiqiy API bilan baholash uchun `.env` ga kalit yozing:

```powershell
python main.py setup
python evaluate.py --generate
```

Natija `openrouter-test-results.json` ga tanlangan modellarning nomlari bilan yoziladi.
Oldingi lokal E5/Qwen sinovlari OpenRouter modeli sifati haqidagi o'lchov hisoblanmaydi.
GitHub Actions unit testlarni Python 3.10 va 3.13 da ishga tushiradi; unda pullik API so'rovlari yuborilmaydi.
