"""O'zbekcha avtomobil atamalarini inglizcha qidiruv atamalariga kengaytirish.

Bu lug'at javob/fakt saqlamaydi. Kitob inglizcha bo'lgani uchun qisqa texnik
so'zlar BM25 qidiruviga qo'shimcha kalit atamalar sifatida beriladi.
"""
import re

# So'z ildizlari ataylab ko'rsatilgan: shina/shinalar/shinaning bir mavzu.
TERMS = [
    (r"moy|yog'", "engine oil recommended viscosity"),
    (r"shina|g'ildirak|balon", "tires wheels"),
    (r"bosim", "pressure"),
    (r"benzin|yoqilg|yonilg", "fuel gasoline octane"),
    (r"bak|sig'im", "tank capacity"),
    (r"tormoz", "brake fluid braking"),
    (r"akkum|batarey", "battery"),
    (r"xavfsizlik kamari|kamar", "seat belt"),
    (r"havo yosti", "airbag"),
    (r"bolalar|bola o'rind|bolani", "child safety child seat"),
    (r"sovut|antifriz", "engine coolant"),
    (r"qizib|harorat", "engine temperature overheating"),
    (r"uzatma|transmiss|karobka", "transmission fluid"),
    (r"filtr", "filter"),
    (r"yon ko'zgu|ko'zgu|oyna", "mirror window"),
    (r"kalit|qulf", "key lock unlock"),
    (r"bagaj", "trunk"),
    (r"kondits|kondens", "air conditioning climate control"),
    (r"chiroq|fara", "lights headlights"),
    (r"texnik xizmat", "maintenance minder service"),
    (r"qachon|tez.tez|har qancha", "how often when"),
    (r"tekshir", "check inspection"),
    (r"almashtir", "change replace"),
    (r"qanday|qaysi", "what recommended"),
    (r"qancha|necha", "how much"),
    (r"narx", "price cost"),
    (r"zavod|kompaniya", "company factory"),
    (r"uglerod|is gazi", "carbon monoxide exhaust gas"),
]


def expand_query(question):
    normalized = question.lower().translate(str.maketrans({"‘": "'", "’": "'", "ʻ": "'", "ʼ": "'"}))
    matches = [english for pattern, english in TERMS if re.search(pattern, normalized)]
    # Savolda kamida bir mazmunli texnik atama bo'lsin; faqat "qanday" yetmaydi.
    if not any(re.search(pattern, normalized) for pattern, _ in TERMS[:20]):
        return question
    # Kichik lug'at barcha savol mazmunini tarjima qilmaydi: yillar/sonlar saqlanadi.
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)
    return " ".join(dict.fromkeys(" ".join(matches + numbers + acronyms).split()))
