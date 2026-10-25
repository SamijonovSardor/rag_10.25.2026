"""OpenRouter HTTP mijoz; xatolarda API kaliti chiqarilmaydi."""
import time

import requests

from . import config as c


class OpenRouterClient:
    def __init__(self):
        self.api_key = c.require_api_key()
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}",
                                     "Content-Type": "application/json", "X-Title": "Honda Manual RAG"})

    def validate_api_key(self):
        """Bepul /key tekshiruvi; faqat maxfiy bo'lmagan holat qaytariladi."""
        try:
            response = self.session.get(f"{c.OPENROUTER_BASE_URL}/key", timeout=(15, 30))
        except requests.RequestException as exc:
            raise RuntimeError("OpenRouter kalitini tekshirib bo'lmadi. Internet ulanishini tekshiring.") from exc
        if response.status_code == 401:
            raise RuntimeError("OPENROUTER_API_KEY tanilmadi yoki muddati tugagan. "
                               "https://openrouter.ai/settings/keys sahifasidan oddiy API kaliti olib, .env ni yangilang.")
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter kalit tekshiruvi HTTP {response.status_code} qaytardi.")
        try:
            data = response.json()["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("OpenRouter kalit ma'lumotini to'g'ri formatda qaytarmadi.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter kalit ma'lumoti noto'g'ri formatda.")
        if data.get("is_management_key") is True or data.get("is_provisioning_key") is True:
            raise RuntimeError("Management/Provisioning kaliti qo'yilgan. Chat va embedding uchun "
                               "https://openrouter.ai/settings/keys sahifasidan oddiy API kaliti yarating "
                               "va .env dagi OPENROUTER_API_KEY ni almashtiring.")
        return {"authenticated": True, "key_type": "inference"}

    def post(self, endpoint, payload):
        for attempt in range(3):
            try:
                response = self.session.post(f"{c.OPENROUTER_BASE_URL}/{endpoint}", json=payload,
                                             timeout=(15, c.REQUEST_TIMEOUT))
            except requests.Timeout as exc:
                # So'rov bajarilgan bo'lishi mumkin; yana pullik so'rov avtomatik yuborilmaydi.
                raise RuntimeError("OpenRouter javob kutish vaqti tugadi. Qayta urinib ko'ring.") from exc
            except requests.RequestException as exc:
                raise RuntimeError("OpenRouter bilan bog'lanib bo'lmadi. Internet ulanishini tekshiring.") from exc
            if response.status_code in (429, 502, 503) and attempt < 2:
                try:
                    delay = min(max(float(response.headers.get("Retry-After", 2 ** attempt)), 0), 10)
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                if response.status_code == 401:
                    try:
                        self.validate_api_key()
                    except RuntimeError as exc:
                        raise RuntimeError(f"OpenRouter HTTP 401: {exc}") from None
                    raise RuntimeError("OpenRouter HTTP 401: Kalit tanildi, ammo model so'rovi rad etildi. "
                                       "OpenRouter hisobidagi ruxsatlarni tekshiring.")
                messages = {
                    400: "So'rov parametrlari yoki tanlangan model mosligini tekshiring.",
                    402: "OpenRouter hisobida kredit yetarli emas.",
                    403: "API kaliti yoki hisob uchun ushbu modelga ruxsat yo'q.",
                    404: "Model yoki endpoint topilmadi. .env dagi model nomini tekshiring.",
                    429: "So'rovlar limiti tugadi. Birozdan keyin qayta urinib ko'ring.",
                }
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: "
                                   + messages.get(response.status_code, "Xizmat vaqtincha javob bermayapti."))
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError("OpenRouter JSON javob qaytarmadi.") from exc
            if not isinstance(data, dict) or "error" in data:
                raise RuntimeError("OpenRouter so'rovni bajara olmadi. Model va hisob holatini tekshiring.")
            return data
        raise RuntimeError("OpenRouter so'rovini bajarib bo'lmadi.")

    def close(self):
        self.session.close()
