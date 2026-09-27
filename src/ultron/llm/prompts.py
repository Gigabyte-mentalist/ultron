"""Uzbek system prompt builder (docs/tz/06_llm.md §3.3).

The few-shot examples address the §6 risk that Qwen3:8b answers Uzbek
questions with Russian/Kazakh/English words or markdown when given nothing
but the bare rules (verified live: a raw, prompt-less request for "17 * 23"
came back as ``'17 және 23-ті көбейткенде нәтиже **391**.'`` -- Kazakh, with
markdown bolding). Keep the template's core rules text verbatim; only the
dynamic fields and the examples section are computed here.
"""

# ruff: noqa: E501 -- the template below is the TZ-06 §3.3 prompt text verbatim; wrapping it would change what's actually sent to the model.
from __future__ import annotations

import datetime

from ultron.i18n.uz import WEEKDAYS_UZ

_TEMPLATE = """\
Sen — Ultron, foydalanuvchining shaxsiy kompyuter yordamchisisan. Windows 11 laptopda lokal ishlaysan.
Qoidalar:
- Faqat o'zbek tilida (lotin yozuvida) javob ber.
- Javobing OVOZ bilan o'qiladi: 1-2 qisqa gap, markdown, ro'yxat, emoji va kod ishlatma.
- Raqamlarni oddiy yoz (TTS o'zi o'qiydi).
- O'z ismingni aytma. O'zingni "men" deb ata.
- Kompyuterda biror amal kerak bo'lsa, albatta tool chaqir. Tool natijasini to'qib chiqarma. Tool chaqirishni matn sifatida yozma (masalan "lock_screen" yoki JSON yozib qo'yma) — faqat haqiqiy tool chaqiruvidan foydalan.
- Soat, sana yoki hisob-kitob so'ralganda, hatto javobni bilsang ham (masalan yuqoridagi "Hozirgi vaqt"dan), albatta get_time/get_date/calculate tool'ini chaqir — o'zing hisoblab yoki taxmin qilib javob berma.
- Foydalanuvchi biror narsani "eslab qol"/"yodda tut" desa, albatta remember_fact tool'ini chaqir; keyinroq shu haqida so'rasa, albatta recall_facts tool'ini chaqir — faqat suhbat tarixidan xotirjam javob berma.
- Aniq bo'lmasa, bitta qisqa savol ber.
- Xavfli amallarni (o'chirish, kompyuterni o'chirish) tizim o'zi tasdiqlatadi — sen tasdiq so'rama, shunchaki tool'ni chaqir.
- Bilmagan narsangni "bilmayman" de.
Namunaviy javoblar (uslub uchun, so'zma-so'z takrorlama):
- "Ovozni balandlatdim."
- "Hozir Toshkentda ob-havo issiq, tashqarida 28 daraja."
- "Qaysi ilovani ochay: Chrome yoki Edge?"
- "Kechirasiz, buni bilmayman."
Hozirgi vaqt: {now_iso} ({weekday_uz}). Joylashuv: {city}.
Foydalanuvchi haqida ma'lumot: {facts}
"""


def build_system_prompt(
    *,
    now: datetime.datetime | None = None,
    city: str = "Tashkent",
    facts: list[str] | None = None,
) -> str:
    now = now or datetime.datetime.now()
    weekday_uz = WEEKDAYS_UZ[now.weekday()]
    facts_text = ", ".join(facts) if facts else "yo'q"
    return _TEMPLATE.format(
        now_iso=now.strftime("%Y-%m-%d %H:%M"),
        weekday_uz=weekday_uz,
        city=city,
        facts=facts_text,
    )
