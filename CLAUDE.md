# CLAUDE.md — Ultron loyihasi

## Loyiha haqida
Ultron — Windows 11 laptopda **to'liq lokal** ishlaydigan, o'zbek tilidagi shaxsiy ovozli AI yordamchi.
Foydalanuvchi uni **"Altron"** (yoki "Ultron") deb chaqiradi. Yordamchi o'zini **Ultron** deb ataydi.

Pipeline: Mikrofon → VAD → Wake word → STT (faster-whisper) → LLM (Ollama + Qwen 7–9B, tool calling) → Tools → TTS → Karnay.

To'liq TZ: `docs/tz/`. Arxitektura va interfeyslar: `docs/tz/00_umumiy_arxitektura.md`.
Bosqichlar holati: `docs/PROGRESS.md`.

## Apparat va cheklovlar
- RTX 4060 Laptop **8 GB VRAM** (asosiy cheklov), Ryzen 7 7435HS, 24 GB RAM, Windows 11.
- VRAM byudjeti: STT ≤ 2 GB, LLM ≤ 5.5 GB. TTS, VAD va wake word **CPU'da** ishlaydi.
- Bulutdagi AI API'lar (OpenAI, Anthropic, Google va h.k.) **TAQIQLANGAN**. Internet faqat brauzer ochish va ixtiyoriy ochiq API'lar (ob-havo) uchun, config'dagi flag orqali.

## Texnologiyalar
- Python **3.11**, paket menejeri **uv** (`uv add`, `uv run`). pip to'g'ridan-to'g'ri ishlatilmaydi.
- asyncio asosidagi arxitektura. Og'ir sinxron ishlar `asyncio.to_thread` yoki alohida thread'da bajariladi.
- Config: `config/config.yaml`, pydantic v2 model orqali (`ultron.config`). Hardcode qilingan qiymatlar bo'lmasligi kerak.
- Log: `loguru`, `data/logs/` papkasiga rotatsiya bilan yoziladi.
- Testlar: `pytest` (+ `pytest-asyncio`). Apparatga bog'liq testlar `@pytest.mark.hardware` bilan belgilanadi.

## Kod qoidalari
- Kod, izohlar, identifikatorlar — **ingliz tilida**. Foydalanuvchiga ko'rinadigan va eshitiladigan matnlar — **o'zbek tilida (lotin yozuvi)**, `ultron/i18n/uz.py` ichida saqlanadi.
- Type hints majburiy. `ruff` bilan lint va format.
- Har bir modul o'z interfeysiga (`Protocol`/ABC) ega bo'ladi. Interfeyslar 00-TZ'da belgilangan, ularni o'zgartirish uchun avval foydalanuvchidan so'rash kerak.
- Xavfli amallar (o'chirish, shutdown, restart, fayl ko'chirish) **faqat ovozli tasdiqdan keyin** bajariladi.
- Fayl operatsiyalari faqat `config.tools.files.allowed_roots` ichida bajariladi.
- Hech qachon `print` ishlatilmaydi, faqat logger.
- Windows yo'llari uchun `pathlib` ishlatiladi. Loyiha `C:\ultron` papkasida joylashgan.

## Buyruqlar
```powershell
uv sync                          # bog'liqliklarni o'rnatish
uv run python -m ultron          # to'liq ovozli rejim
uv run python -m ultron --text   # matnli debug rejim (mikrofonsiz)
uv run python scripts/check_env.py
uv run pytest -m "not hardware"
uv run ruff check . ; uv run ruff format .
```

## Ish tartibi
- Har bir sessiyada **faqat bitta bosqich** TZ'si bajariladi. Boshqa bosqichlarning kodiga faqat TZ'da ko'rsatilgan integratsiya nuqtalarida tegiladi.
- Bosqich oxirida: qabul mezonlari tekshiriladi → `docs/PROGRESS.md` ga yozuv qo'shiladi → foydalanuvchiga hisobot beriladi.
- Agar TZ'da noaniqlik bo'lsa yoki kutubxona API'si TZ'dagidan farq qilsa, rasmiy hujjatga tayaniladi va farq hisobotda ko'rsatiladi.
