# TZ-06. LLM "miya": Qwen + Ollama, tool calling, fast-path, text rejim

**Bog'liqlik:** TZ-01 (TZ-03 va TZ-05 ixtiyoriy, text rejim ularsiz ham ishlaydi)
**Natija:** matn → (tool chaqiruvlari) → o'zbekcha javob **gaplar oqimi** ko'rinishida. `python -m ultron --text` orqali klaviaturada to'liq sinab ko'rish mumkin.

---

## 1. Maqsad
Foydalanuvchi niyatini tushunish, kerakli tool'larni chaqirish va qisqa, og'zaki nutqqa mos o'zbekcha javob qaytarish. Tezlik uchun eng ko'p ishlatiladigan buyruqlar **LLM'siz** bajariladi (fast-path).

## 2. Qamrov
**Kiradi:** `llm/client.py`, `llm/agent.py`, `llm/prompts.py`, `llm/fastpath.py`, `llm/sentence_splitter.py`, `tools/base.py` (registry, decorator), 3 ta demo tool (`get_time`, `get_date`, `calculate`), `__main__.py --text` rejimi, `scripts/eval_llm_tools.py`.
**Kirmaydi:** haqiqiy tizim/fayl/brauzer tool'lari (TZ-07), ovoz pipeline (TZ-08).

---

## 3. Texnik talablar

### 3.1. `OllamaClient` (`client.py`)
- `ollama.AsyncClient(host=config.llm.host)`.
- `chat_stream(messages, tools) -> AsyncIterator[ChatChunk]`: `stream=True`, `tools=registry.schemas()`, `options={"temperature", "num_ctx"}`, `keep_alive`, `think=config.llm.think`.
  > Agar o'rnatilgan `ollama` kutubxonasi yoki server versiyasi `think` parametrini qo'llab-quvvatlamasa, fallback: system prompt oxiriga `/no_think` qo'shiladi va javobdagi `<think>...</think>` bloklari tozalanadi.
- Streaming paytida `tool_calls` kelishi mumkin. Ular yig'iladi va oqim oxirida qaytariladi.
- `health()`: `/api/tags` va model mavjudligi. `ensure_loaded()`: bo'sh so'rov bilan modelni VRAM'ga yuklash (ishga tushishda chaqiriladi).
- Xatolar: ulanish xatosi → 3 marta qayta urinish (0.5, 1, 2 s). Keyin `LLMUnavailable`. Timeout → `request_timeout_s`.

### 3.2. Tool freymvorki (`tools/base.py`)
```python
@tool(
    name="open_app",
    description="Open an installed application by name. Use for requests like 'open Telegram'.",
    dangerous=False,
)
async def open_app(name: Annotated[str, "Application name as the user said it"]) -> ToolResult: ...
```
- Decorator funksiya signaturasi va type hint'lardan **JSON Schema** yaratadi (`str`, `int`, `float`, `bool`, `Literal[...]` → enum, `Optional`, default qiymatlar). `Annotated` ichidagi matn → `description`.
- **Tool tavsiflari inglizcha** yoziladi (Qwen tool tanlashda inglizchada aniqroq). Foydalanuvchiga javob esa o'zbekcha bo'ladi.
- `ToolRegistry`: `register`, `schemas(enabled_only=True)`, `async call(name, args) -> ToolResult`.
  - Argumentlar pydantic bilan validatsiya qilinadi. Xato bo'lsa, `ToolResult(ok=False, error="...")` LLM'ga qaytariladi, u tuzatib qayta urinishi mumkin.
  - Har bir chaqiruv audit log'ga yoziladi: vaqt, nom, argumentlar, natija, davomiylik.
  - `dangerous=True` bo'lsa, registry avval `ConfirmationProvider.confirm(summary_uz) -> bool` ni chaqiradi. Text rejimida bu konsolda `[ha/yo'q]`, ovoz rejimida TZ-08 da.
  - Tool'ning maksimal ishlash vaqti 15 s (timeout).
- Tool'lar modul import qilinganda avtomatik ro'yxatga olinadi (`tools/__init__.py` barcha `*_tools.py` fayllarni import qiladi).

### 3.3. System prompt (`prompts.py`)
O'zbekcha, ixcham (~400 token). Dinamik qismlar har so'rovda qo'shiladi:
```
Sen — Ultron, foydalanuvchining shaxsiy kompyuter yordamchisisan. Windows 11 laptopda lokal ishlaysan.
Qoidalar:
- Faqat o'zbek tilida (lotin yozuvida) javob ber.
- Javobing OVOZ bilan o'qiladi: 1–2 qisqa gap, markdown, ro'yxat, emoji va kod ishlatma.
- Raqamlarni oddiy yoz (TTS o'zi o'qiydi).
- O'z ismingni aytma. O'zingni "men" deb ata.
- Kompyuterda biror amal kerak bo'lsa, albatta tool chaqir. Tool natijasini to'qib chiqarma.
- Aniq bo'lmasa, bitta qisqa savol ber.
- Xavfli amallarni (o'chirish, kompyuterni o'chirish) tizim o'zi tasdiqlatadi, sen shunchaki tool'ni chaqir.
- Bilmagan narsangni "bilmayman" de.
Hozirgi vaqt: {now_iso} ({weekday_uz}). Joylashuv: {city}.
Foydalanuvchi haqida ma'lumot: {facts}
```
- `facts` — TZ-07 dagi xotiradan (hozircha bo'sh ro'yxat).
- Muloqot tarixi: oxirgi `history_turns` (8) juftlik. Tool natijalari tarixda qisqartirilgan holda saqlanadi (≤ 500 belgi).
- **Sessiya:** follow-up oynasi tugagach yoki 5 daqiqa faolsizlikdan keyin tarix tozalanadi (xotiradagi faktlar saqlanadi).

### 3.4. Agent sikli (`agent.py`)
```python
async def respond(self, user_text: str) -> AsyncIterator[AgentEvent]:
    # AgentEvent: SentenceEvent(text) | ToolStartEvent(name, args) | ToolEndEvent(result) | DoneEvent | ErrorEvent
```
1. Avval **fast-path** tekshiriladi (3.5). Mos kelsa → tool bevosita chaqiriladi → `ToolResult.speak` yoki shablon javob → tugaydi.
2. Aks holda: messages = system + history + user → `chat_stream`.
3. Stream'dagi matn `SentenceSplitter` dan o'tadi. Tayyor bo'lgan har bir gap darhol `SentenceEvent` bo'lib chiqariladi (TTS darhol boshlaydi).
4. `tool_calls` bo'lsa → `ToolStartEvent` (orchestrator `ack` earcon chaladi) → tool'lar bajariladi (mustaqil bo'lsa parallel, `asyncio.gather`) → natijalar `role="tool"` sifatida qo'shiladi → 2-qadamga qaytiladi.
5. Maksimum `max_tool_iterations` (5). Oshsa → "Kechirasiz, buni bajara olmadim."
6. Tool `ToolResult.speak` qaytarsa va LLM keyingi iteratsiyada hech narsa demasa → `speak` matni aytiladi.
7. Bekor qilish: `respond()` generatori `aclose()` qilinsa, Ollama so'rovi ham to'xtatiladi (barge-in).

### 3.5. Fast-path (`fastpath.py`)
LLM'siz, regex + fuzzy asosida (`normalized_lower` matn ustida). Taxminan **15–25 ta intent**. Ularning har biri tool + argument qaytaradi:
| Intent | Misol iboralar | Tool |
|---|---|---|
| volume_up / down | "ovozni balandlat/ko'tar/qo'sh", "ovozni pasaytir" | `set_volume(delta=±10)` |
| volume_set | "ovozni ellik foiz qil", "ovoz 30" | `set_volume(level=N)` |
| mute / unmute | "ovozni o'chir", "ovozni yoq" | `mute(True/False)` |
| open_app | "<X>ni och", "<X> och", "<X>ni ishga tushir" | `open_app(name=X)` |
| close_app | "<X>ni yop" | `close_app(name=X)` |
| lock | "kompyuterni qulfla", "ekranni qulfla" | `lock_screen()` |
| time / date | "soat nechi", "bugun nechanchi" | `get_time()` / `get_date()` |
| screenshot | "skrinshot ol", "ekranni rasmga ol" | `screenshot()` |
| brightness | "yorug'likni oshir/kamaytir", "yorug'lik 70" | `set_brightness(...)` |
| youtube | "YouTube'da <X> qo'y/och" | `youtube_search(query=X)` |
- O'zbekcha raqam so'zlari ("ellik", "o'ttiz besh") → son (`text/numbers_uz.py` ga `words_to_number` qo'shiladi).
- Fast-path faqat **ishonch yuqori** bo'lganda ishlaydi (to'liq regex moslik). Shubha bo'lsa → LLM.
- TZ-07 gacha tool'lar mavjud bo'lmasa, fast-path ularni o'tkazib yuboradi (registry'da yo'q intent o'chirilgan hisoblanadi).

### 3.6. `SentenceSplitter`
- Token oqimini qabul qiladi, `.!?` + bo'shliq yoki qator oxirida gap chiqaradi. "5.5", "t.b.", "va h.k." kabi holatlarda bo'lmaydi.
- Birinchi gap uchun **erta chiqarish**: 60+ belgi to'plansa va vergul uchrasa, shu yerda chiqariladi (birinchi audio tezroq boshlanishi uchun).
- `<think>` bloklari va markdown belgilari shu yerda ham tozalanadi.

### 3.7. Text rejim (`python -m ultron --text`)
- Konsol REPL: `Siz: ...` → gaplar oqim bo'lib chiqadi, tool chaqiruvlari `[tool] open_app {"name": "telegram"} → ok (35 ms)` ko'rinishida.
- `--speak` flag'i: javob TTS orqali aytiladi (TZ-05 tayyor bo'lsa).
- Buyruqlar: `/reset` (tarixni tozalash), `/tools` (ro'yxat), `/fast on|off`, `/exit`.
- Bu rejim keyinchalik ham **asosiy debug vositasi** bo'lib qoladi.

### 3.8. `scripts/eval_llm_tools.py`
- `scripts/data/llm_eval_uz.yaml`: **kamida 40 ta** holat: `input`, `expected_tool` (yoki `null`), `expected_args` (qisman moslik), `must_contain` (javobda).
- Tool'lar **mock** rejimda (haqiqiy amal bajarilmaydi, faqat chaqiruv yoziladi).
- Metrikalar: tool tanlash aniqligi, argument aniqligi, javob o'zbekchada ekanligi (oddiy evristika: kirill yoki ingliz so'zlari ulushi), o'rtacha va p95 latency (birinchi gapgacha).
- Bir nechta model yoki prompt versiyasini solishtirish uchun: `--model qwen3:8b --model qwen2.5:7b`.
- Hisobot: `data/reports/llm_eval_*.md`.

---

## 4. Testlar
- `test_tool_schema.py`: decorator'dan to'g'ri JSON schema chiqadi (Literal, Optional, default).
- `test_registry.py`: validatsiya xatosi → `ok=False`; dangerous → confirm chaqiriladi; timeout.
- `test_sentence_splitter.py`: kasr sonlar, qisqartmalar, erta chiqarish, think teglar.
- `test_fastpath.py`: 40+ ibora → kutilgan intent va argumentlar; shubhali iboralar → `None`.
- `test_agent.py`: mock Ollama client bilan tool sikli, max iteratsiya, bekor qilish.

## 5. Qabul mezonlari
- [ ] `--text` rejimida "soat nechi?", "bugun nechanchi sana?", "17 ni 23 ga ko'paytir" to'g'ri tool orqali javob beradi
- [ ] `eval_llm_tools.py` (demo va mock tool'lar bilan): tool tanlash aniqligi **≥ 85%**
- [ ] Birinchi gapgacha latency (LLM yo'li, tool'siz): p50 ≤ 700 ms, p95 ≤ 1200 ms (model VRAM'da yuklangan holda)
- [ ] Fast-path intentlari ≤ 50 ms ichida aniqlanadi
- [ ] Javoblar markdown'siz, 1–2 gapli, o'zbekcha (eval evristikasi ≥ 90%)
- [ ] Ollama o'chirib qo'yilsa, tizim yiqilmaydi, "Hozir o'ylay olmayapman, bir oz kuting" deydi va qayta ulanadi
- [ ] Testlar o'tadi, `PROGRESS.md` yangilangan

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| Qwen 8B o'zbekchada grammatik xato qiladi yoki ruscha/turkcha so'z qo'shadi | Prompt'ga 3–4 ta qisqa namunaviy javob (few-shot) qo'shish; eval orqali prompt versiyalarini solishtirish; kerak bo'lsa javob matnini qisqa ushlash |
| Qwen tool'ni noto'g'ri argument bilan chaqiradi | Pydantic xatosini LLM'ga qaytarib, qayta urinish; fast-path qamrovini kengaytirish |
| Streaming + tool_calls kutubxona versiyasida ishlamaydi | Tool chaqirilishi kutilgan birinchi so'rovni `stream=False`, yakuniy javobni `stream=True` bilan yuborish (fallback rejimi, config flag) |
| Thinking rejimi kechikish qo'shadi | `think=false`, `/no_think`, `<think>` tozalash |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/00_umumiy_arxitektura.md (§5 llm/fastpath, §6 tools/base, §7) ni o'qi.
docs/tz/06_llm.md ni to'liq o'qi va faqat shu bosqichni bajar.
O'rnatilgan ollama kutubxonasi va server versiyasini tekshir; think va streaming tool_calls qo'llab-quvvatlanishini sinab ko'r va natijani yoz.
Oxirida eval natijasi va --text rejimidan 5 ta namunaviy dialogni ko'rsat.
```
