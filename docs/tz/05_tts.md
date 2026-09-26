# TZ-05. TTS: o'zbekcha ovoz

**Bog'liqlik:** TZ-01, TZ-02 (player), TZ-03 (translit)
**Natija:** o'zbekcha lotin matn → tabiiy eshitiladigan audio, CPU'da, gap-gap bo'lib streaming.

---

## 1. Maqsad
Ultron o'zbek tilida, tushunarli va yetarlicha tez gapirishi kerak. **VRAM LLM uchun kerak**, shuning uchun TTS **CPU'da** ishlaydi.

## 2. Qamrov
**Kiradi:** `tts/base.py`, `tts/mms_engine.py`, `tts/preprocess.py`, `text/numbers_uz.py`, `download_models.py --tts`, `scripts/tts_say.py`, TTS modellarini solishtirish.
**Kirmaydi:** ovoz klonlash, GPU TTS (keyinroq).

---

## 3. Texnik talablar

### 3.1. Model tanlash
- **Asosiy nomzod:** Meta MMS-TTS (VITS) o'zbek modeli. HF'da `facebook/mms-tts-uzb-script_cyrillic` nomi bilan bo'lishi kutiladi. Claude Code **avval HF'da tekshiradi**: model nomi, yozuvi (kirill yoki lotin), litsenziyasi (MMS — CC-BY-NC 4.0, shaxsiy foydalanish uchun mos).
- **Muqobillar:** HF'da `uzbek tts`, `uz vits`, `uzbek piper` qidiriladi. Ochiq litsenziyali va CPU'da real vaqtdan tez ishlaydigan 1–2 ta model tanlanadi.
- **Solishtirish** (`scripts/tts_compare.py`): 15 ta test gapni har bir model bilan generatsiya qiladi → `data/reports/tts_compare/<model>/NN.wav` + RTF (real-time factor) jadvali. Foydalanuvchi eshitib, 1–5 ball qo'yadi (skript so'raydi va `scores.tsv` ga yozadi). G'olib config'ga yoziladi.
- **Talab:** CPU'da RTF ≤ 0.3 (1 s audio ≤ 0.3 s da generatsiya qilinadi).

### 3.2. `MMSEngine` (`mms_engine.py`)
- `transformers.VitsModel` + `AutoTokenizer`. Model `data/models/tts/` papkasiga saqlanadi.
- `device="cpu"`, `torch.inference_mode()`, `torch.set_num_threads(4)` (qolgan yadrolar audio va boshqa ishlar uchun bo'sh qoladi).
- Agar tokenizer `is_uroman=True` bo'lsa, uroman romanizatsiyasi kerak bo'ladi. Bu tekshiriladi va hisobotda aytiladi.
- `speaking_rate` = `tts.speed`, `noise_scale` va `noise_scale_duration` sozlanadigan. Barqaror ovoz uchun `torch.manual_seed` belgilanadi.
- Chiqish: float32, `model.config.sampling_rate` (odatda 16 kHz). Player o'zi resample qiladi.
- `load()` vaqtida warmup (bitta qisqa gap).
- **Kesh:** tez-tez aytiladigan iboralar ("Ha, xo'jayin", "Bajarildi", "Kechirasiz, tushunmadim", "Tasdiqlaysizmi?") ishga tushishda oldindan generatsiya qilinadi va xotirada saqlanadi. `i18n/uz.py` dagi barcha qisqa javoblar keshlanadi. Natijada ular 0 ms kechikish bilan chaladi.

### 3.3. Matnni tayyorlash (`tts/preprocess.py`)
Tartib muhim:
1. **Wake so'zlarini olib tashlash:** "Ultron", "Altron" va variantlari o'chiriladi (self-trigger oldini olish). Ultron o'zini "men" deb ataydi.
2. **Markdown va maxsus belgilarni tozalash:** `*`, `#`, `` ` ``, `[]()`, emoji, URL'lar ("havola" so'zi bilan almashtiriladi).
3. **Raqamlar → so'zlar** (`text/numbers_uz.py`):
   - Butun sonlar: `0..999 999 999 999` ("ikki ming yigirma olti").
   - O'nli kasrlar: `3.5` → "uch butun o'ndan besh".
   - Foiz: `50%` → "ellik foiz".
   - Vaqt: `09:30` → "to'qqiz o'ttiz"; `14:05` → "o'n to'rt nol besh".
   - Sana: `27.09.2026` → "yigirma yettinchi sentyabr, ikki ming yigirma oltinchi yil".
   - Tartib son: `5-` → "beshinchi".
   - Harorat: `-3°C` → "minus uch daraja".
4. **Qisqartmalar lug'ati:** `GB` → "gigabayt", `MB`, `km`, `kg`, `CPU` → "protsessor", `%` va h.k. (`preprocess.py` dagi dict).
5. **Lotin → kirill** (agar `model_script == "cyrillic"`): `text/translit.py` orqali.
6. **Gaplarga bo'lish:** `.`, `!`, `?`, `;` bo'yicha. 200 belgidan uzun gap vergul bo'yicha bo'linadi. Juda qisqa bo'laklar (< 3 so'z) qo'shnisiga qo'shiladi.

### 3.4. Streaming interfeysi
```python
async def speak_stream(self, sentences: AsyncIterator[str], player: AudioPlayer) -> None:
    """LLM'dan kelayotgan gaplarni navbat bilan sintez qilib, player'ga uzatadi.
    Keyingi gap sintezi oldingisi chalinayotganda boshlanadi (pipelining)."""
```
- Sintez `asyncio.to_thread` da bajariladi. Bir vaqtda bittadan ortiq sintez bo'lmaydi.
- Gaplar orasida `sentence_pause_ms` jimlik qo'shiladi.
- `cancel()` sintez navbatini tozalaydi (barge-in uchun).

### 3.5. CLI
```
uv run python scripts/tts_say.py "Salom! Bugun yigirma yettinchi sentyabr, soat 09:30."
uv run python scripts/tts_say.py --file long_text.txt --stream
```

---

## 4. Testlar
- `test_numbers_uz.py`: kamida 50 holat (0, 11, 20, 100, 101, 1000, 1 000 000, 2026, kasrlar, foiz, vaqt, sana, tartib son, manfiy).
- `test_preprocess.py`: wake so'zlarini o'chirish, markdown tozalash, gapga bo'lish.
- `test_mms_engine.py` (`@hardware` yoki sekin test): generatsiya bo'sh emas, RTF chegarada.

## 5. Qabul mezonlari
- [ ] Kamida 2 ta TTS modeli solishtirilgan, foydalanuvchi ballari asosida g'olib tanlangan
- [ ] CPU'da RTF ≤ 0.3, birinchi gapning birinchi audiosi ≤ 400 ms (o'rtacha uzunlikdagi gap)
- [ ] Raqamlar, sana va vaqt o'zbekcha to'g'ri o'qiladi (`tts_say.py` bilan qo'lda tekshiriladi)
- [ ] Keshlangan iboralar darhol chaladi
- [ ] Chiqish matnida hech qachon "Ultron/Altron" bo'lmaydi
- [ ] Testlar o'tadi, `PROGRESS.md` yangilangan

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| MMS ovozi "robotga o'xshash" | Qabul qilinadi (MVP). Keyinroq: yaxshiroq o'zbekcha model yoki o'z ovozingizda VITS/Piper fine-tune |
| Kirill modeli lotin tokenlarni tushirib qoldiradi | Translit sifati testlari, noma'lum belgilarni log qilish |
| Ruscha yoki inglizcha so'zlar (ilova nomlari) noto'g'ri o'qiladi | `preprocess.py` da talaffuz lug'ati: "Telegram" → "Telegram", "YouTube" → "Yutub", "Chrome" → "Xrom" |
| CPU yuklamasi LLM streaming bilan to'qnashadi | `set_num_threads(4)`. LLM GPU'da bo'lgani uchun muammo kichik |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/00_umumiy_arxitektura.md (§5 tts, §6, §7.6) ni o'qi.
docs/tz/05_tts.md ni to'liq o'qi va faqat shu bosqichni bajar.
Avval HF'dagi mavjud o'zbekcha TTS modellarini tekshirib, menga qisqa ro'yxat ber (nomi, yozuvi, litsenziyasi),
keyin solishtirish skriptini ishga tushir va menga eshitib baholashimni ayt.
```
