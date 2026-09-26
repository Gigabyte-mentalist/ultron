# TZ-03. STT: nutqni matnga aylantirish (o'zbek tili)

**Bog'liqlik:** TZ-01, TZ-02
**Natija:** `Utterance` → `Transcript` (lotin yozuvida, tozalangan). O'zbekcha test to'plamida WER o'lchanadi va eng yaxshi model tanlanadi.

---

## 1. Maqsad
O'zbekcha og'zaki buyruqlarni tez (≤ 500 ms) va iloji boricha aniq matnga aylantirish. **O'zbek tili — loyihaning eng zaif nuqtasi**. Shuning uchun bu bosqichda model tanlash **o'lchov asosida** qilinadi, taxmin bilan emas.

## 2. Qamrov
**Kiradi:** `stt/base.py`, `stt/whisper_engine.py`, `stt/filters.py`, `text/translit.py`, `text/normalize.py`, `scripts/benchmark_stt.py`, test to'plamini tayyorlash, `download_models.py --stt`.
**Kirmaydi:** fine-tune (TZ-11).

---

## 3. Texnik talablar

### 3.1. `WhisperEngine` (`whisper_engine.py`)
- `faster_whisper.WhisperModel(model, device="cuda", compute_type="int8_float16", download_root=data/models/stt)`.
- `transcribe()` parametrlari (config'dan):
  - `language="uz"` (majburiy; avtomatik aniqlash o'zbekni ko'pincha turk yoki qozoq deb o'ylaydi)
  - `beam_size=1` (tezlik uchun, benchmark'da 1 va 3 solishtiriladi)
  - `vad_filter=False` (VAD bizda allaqachon bor)
  - `condition_on_previous_text=False`
  - `initial_prompt` — domen so'zlari ("Altron", ilova nomlari, buyruq namunalari). Bu o'zbekcha imlo va "Altron" so'zining yozilishini sezilarli yaxshilaydi
  - `without_timestamps=True`
  - `temperature=0.0`
- `load()` vaqtida **warmup**: 1 s jimlikni transkripsiya qiladi (birinchi so'rov sekin bo'lmasligi uchun).
- Thread-safe: bitta `threading.Lock`. Chaqiruvlar `asyncio.to_thread` orqali.
- Bir model instance'i bir nechta "profil" bilan ishlatilishi mumkin (asosiy va KWS uchun). Agar `stt.kws_model == stt.model` bo'lsa, qayta yuklanmaydi.
- `Transcript.confidence` = segmentlarning `exp(avg_logprob)` o'rtachasi.

### 3.2. Hallucination filtrlari (`filters.py`)
Whisper jimlik yoki shovqinda "gap to'qiydi". Filtrlar:
1. `no_speech_prob > no_speech_threshold` **va** `avg_logprob < log_prob_threshold` bo'lsa → bo'sh natija.
2. Qora ro'yxat (regex, katta-kichik harf farqsiz): `subtit`, `субтитр`, `продолжение следует`, `rahmat tomosha`, `obuna bo'ling`, `подписывайтесь`, `amara.org`, `♪` va h.k. (ro'yxat `filters.py` ichida, kengaytiriladigan).
3. Takrorlanish: bir xil 3+ so'zli ibora 3 martadan ko'p takrorlansa → takrorlar kesiladi.
4. Juda qisqa audio (< 0.3 s) → transkripsiya qilinmaydi.

### 3.3. Transliteratsiya (`text/translit.py`)
O'zbek lotin ↔ kirill, **ikki tomonlama**, qoidalar asosida (TTS ham ishlatadi):
- Kirill → lotin: `ш→sh, ч→ch, ў→o', ғ→g', қ→q, ҳ→h, ё→yo, ю→yu, я→ya, ц→ts (so'z o'rtasida)/s (boshida), э→e, ъ→', ь→(olib tashlanadi), е→ye (so'z boshida va unlidan keyin)/e`.
- Lotin → kirill: `sh, ch, o', g', ng` ketma-ketliklari birinchi ishlanadi; `yo, yu, ya, ye` → `ё, ю, я, е`; so'z boshidagi `e` → `э`; `'` (tutuq belgisi) → `ъ`.
- Apostrof variantlari normallashtiriladi: `ʻ ʼ ' ' \` ´` → `o'` va `g'` uchun bitta kanonik belgi (`'`).
- Rus tilidan kirgan so'zlar uchun istisno lug'ati (`kompyuter`, `sentyabr`, ...). Kichik bo'lsa ham bo'lishi kerak, keyin kengaytiriladi.
- Testlar: kamida 40 ta juftlik (`tests/data/translit_pairs.tsv`).

### 3.4. Normallashtirish (`text/normalize.py`)
`normalize_transcript(raw: str) -> str`:
1. Kirill harflari 30% dan ko'p bo'lsa → lotinga o'tkaziladi.
2. Apostroflar kanonik ko'rinishga keltiriladi.
3. Ortiqcha bo'shliqlar va boshidagi/oxiridagi tinish belgilari tozalanadi.
4. Wake word variantlari ("Ultron", "Altron", "Oltron", "Al tron") kanonik `Altron` shakliga keltiriladi (fuzzy matcher TZ-04 da, bu yerda oddiy regex).
5. Kichik harfga o'tkazilmaydi (LLM uchun asl ko'rinish saqlanadi), lekin `normalized_lower` alohida qaytariladi (fast-path uchun).

### 3.5. Test to'plami va benchmark
**Test to'plami** (foydalanuvchi yozadi, `record_samples.py testset` orqali):
- `scripts/data/prompts_uz.txt` — Claude Code tayyorlaydi, **kamida 80 ta gap**:
  - 30 ta tizim buyrug'i ("Ovozni ellik foizga qo'y", "Telegramni och", "Kompyuterni qulfla")
  - 15 ta fayl buyrug'i ("Yuklamalar papkasini och", "Rezyume degan faylni top")
  - 15 ta brauzer ("YouTube'da o'zbek musiqalarini qo'y", "Google'da ob-havoni qidir")
  - 10 ta xotira/eslatma ("Ertaga soat to'qqizda uchrashuvni eslat", "Mening tug'ilgan kunim o'n beshinchi mart")
  - 10 ta erkin savol ("Bugun qanday kun?", "Menga bir hikmatli gap ayt")
  - "Altron, ..." bilan boshlanadigan 20 ta gap (ular yuqoridagilar ichida)
- Yozish sharoiti: laptop mikrofoni, 0.5–2 m masofa, oddiy xona. (Keyinroq naushnik bilan ham.)

**`scripts/benchmark_stt.py`:**
```
uv run python scripts/benchmark_stt.py --models large-v3-turbo large-v3 medium small <hf_uz_modellar...> --beam 1 3
```
- Har bir model va beam uchun: **WER, CER** (`jiwer`, normallashtirilgan va kichik harfli matnda), o'rtacha va p95 latency, VRAM (`nvidia-smi` orqali), "Altron" so'zining to'g'ri tanilish ulushi.
- Natija: `data/reports/stt_benchmark_YYYYMMDD.md` jadvali.
- **O'zbekcha fine-tune modellar:** Claude Code Hugging Face'da `whisper uzbek` / `whisper uz` qidiradi. Litsenziyasi ochiq, yuklab olishlar soni yuqori 2–4 ta modelni tanlaydi va `ct2-transformers-converter --quantization int8_float16` bilan CTranslate2 formatiga o'giradi (`data/models/stt/<nom>-ct2`). Model nomlarini TZ'da oldindan belgilamayman, chunki ular tez o'zgaradi. Tanlov benchmark natijasiga qarab qilinadi.
- **Tanlash qoidasi:** WER eng past model, lekin p95 latency ≤ 800 ms (5 s audio uchun) va VRAM ≤ 2.2 GB bo'lishi shart. G'olib `config.yaml` ga yoziladi.

### 3.6. CLI debug
```
uv run python -m ultron.stt --file some.wav     # bitta faylni transkripsiya
uv run python -m ultron.stt --live              # VAD + STT jonli: har bir gapni matn qilib chiqaradi
```

---

## 4. Testlar
- `test_translit.py`: 40+ juftlik, ikki tomonlama, apostrof variantlari.
- `test_normalize.py`: kirill/lotin aralash, wake word variantlari, tinish belgilari.
- `test_filters.py`: qora ro'yxat, takrorlanish, bo'sh natija.
- `test_whisper_engine.py` (`@hardware`): test wav → matn bo'sh emas, latency < 1 s.

## 5. Qabul mezonlari
- [ ] Kamida 80 ta gapdan iborat test to'plami yozilgan (`data/recordings/testset/`)
- [ ] Benchmark kamida 5 ta konfiguratsiyani (shu jumladan ≥ 2 ta o'zbekcha fine-tune) solishtirgan, hisobot mavjud
- [ ] Tanlangan model: WER ≤ 25% (MVP maqsadi). Agar erishilmasa, hisobotda aniq raqam va TZ-11 ni oldinroq bajarish tavsiyasi yoziladi
- [ ] 5 s gap uchun p95 latency ≤ 800 ms
- [ ] `--live` rejimida 1 soat jimlik yoki fon shovqinida hallucination matni chiqmaydi (yoki filtrdan o'tmaydi)
- [ ] Barcha chiqish lotin yozuvida
- [ ] Testlar o'tadi, `PROGRESS.md` yangilangan

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| Hech bir model WER ≤ 25% bermaydi | `initial_prompt`ni boyitish, beam 3, keyin TZ-11 (LoRA) ni ustuvor qilish |
| O'zbekcha fine-tune modellar turbo emas, sekin | large-v2/v3 asosidagi modellarni int8'da sinash; distil yoki turbo asosdagilarini qidirish |
| Whisper aralash yozuv beradi | `normalize.py` (kirill → lotin) |
| Rus/ingliz so'zlari (ilova nomlari) buziladi | `initial_prompt`ga ilova nomlarini qo'shish, fast-path'da fuzzy moslashtirish |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/00_umumiy_arxitektura.md (§5 stt, §6, §7) ni o'qi.
docs/tz/03_stt.md ni to'liq o'qi va faqat shu bosqichni bajar.
Avval kod va prompts_uz.txt faylini tayyorla. Keyin menga test to'plamini yozish buyrug'ini ber, men yozib bo'lgach benchmark'ni ishga tushir.
HF modellarini tanlashda sababini qisqa yoz.
```
