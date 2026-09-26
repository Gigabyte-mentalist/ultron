# TZ-04. Wake word: "Altron" / "Ultron"

**Bog'liqlik:** TZ-02, TZ-03
**Natija:** "Altron" deyilganda **ishonchli** `WakeEvent` chiqaradigan detektor. U birinchi kundan ishlaydi (maxsus model o'qitishsiz) va keyinchalik openWakeWord modeliga o'tishga tayyor.

---

## 1. Maqsad va asosiy talab
Foydalanuvchi talabi: **"Altron" deganimda har doim javob qaytarishi shart.** Maxfiylik muhim emas, doimiy tinglashga ruxsat berilgan.

Shuning uchun strategiya: **"yolg'on rad etish"dan (false reject) ko'ra "yolg'on uyg'onish" (false accept) afzal**, lekin keyingisi ham nazorat ostida bo'lishi kerak. Ikki bosqichli yondashuv:

| Rejim | Qanday ishlaydi | Afzalligi | Kamchiligi |
|---|---|---|---|
| `stt_kws` (MVP, default) | VAD har bir gap segmentini ajratadi → Whisper tez transkripsiya qiladi → birinchi 1–2 so'zda "altron" varianti fuzzy qidiriladi | O'qitish kerak emas, "Altron, brauzerni och" kabi bir nafasdagi buyruqni tabiiy qo'llab-quvvatlaydi, juda aniq | Har bir gap uchun GPU ishlatadi, reaktsiya gap tugagandan keyin keladi (~0.8–1 s) |
| `oww` (TZ-10 dan keyin) | openWakeWord maxsus modeli har 80 ms frame'ni baholaydi | Juda tez (so'z aytilishi bilan), yengil | Model o'qitish kerak |
| `hybrid` (yakuniy) | oww past threshold bilan trigger → STT bilan tasdiqlanadi. Parallel ravishda stt_kws zaxira bo'lib ishlaydi | Eng ishonchli | Murakkabroq |

Bu bosqichda **`stt_kws` to'liq**, `oww` esa **interfeys darajasida** (model fayli bo'lsa ishlaydigan qilib) amalga oshiriladi. `hybrid` ham shu yerda yoziladi, lekin to'liq sozlash TZ-10 da.

## 2. Qamrov
**Kiradi:** `wake/base.py`, `wake/matcher.py`, `wake/stt_kws.py`, `wake/oww_detector.py`, `wake/__init__.py` (factory `create_wake_detector(config)`), `scripts/benchmark_wake.py`, `scripts/wake_live.py`.
**Kirmaydi:** openWakeWord modelini o'qitish (TZ-10).

---

## 3. Texnik talablar

### 3.1. `WakeMatcher` (`matcher.py`)
- Kirish: normallashtirilgan transkript (lotin, kichik harf).
- Birinchi `max_prefix_words` (2) so'z va ularning **birlashmasi** tekshiriladi ("al tron" → "altron").
- `rapidfuzz.fuzz.ratio` bilan `wake.phrases` ro'yxatidagi har bir variantga solishtiriladi. Eng yuqori ball ≥ `fuzzy_threshold` (82) bo'lsa → mos keldi.
- Fonetik qo'shimcha: oddiy o'zbek/rus fonetik kaliti (`u↔a↔o` boshida, `lt↔ltr`, qo'shaloq harflar qisqartiriladi). Shunda "Oltron", "Altran", "Ultiron" ham tutiladi.
- Qaytaradi: `MatchResult(matched: bool, score: float, variant: str, trailing_text: str)`. `trailing_text` = wake word'dan keyingi matn, boshidagi vergul/tinish belgilarisiz.
- **Salbiy ro'yxat** (config'da kengaytiriladi): "elektron", "patron", "altin" kabi o'xshash so'zlar faqat to'liq mos kelsa rad etiladi. Bu false accept'ni kamaytiradi.
- Gap o'rtasidagi "altron" (masalan "men altron haqida o'qidim") **trigger qilmaydi**. Faqat gap boshidagisi hisobga olinadi.

### 3.2. `STTKeywordSpotter` (`stt_kws.py`)
- `process_utterance(utt)`: IDLE holatidagi har bir VAD segmenti uchun chaqiriladi.
  1. Segment 0.3–15 s oralig'ida bo'lmasa → e'tiborsiz qoldiriladi.
  2. **Tez yo'l:** segmentning **birinchi 1.5 soniyasi** `kws_model` bilan transkripsiya qilinadi (initial_prompt: `"Altron."`, beam 1). Bu ~150–250 ms oladi.
  3. `WakeMatcher` → mos kelsa:
     - Segment 1.8 s dan uzun bo'lsa (demak, buyruq ham aytilgan), **butun segment** asosiy STT modeli bilan qayta transkripsiya qilinadi → `trailing_text` shu natijadan olinadi.
     - Qisqa bo'lsa → `trailing_text=None` (foydalanuvchi buyruqni keyin aytadi).
  4. `WakeEvent(source="stt_kws", score=match.score/100, trailing_text=...)`.
- **Parallel ishlov:** agar oldingi segment hali transkripsiya qilinayotgan bo'lsa, yangisi navbatga qo'yiladi (maksimum 3 ta). Eskilari tashlanadi.
- GPU'ni tejash uchun: RMS -50 dBFS dan past segmentlar o'tkazib yuboriladi.

### 3.3. `OWWDetector` (`oww_detector.py`)
- `openwakeword` kutubxonasi, ONNX inference framework, CPU.
- `oww_model_path` fayli mavjud bo'lmasa → `load()` aniq xabar bilan `WakeModelMissing` xatosini chiqaradi, factory esa `stt_kws` rejimiga qaytadi (log'da ogohlantirish).
- `process_frame(frame)`: openWakeWord 80 ms (1280 namuna) chunk kutadi. 512 namunali frame'lar ichki buferda yig'iladi.
- Score ≥ `oww_threshold` bo'lsa va `oww_verify_with_stt=True` bo'lsa: RingBuffer'dan oxirgi 1.5 s olinadi → KWS STT → `WakeMatcher`. Tasdiqlansa → `WakeEvent`.
- Score ≥ 0.85 bo'lsa, STT tasdig'isiz qabul qilinadi (tezlik uchun).
- `cooldown_ms` ichida qayta trigger bo'lmaydi.
- openWakeWord'ning "custom verifier model" imkoniyatini qo'llab-quvvatlash uchun joy qoldiriladi (TZ-10).

### 3.4. `HybridDetector`
- `OWWDetector` (frame darajasida) + `STTKeywordSpotter` (utterance darajasida) bir vaqtda ishlaydi.
- Qaysi biri birinchi tasdiqlasa, o'sha `WakeEvent` beradi. Ikkinchisi shu gap uchun bekor qilinadi (dublikat bo'lmasligi uchun, vaqt oynasi 2 s).

### 3.5. Echo va self-trigger himoyasi
- Orchestrator `SPEAKING` holatida detektorga `set_speaking(True)` beradi:
  - `barge_in_enabled=False` bo'lsa → detektor to'liq pauza qilinadi.
  - `True` bo'lsa → thresholdlar `barge_in_threshold_boost` ga oshiriladi, `stt_kws` esa faqat RMS'i player chiqishidan sezilarli baland segmentlarni tekshiradi.
- TTS matnida wake so'zlari bo'lmaydi (TZ-05 da filtr).
- Ijro tugagandan keyin 300 ms "sovish" vaqti.

### 3.6. Vositalar
```
uv run python scripts/wake_live.py                  # jonli: har trigger'da vaqt, score, manba, trailing_text chiqadi + earcon
uv run python scripts/benchmark_wake.py \
    --positives data/recordings/wake_pos \           # record_samples.py wake --count 100
    --negatives data/recordings/wake_neg \           # 30+ daqiqa oddiy gap, TV, musiqa (yozib olinadi yoki fayl beriladi)
    --mode stt_kws
```
Benchmark natijasi: **recall** (tanish ulushi), **soatiga false accept**, o'rtacha va p95 latency (gap tugashidan `WakeEvent`gacha). Bir nechta `fuzzy_threshold` qiymatlari uchun jadval chiqariladi (ROC). Hisobot `data/reports/wake_benchmark_*.md` ga yoziladi.

**Pozitiv namunalar (foydalanuvchi yozadi, 100 ta):** har xil intonatsiya (baland, past, tez, sekin, uzoqdan, pichirlab), faqat "Altron" va "Altron, <buyruq>" aralash, 10 tasi "Ultron". Negativ: 30 daqiqa kundalik gap, YouTube yoki TV fon ovozi.

---

## 4. Testlar
- `test_matcher.py`: kamida 30 ta ijobiy ("altron", "Altron,", "al tron", "oltron", "ультрон", "altron brauzerni och") va 30 ta salbiy ("elektron", "patron", "men altron haqida", "salom") holat.
- `test_stt_kws.py`: mock STT bilan: qisqa segment → `trailing_text=None`; uzun segment → qayta transkripsiya chaqiriladi; navbat to'lishi.
- `test_factory.py`: oww modeli yo'q → stt_kws'ga fallback.

## 5. Qabul mezonlari
- [ ] `wake_live.py` ishlaydi, "Altron" va "Altron, ovozni balandlat" ikkalasini ham to'g'ri ajratadi
- [ ] Benchmark (`stt_kws`): recall **≥ 95%** (100 ta pozitivda), false accept **≤ 1/soat** (30+ daqiqa negativda, soatga normallashtirilgan)
- [ ] Wake latency (gap tugashidan): p95 ≤ 400 ms qisqa "Altron" uchun
- [ ] oww modeli yo'q bo'lganda tizim yiqilmaydi, stt_kws'ga o'tadi
- [ ] Gap o'rtasidagi "altron" trigger qilmaydi
- [ ] Testlar o'tadi, `PROGRESS.md` yangilangan. Tanlangan `fuzzy_threshold` config'ga yozilgan

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| Whisper "Altron"ni "Al-Tron", "Altro", "Aldron" deb yozadi | `initial_prompt="Altron."`, variantlar ro'yxatini benchmark natijasi asosida kengaytirish |
| TV yoki YouTube'da "Ultron" (Marvel) so'zi | Qabul qilinadigan xavf, faqat gap boshidagisi hisobga olinadi. Kerak bo'lsa TZ-10 dagi shaxsiy ovoz verifier |
| GPU doim band (ko'p gapli muhit) | RMS filtri, navbat limiti. Kelajakda `oww` rejimiga o'tish |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/00_umumiy_arxitektura.md (§2.2, §5 wake, §6) ni o'qi.
docs/tz/04_wake_word.md ni to'liq o'qi va faqat shu bosqichni bajar.
Kodni yozib bo'lgach menga namunalarni yozish buyruqlarini ber (100 pozitiv, 30 daqiqa negativ),
keyin benchmark'ni ishga tushirib, eng yaxshi fuzzy_threshold'ni tanla va config'ga yoz.
```
