# TZ-02. Audio qatlami: yozish, VAD, endpointing, ovoz chiqarish

**Bog'liqlik:** TZ-01
**Natija:** mikrofondan uzluksiz oqim, gap boshlanishi va tugashini aniqlash (`Utterance` hosil qilish), to'xtatsa bo'ladigan audio player va earcon'lar.

---

## 1. Maqsad
Butun tizimning "quloq" va "og'iz" qismini yaratish. Bu qatlam 24/7 ishlaydi. Shuning uchun u **yengil (CPU < 3%)**, **barqaror** (qurilma uzilsa ham qayta ulanadigan) va **kechikishsiz** bo'lishi shart.

## 2. Qamrov
**Kiradi:** `audio/capture.py`, `audio/ring_buffer.py`, `audio/vad.py`, `audio/player.py`, `audio/earcons.py`, `scripts/record_samples.py` (asosiy qism), testlar.
**Kirmaydi:** wake word, STT (keyingi bosqichlar).

---

## 3. Texnik talablar

### 3.1. `AudioCapture` (`capture.py`)
- `sounddevice.InputStream`: 16 kHz, mono, `int16`, `blocksize=512` (32 ms). Windows'da WASAPI host API afzal. Qurilma 16 kHz'ni qo'llab-quvvatlamasa, 48 kHz'da ochib, `scipy.signal.resample_poly` bilan 16 kHz'ga tushiriladi.
- Callback **faqat** frame'ni thread-safe navbatga (`queue.Queue(maxsize=200)`) qo'yadi, boshqa hech narsa qilmaydi. Navbat to'lsa, eng eski frame tashlanadi va ogohlantirish log'ga yoziladi (sekundiga ko'pi bilan 1 marta).
- Asinxron interfeys: `async for frame in capture.frames(): ...` (ichida `loop.run_in_executor` yoki `asyncio.Queue` ko'prigi).
- `input_device`: `null` bo'lsa default qurilma, aks holda nomi bo'yicha qidiriladi (katta-kichik harf farqsiz, qisman moslik).
- **Chidamlilik:** stream xatosida (qurilma uzildi, laptop uyqudan uyg'ondi, naushnik ulandi) 1 s, 2 s, 5 s, 10 s backoff bilan qayta ochiladi. Qurilma ro'yxati har urinishda yangilanadi. 60 s davomida ulanib bo'lmasa, `AudioDeviceLost` hodisasi chiqariladi (tray'da ko'rinadi), lekin urinishlar davom etadi.
- `mute()` / `unmute()`: foydalanuvchi tray orqali mikrofonni o'chirishi mumkin (frame'lar tashlanadi).
- Statistika: RMS darajasi (dBFS), dropped frames soni.

### 3.2. `RingBuffer` (`ring_buffer.py`)
- Oxirgi `preroll_seconds` (2.0 s) audioni saqlaydi. Numpy asosidagi aylana bufer, nusxa ko'chirishlar minimal.
- `snapshot(seconds: float) -> np.ndarray`: oxirgi N soniya.
- Maqsad: gap boshlanishi kesilib qolmasligi ("Altron"ning "Al" qismi yo'qolmasin).

### 3.3. `SileroVAD` va `Endpointer` (`vad.py`)
- Silero VAD **ONNX** rejimida, CPU'da (`silero-vad` pip paketi `onnx=True` bilan, yoki modelni to'g'ridan-to'g'ri onnxruntime orqali). Model `data/models/vad/` papkasiga saqlanadi.
- Har 512 namunali frame uchun speech ehtimoli chiqadi. Model holati (state) oqim davomida saqlanadi va `reset()` bilan tozalanadi.
- `Endpointer` holat mashinasi: `SILENCE → SPEECH → TRAILING_SILENCE → emit Utterance`.
  - Gap boshlanishi: ketma-ket `min_speech_ms` davomida `prob > threshold`.
  - Gap tugashi: `end_silence_ms` (700 ms) davomida `prob < threshold - 0.15` (histerezis).
  - `max_utterance_s` (15 s) oshsa, majburan kesiladi, `reason="max_length"`.
  - Utterance'ga gap boshlanishidan oldingi **300 ms** pre-roll qo'shiladi.
- Ikki rejim:
  - `passive` (IDLE holatida): barcha gap segmentlari wake detektorga beriladi (stt_kws rejimi uchun).
  - `active` (LISTENING): wake'dan keyin buyruqni kutadi. `listen_timeout_s` ichida gap boshlanmasa, `ListenTimeout` hodisasi chiqadi.
- `end_silence_ms` runtime'da o'zgartirilishi mumkin (masalan tasdiqlash "ha/yo'q" uchun 400 ms).

### 3.4. `AudioPlayer` (`player.py`)
- `sounddevice.OutputStream` asosida, doim ochiq turadigan oqim. Har safar yangi stream ochilmaydi, aks holda 100–200 ms kechikish bo'ladi.
- `async play(pcm: np.ndarray, sr: int)`: kerak bo'lsa resample, `output_volume` bilan ko'paytirish, navbatga qo'yish.
- `enqueue()` va streaming: TTS gap-gap bo'lib yuboradi, player ularni ketma-ket, uzilishsiz chaladi.
- `stop()`: **50 ms ichida** ovozni to'xtatadi va navbatni tozalaydi (barge-in uchun). Qisqa fade-out (10 ms) bilan, "chirt" tovushisiz.
- `is_playing` xossasi va `on_playback_start/end` callback'lari. Orchestrator bular orqali echo himoyasini boshqaradi.
- Qurilma xatosida `AudioCapture` kabi qayta ulanadi.

### 3.5. `earcons.py`
Kod bilan generatsiya qilinadigan qisqa tovushlar (fayl kerak emas, `numpy` sinusoidlari va envelope):
| Nomi | Qachon | Tavsif |
|---|---|---|
| `wake` | Wake word tanildi | ~120 ms, ko'tariluvchi ikki nota |
| `ack` | Buyruq qabul qilindi, o'ylayapti | ~80 ms, yumshoq |
| `done` | Muloqot tugadi | ~120 ms, pasayuvchi |
| `error` | Xato | ~200 ms, past ton |
| `confirm` | Tasdiq kutilmoqda | ~150 ms |

### 3.6. `scripts/record_samples.py`
Konsol vositasi (keyingi bosqichlarda ham ishlatiladi):
```
uv run python scripts/record_samples.py mic-test              # 5 s yozib, RMS va VAD grafigini matnda ko'rsatadi
uv run python scripts/record_samples.py vad-live              # jonli: har bir gap segmenti vaqtini ko'rsatadi
uv run python scripts/record_samples.py wake --count 100      # 04 va 10-bosqich uchun "Altron" namunalari
uv run python scripts/record_samples.py testset --file prompts_uz.txt   # 03-bosqich uchun
```
`wake` va `testset` rejimlari: Enter → yozish → VAD gap tugashini aniqlaydi → `data/recordings/<tur>/NNNN.wav` fayliga saqlaydi. Har bir namunadan keyin "qayta yozish / keyingisi" tanlovi beriladi. `testset` rejimida transkriptlar `transcripts.tsv` fayliga yoziladi.

---

## 4. Testlar
- `test_ring_buffer.py`: to'lib toshish, snapshot uzunligi, tartib.
- `test_endpointer.py`: sintetik ma'lumot (jimlik + shovqin + "gap" sifatida VAD prob ketma-ketligi mock) bilan: start/end vaqtlari, pre-roll, max_length, histerezis.
- `test_vad_onnx.py` (hardware emas): 1 s jimlikda prob < 0.2, test wav'da (`tests/data/speech_uz.wav`, 03-bosqichdan oldin sintetik yoki yozilgan) prob > 0.5.
- `test_player.py` (`@hardware`): `stop()` 50 ms ichida qaytishi.
- `test_capture.py` (`@hardware`): 3 s ichida ≥ 90 frame keladi.

## 5. Qabul mezonlari
- [ ] `record_samples.py vad-live` 10 ta gapning har birini alohida segment sifatida aniqlaydi, gap boshi kesilmaydi
- [ ] 700 ms jimlikdan keyin segment yopiladi. So'zlar orasidagi qisqa pauzalarda (≤ 400 ms) bo'linib ketmaydi
- [ ] Ishlash vaqtida naushnikni uzib-ulash jarayonni yiqitmaydi, 10 s ichida qayta ulanadi
- [ ] Bo'sh holatda CPU yuklamasi ≤ 3% (Task Manager, 1 daqiqa o'rtachasi)
- [ ] `player.stop()` ≤ 50 ms
- [ ] Earcon'lar eshitiladi, chirtillamaydi
- [ ] Testlar o'tadi, `PROGRESS.md` yangilangan

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| Windows mikrofonda avtomatik "enhancement" (AGC/shovqin bostirish) | Ma'lumot uchun: Sound settings → Microphone → Audio enhancements. Default holatda qoldiriladi va natija solishtiriladi |
| Laptop uyqudan chiqqanda stream "osilib" qoladi | Watchdog: 2 s davomida frame kelmasa, stream qayta ochiladi |
| 48 kHz'dan resample CPU yuklaydi | `resample_poly(3,1)` arzon, muammo emas |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md va docs/tz/00_umumiy_arxitektura.md (§2, §5 audio/vad, §6 events) ni o'qi.
docs/tz/02_audio.md ni to'liq o'qi va faqat shu bosqichni bajar.
Hardware testlarini men qo'lda ishga tushiraman. Menga qaysi buyruqlarni ishga tushirishimni va nimaga e'tibor berishimni ayt.
```
