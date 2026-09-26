# TZ-08. Orchestrator: to'liq ovozli pipeline

**Bog'liqlik:** TZ-02 … TZ-07
**Natija:** `python -m ultron` → "Altron, ..." deysiz → Ultron tinglaydi, tushunadi, bajaradi va o'zbekcha gapiradi. Streaming, follow-up, barge-in, ovozli tasdiqlash va latency metrikalari ishlaydi.

> ⚠️ Bu bosqich **2 ta sessiyaga** bo'linadi: **8A** (asosiy sikl) → **8B** (streaming, barge-in, tasdiqlash, metrikalar).

---

## 1. Umumiy dizayn

### 1.1. Jarayon ichidagi tuzilma
```
main thread (asyncio loop)
 ├─ task: audio_loop        — capture.frames() → ring_buffer, vad, wake(frame)
 ├─ task: utterance_loop    — Endpointer'dan kelgan Utterance'larni holatga qarab yo'naltiradi
 ├─ task: orchestrator      — EventBus'dan hodisalarni o'qiydi, holat mashinasi
 ├─ task: scheduler         — eslatmalar (TZ-07D)
 ├─ task: health_monitor    — Ollama, audio, VRAM (har 30 s)
 └─ thread: tray (TZ-09)
audio callback thread (sounddevice) → queue → audio_loop
STT/TTS/tool'lar → asyncio.to_thread (GPU/CPU pool)
```

### 1.2. EventBus (`events.py`)
- `asyncio.Queue` asosidagi oddiy pub/sub: `publish(event)`, `subscribe(type) -> AsyncIterator`.
- Hodisalar: `WakeEvent`, `UtteranceEvent`, `ListenTimeout`, `TranscriptEvent`, `AgentEvent`lar, `PlaybackStarted/Ended`, `ReminderDue`, `AudioDeviceLost/Restored`, `LLMUnavailable/Restored`, `UserCommand` (tray'dan: mute, restart, quit).

### 1.3. Holatlar (`core/state.py`)
`IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `FOLLOW_UP`, `CONFIRMING`, `MUTED`, `ERROR`. Holat o'zgarishlari faqat `Orchestrator._transition(new, reason)` orqali bo'ladi va log'ga yoziladi. Tray ikonkasi holatga qarab o'zgaradi (TZ-09).

---

## 8A. Asosiy sikl

### 8A.1. Ishga tushish ketma-ketligi (`Orchestrator.start()`)
1. Config va log → 2. `_cuda_dlls` → 3. Parallel yuklash: STT (GPU), TTS (CPU), VAD, Wake, xotira DB, app indeksi → 4. Ollama `health()` + `ensure_loaded()` (Ollama ishlamayotgan bo'lsa, TZ-09 dagi avtostart logikasi) → 5. Scheduler → 6. Audio capture va player → 7. `announce_ready` bo'lsa: "Tayyorman" (keshlangan ibora).
- Har bir qadamning davomiyligi log'ga yoziladi. Maqsad: jami ≤ 40 s (sovuq start, modellar diskda).
- Biror komponent yuklanmasa: kritik bo'lsa (audio, STT) → `ERROR` holati, har 30 s qayta urinish, tray'da qizil ikonka; kritik bo'lmasa (TTS) → ishlashda davom etadi (javob toast bilan ko'rsatiladi).

### 8A.2. Oqim
```
IDLE:
  wake.mode=stt_kws → har bir passive Utterance → wake.process_utterance()
  wake.mode=oww/hybrid → har bir frame → wake.process_frame()
  WakeEvent:
     trailing_text bor va ≥ 2 so'z → earcon "ack" → THINKING(trailing_text)
     aks holda → earcon "wake" → LISTENING (vad active rejimi)
LISTENING:
  Utterance → STT → Transcript
     bo'sh / past ishonch (< 0.35) → "Tushunmadim, qaytaring" → LISTENING (1 marta), keyin IDLE
     cancel_phrases → earcon "done" → IDLE
     aks holda → THINKING(text)
  ListenTimeout → earcon "done" → IDLE
THINKING:
  agent.respond(text) → gaplar → SPEAKING (TTS navbati)
SPEAKING:
  playback tugadi → FOLLOW_UP
FOLLOW_UP (follow_up_window_s = 8):
  wake word'siz Utterance → STT →
     end_phrases → "Xizmatingizdaman" (ixtiyoriy) → IDLE
     boshqa → THINKING
  timeout → IDLE (ovozsiz)
```
- **FOLLOW_UP'da yolg'on trigger'ni kamaytirish:** Utterance ≥ 0.5 s va STT confidence ≥ 0.45 bo'lishi kerak. Aks holda e'tiborsiz qoldiriladi.
- IDLE'ga qaytishda agent tarixi saqlanadi (5 daqiqa ichida qaytsa kontekst davom etadi, TZ-06 §3.3).

### 8A.3. Xatolarga chidamlilik
- Har bir holat handler'i `try/except` ichida. Xato → log (traceback) → earcon "error" + "Kechirasiz, xatolik yuz berdi" → IDLE.
- Ketma-ket 5 ta xato 1 daqiqa ichida → komponentlarni qayta yuklash (`soft_restart`). Bu ham yordam bermasa → `exit(3)` (supervisor qayta ishga tushiradi, TZ-09).

### 8A.4. Qabul (8A)
- [ ] "Altron" → ding → "Telegramni och" → Telegram ochiladi, "Telegram ochildi" deydi
- [ ] "Altron, soat nechi?" (bir nafasda) → darhol javob
- [ ] Javobdan keyin 8 s ichida wake word'siz "ovozni balandlat" → bajariladi
- [ ] "Altron" → 6 s jimlik → "done" earcon → IDLE
- [ ] 30 daqiqa uzluksiz ishlash, xotira oqishi yo'q (RSS o'sishi < 100 MB)

---

## 8B. Streaming, barge-in, tasdiqlash, metrikalar

### 8B.1. Streaming (latency'ning asosi)
- `agent.respond()` dan kelgan birinchi `SentenceEvent` → darhol TTS → player. Keyingi gaplar sintezi parallel (TZ-05 §3.4).
- `ToolStartEvent` → agar 400 ms ichida hech qanday gap aytilmagan bo'lsa, `ack` earcon chalinadi (foydalanuvchi "eshitdi" ekanini bilsin).
- Tool 2 s dan uzoq ishlasa → keshlangan "Bir daqiqa..." iborasi aytiladi.

### 8B.2. Barge-in (gapini bo'lish)
- `SPEAKING` holatida wake detektor oshirilgan threshold bilan ishlaydi (TZ-04 §3.5).
- `WakeEvent` → `player.stop()` + `tts.cancel()` + `agent_task.cancel()` (Ollama so'rovi ham to'xtaydi) → earcon "wake" → LISTENING (yoki trailing_text bo'lsa → THINKING).
- `cancel_phrases` ("to'xta", "jim") SPEAKING paytida ham ishlaydi, lekin faqat wake word bilan: "Altron, to'xta".
- **Echo himoyasi:** player ijrosi paytida Endpointer passive segmentlari `during_playback=True` bilan belgilanadi. `stt_kws` ular uchun `fuzzy_threshold + 8` talab qiladi. Naushnik aniqlansa (qurilma nomida "headset/headphones/naushnik" so'zlari bo'lsa), bu himoya yumshatiladi.

### 8B.3. Ovozli tasdiqlash (`core/confirm.py`)
- `VoiceConfirmationProvider.confirm(question_uz) -> bool`:
  1. Holat → `CONFIRMING`; `confirm` earcon + TTS savol ("Kompyuterni o'chirishni tasdiqlaysizmi? Ha yoki yo'q deng.").
  2. VAD `end_silence_ms=400`, `confirm_timeout_s` (8 s) kutadi. Wake word **shart emas**.
  3. STT → ha-lug'ati: `ha, xa, ha ha, albatta, tasdiqlayman, bo'pti, mayli, ok, davay, да` → `True`; yo'q-lug'ati: `yo'q, yoq, kerak emas, bekor, to'xta, нет` → `False`. Noma'lum → bir marta qayta so'raydi → baribir noma'lum → `False`.
  4. Timeout → `False` + "Bekor qilindi".
- Text rejimida (`--text`) konsol varianti ishlatiladi (TZ-06).

### 8B.4. Eslatmalar integratsiyasi
- `ReminderDue` IDLE/FOLLOW_UP holatida → darhol aytiladi. SPEAKING/THINKING holatida → navbatga qo'yiladi va joriy javobdan keyin aytiladi. LISTENING holatida → buyruq tugashi kutiladi.

### 8B.5. Metrikalar (`core/metrics.py`)
Har bir so'rov uchun bitta JSON qator `data/logs/metrics.jsonl` fayliga yoziladi:
```json
{"ts": "...", "wake_source": "stt_kws", "wake_ms": 310, "vad_end_to_stt_ms": 420, "stt_to_first_token_ms": 380,
 "first_token_to_first_audio_ms": 260, "total_to_first_audio_ms": 1370, "path": "llm|fastpath", "tools": ["open_app"],
 "stt_text": "telegramni och", "confidence": 0.78, "error": null}
```
- `scripts/metrics_report.py`: oxirgi N kun bo'yicha p50/p95 va eng sekin 10 ta so'rov.

### 8B.6. `--debug` rejimi
Konsolda jonli holat satri: `[IDLE] mic -42 dBFS | VAD 0.03 | VRAM 7.1/8.0 GB | last: 1.37 s`.

### 8B.7. Qabul (8B)
- [ ] Gap tugashidan birinchi tovushgacha: **fast-path p95 ≤ 1.0 s**, **LLM yo'li p95 ≤ 2.0 s** (`metrics_report.py`, kamida 50 ta so'rov)
- [ ] Uzun javob paytida "Altron" → ≤ 300 ms ichida jim bo'ladi va tinglaydi
- [ ] Ultron o'z ovozidan uyg'onib ketmaydi (laptop karnayida 20 ta uzun javob → 0 ta self-trigger)
- [ ] "Yuklamalardagi oxirgi faylni o'chir" → ovozli tasdiq → "ha" → Recycle Bin; "yo'q" / jimlik → bekor
- [ ] Eslatma javob paytida kelsa, javobdan keyin aytiladi
- [ ] 24 soatlik "soak test": har 10 daqiqada avtomatik skript (`scripts/soak_test.py`, oldindan yozilgan wav'larni virtual mikrofon o'rniga to'g'ridan-to'g'ri pipeline'ga beradi) → qulash yo'q, RSS o'sishi < 200 MB, VRAM barqaror
- [ ] `PROGRESS.md` yangilangan

---

## 2. Testlar
- `test_orchestrator_fsm.py`: barcha holat o'tishlari mock komponentlar bilan (wake → listening → thinking → speaking → follow_up → idle; timeout'lar; cancel; barge-in; confirm ha/yo'q/timeout).
- `test_confirm.py`: lug'atlar va noma'lum javob.
- `test_pipeline_offline.py`: `tests/data/*.wav` fayllarini to'g'ridan-to'g'ri `utterance_loop`ga berib, mock tool'lar bilan end-to-end (hardware talab qilmaydi, faqat modellar).

## 3. Claude Code uchun promptlar
```
[8A] CLAUDE.md, docs/tz/00_umumiy_arxitektura.md (§2, §6) va docs/tz/08_orchestrator.md ning §1 va "8A" qismini o'qi.
     Faqat 8A ni bajar. Oldingi modullarning public interfeysini o'zgartirma. Kerak bo'lsa adapter yoz.
[8B] CLAUDE.md va docs/tz/08_orchestrator.md ning §1, "8B" va §2 qismini o'qi. Faqat 8B ni bajar.
     Oxirida metrics_report natijasini ko'rsat.
```
