# TZ-10. "Altron" uchun maxsus wake word modelini o'qitish (openWakeWord)

**Bog'liqlik:** TZ-04 (interfeys, benchmark), TZ-09 (barqaror tizim)
**Qachon:** MVP (1–9) tugab, kamida 1 hafta real foydalanishdan keyin. Shu vaqtda sizning ovozingizdagi real namunalar yig'iladi.
**Natija:** `data/models/wake/altron.onnx` (+ shaxsiy verifier), `wake.mode: hybrid`. Tezroq (so'z aytilishi bilan) va GPU'ni kamroq band qiladigan detektor.

---

## 1. Nega kerak
`stt_kws` ishonchli, lekin: (1) gap **tugagandan keyin** reaksiya beradi (~0.3–0.8 s kechikish), (2) atrofdagi har bir gapni GPU'da transkripsiya qiladi. openWakeWord modeli CPU'da 80 ms'lik bo'laklarni baholaydi va so'z aytilishi bilan trigger beradi. `hybrid` rejimda ikkalasi birga ishlaydi, shuning uchun ishonchlilik pasaymaydi.

## 2. Qamrov
**Kiradi:** WSL2 o'qitish muhiti, sintetik ma'lumot generatsiyasi, real namunalarni tayyorlash, o'qitish, ONNX eksport, shaxsiy verifier, benchmark, `hybrid` rejimini sozlash.
**Kirmaydi:** runtime kodini qayta yozish (TZ-04 da tayyor).

---

## 3. Texnik talablar

### 3.1. O'qitish muhiti
- openWakeWord o'qitish pipeline'i **Linux**'ga mo'ljallangan. Shuning uchun **WSL2 + Ubuntu** (lokal, GPU bilan) ishlatiladi:
  ```powershell
  wsl --install -d Ubuntu-24.04
  ```
  WSL ichida `nvidia-smi` ishlashi tekshiriladi (Windows drayveri WSL'ga CUDA beradi).
- Alohida papka: `C:\ultron\training\wake\` (WSL'dan `/mnt/c/ultron/training/wake`). Runtime `.venv` ga tegilmaydi.
- Claude Code **openWakeWord rasmiy repozitoriyasidagi o'qitish hujjati va notebook'iga** (automatic model training) amal qiladi. Buyruq va parametrlar kutubxona versiyasiga qarab o'zgarishi mumkin, shuning uchun rasmiy hujjat asosiy manba hisoblanadi, bu TZ esa talablarni belgilaydi.
- Disk: o'qitish uchun negativ feature'lar va shovqin datasetlari ~20–40 GB joy oladi.

### 3.2. Ma'lumotlar
**Pozitiv (sintetik):**
- Piper TTS asosidagi sample generator (openWakeWord hujjatida ko'rsatilgan) bilan **kamida 20 000** klip.
- Matn variantlari (inglizcha fonetika bilan o'zbekcha talaffuzga yaqinlashtirish): `"altron"`, `"al tron"`, `"ultron"`, `"oltron"`, `"altrun"`, `"ahl-tron"`. Har xil speaker, tezlik, pitch.
- Augmentatsiya: room impulse response'lar, fon shovqini (musiqa, gap, ko'cha), SNR 0–20 dB, turli balandliklar.

**Pozitiv (real, eng qimmatli):**
- `data/recordings/wake_pos/` (TZ-04 da 100 ta) + real foydalanish davomida saqlangan "Altron" segmentlari (`memory.save_utterances=True` bo'lsa, wake trigger bergan segmentlarning boshidagi 1.5 s avtomatik `data/recordings/wake_auto/` ga saqlanadi. **Bu TZ-10 uchun TZ-08 ga qo'shiladigan kichik o'zgarish**).
- Kamida **300** ta real namuna: turli masofa (0.5, 1, 2, 3 m), ovoz balandligi, xonalar, naushnik va laptop mikrofoni. `record_samples.py wake` orqali to'ldiriladi.

**Negativ:**
- openWakeWord taqdim etadigan umumiy negativ feature to'plamlari (rasmiy hujjat bo'yicha).
- **Shaxsiy negativ:** 2+ soat o'z xonangizdagi fon (o'zingizning oddiy gapingiz, oila a'zolari, TV, YouTube, musiqa) → `data/recordings/wake_neg/`.
- **Qiyin negativlar** (o'xshash so'zlar, har biri 20+ marta yoziladi): "elektron", "patron", "Aldar", "alt", "atrof", "Ultra", "kontrol", "altin".

### 3.3. O'qitish
- Model: openWakeWord standart arxitekturasi (umumiy embedding ustida kichik klassifikator).
- Ikki model versiyasi o'qitiladi va solishtiriladi: (a) faqat sintetik, (b) sintetik + real (real namunalar oversampling ×10).
- Target false accept: ≤ 0.5/soat, "qiyin negativ"larda ≤ 5%.
- Chiqish: `altron_v1.onnx`, `altron_v2.onnx` → `data/models/wake/`.

### 3.4. Shaxsiy verifier
- openWakeWord "custom verifier" imkoniyati (foydalanuvchi ovozi bo'yicha kichik logistik regressiya) bilan real pozitiv va shaxsiy negativlarda o'qitiladi → `altron_verifier.pkl`.
- Maqsad: TV yoki boshqa odamlar aytgan "Ultron" ga javob bermaslik. **Ixtiyoriy:** agar oiladagi boshqa odamlar ham Ultron'dan foydalanishi kerak bo'lsa, verifier o'chiriladi.

### 3.5. Benchmark va tanlash
`scripts/benchmark_wake.py` (TZ-04) har uch rejim uchun:
| Rejim | Recall | FA/soat | Latency p95 (so'z oxiridan) | CPU % | GPU band vaqti |
|---|---|---|---|---|---|
| stt_kws | | | | | |
| oww (v1/v2, 3 xil threshold) | | | | | |
| hybrid | | | | | |

**Qabul qilish qoidasi:** `hybrid` rejimi `stt_kws`dan **recall bo'yicha yomon bo'lmasligi** (≥ stt_kws − 1%) va latency p95 ≤ 250 ms bo'lishi kerak. Aks holda `stt_kws` default bo'lib qoladi.

---

## 4. Qabul mezonlari
- [ ] O'qitish muhiti WSL2'da hujjatlashtirilgan (`training/wake/README.md`, qadamlar takrorlanadigan)
- [ ] ≥ 300 real pozitiv, ≥ 2 soat shaxsiy negativ yig'ilgan
- [ ] Benchmark jadvali to'ldirilgan, hisobot `data/reports/`da
- [ ] Tanlangan rejim: recall ≥ 97%, FA ≤ 0.5/soat, latency p95 ≤ 250 ms (hybrid uchun)
- [ ] `config.yaml` yangilangan, 48 soat real foydalanishda regressiya yo'q
- [ ] `PROGRESS.md` yangilangan

## 5. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/04_wake_word.md (§3.3–3.6) va docs/tz/10_wake_word_training.md ni o'qi.
Avval openWakeWord'ning joriy rasmiy o'qitish hujjatini tekshir va TZ bilan farqlarini ayt.
Keyin WSL2 muhitini tayyorlash qadamlarini ber, men bajaraman. Keyin ma'lumot tayyorlash va o'qitishni bosqichma-bosqich olib bor.
```
