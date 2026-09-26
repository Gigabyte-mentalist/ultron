# TZ-11. O'zbekcha STT sifatini oshirish: shaxsiy dataset + LoRA fine-tune

**Bog'liqlik:** TZ-03 (benchmark), TZ-08 (utterance saqlash)
**Qachon:** MVP'dan keyin, 2–4 hafta real foydalanishdan so'ng (yoki TZ-03 da WER > 25% chiqsa, ertaroq).
**Natija:** sizning ovozingiz, mikrofoningiz va buyruqlaringizga moslashtirilgan Whisper modeli (CTranslate2 formatida). Maqsad WER ≤ 12%.

---

## 1. Nega kerak
Umumiy o'zbekcha modellar yangiliklar yoki audiokitoblar kabi ma'lumotlarda o'qitilgan. Sizning buyruqlaringizda ilova nomlari, texnik atamalar, rus va ingliz so'zlari va o'ziga xos talaffuz bor. Bir necha soatlik **shaxsiy** ma'lumot bilan LoRA fine-tune WER'ni sezilarli pasaytiradi. AI engineer sifatida siz uchun bu loyihaning eng qiziq qismi bo'ladi.

## 2. Qamrov
**Kiradi:** ma'lumot yig'ish pipeline'i, transkriptlarni tuzatish vositasi, ochiq datasetlarni qo'shish, LoRA o'qitish, baholash, CT2 eksport, A/B tanlash.
**Kirmaydi:** noldan model o'qitish.

---

## 3. Texnik talablar

### 3.1. Ma'lumot yig'ish (runtime'ga qo'shimcha, TZ-08 integratsiyasi)
- `memory.save_utterances=True` bo'lganda har bir LISTENING/FOLLOW_UP utterance'i saqlanadi: `data/recordings/utterances/YYYY-MM-DD/<uuid>.wav` + `index.jsonl` (`stt_text`, `confidence`, `model`, `tool_called`, `tool_ok`).
- **Avtomatik "kumush" belgi:** tool muvaffaqiyatli bajarilgan va foydalanuvchi 10 s ichida qayta aytmagan yoki bekor qilmagan bo'lsa → transkript ehtimol to'g'ri (`label=silver`).
- Diskni tejash: 16 kHz mono FLAC, 60 kundan eski va tasdiqlanmagan yozuvlar o'chiriladi.

### 3.2. Tuzatish vositasi (`scripts/label_tool.py`)
- Oddiy lokal web UI (Gradio yoki FastAPI + oddiy HTML, `127.0.0.1` da): audio pleer, STT matni, tahrirlash maydoni, tugmalar: ✅ to'g'ri / ✏️ tuzatildi / 🗑 tashlash, klaviatura yorliqlari.
- Ustuvorlik bo'yicha tartib: avval past `confidence` va muvaffaqiyatsiz tool chaqiruvlari.
- Natija: `data/datasets/personal/{train,dev,test}.jsonl` (spiker bitta bo'lgani uchun **sana bo'yicha** bo'linadi, masalan oxirgi hafta = test, ma'lumot oqishining oldini olish uchun).
- Maqsadli hajm: **3–5 soat** tasdiqlangan audio (~3000–5000 ta buyruq). Birinchi tajriba uchun 1 soat ham yetarli.

### 3.3. Ochiq datasetlar (umumiy bilimni saqlab qolish uchun aralashtiriladi)
- Mozilla Common Voice (uz), Google FLEURS (uz_uz) va boshqa ochiq o'zbekcha nutq korpuslari (litsenziya tekshiriladi).
- Aralashma nisbati: shaxsiy 50% / ochiq 50% (catastrophic forgetting'ga qarshi). Tajriba orqali sozlanadi.
- Barcha transkriptlar TZ-03 dagi `normalize.py` orqali bir xil ko'rinishga keltiriladi (lotin, apostroflar).

### 3.4. O'qitish (8 GB VRAM uchun)
- Muhit: alohida venv yoki WSL2 (`training/stt/`). `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes` (WSL'da), `evaluate`, `jiwer`.
- Asosiy model: TZ-03 benchmark g'olibi (HF formatidagi asli). Agar u large-v3-turbo bo'lsa:
  - LoRA (r=16–32, alpha=32–64) `q_proj, k_proj, v_proj, out_proj` (encoder + decoder), yoki faqat decoder + encoder'ning yuqori qatlamlari.
  - fp16, **gradient checkpointing**, `per_device_batch_size=2–4`, `gradient_accumulation=8–16`, 8-bit AdamW.
  - VRAM tor bo'lsa → 8-bit asosiy model (QLoRA uslubida) yoki `medium`/`small` asosga o'tish.
- Hiperparametrlar: lr 1e-4 (LoRA), 3–5 epoch, warmup 10%, har 200 qadamda dev WER bilan baholash, eng yaxshi checkpoint saqlanadi.
- **Ultron o'qitish paytida:** tray'dagi "GPU'ni bo'shatish" rejimi (TZ-09 §3.7) yoqiladi, aks holda VRAM yetmaydi.

### 3.5. Baholash va eksport
- Test to'plamlari: (1) TZ-03 dagi 80 gaplik to'plam, (2) shaxsiy test split, (3) FLEURS uz test (umumiy sifat pasaymaganini tekshirish uchun).
- Metrikalar: WER, CER, "Altron" tanish ulushi, ilova nomlari aniqligi.
- LoRA merge → `ct2-transformers-converter --quantization int8_float16` → `data/models/stt/ultron-uz-vN-ct2/`.
- `benchmark_stt.py` orqali eski model bilan solishtiriladi. **Yangi model faqat (1) va (2) da yaxshiroq, (3) da esa 3 punktdan ko'p yomonlashmagan bo'lsa** qabul qilinadi.
- Har bir versiya uchun model kartasi: `data/models/stt/ultron-uz-vN-ct2/MODEL_CARD.md` (ma'lumot hajmi, hiperparametrlar, metrikalar).

---

## 4. Qabul mezonlari
- [ ] Utterance yig'ish va `label_tool.py` ishlaydi
- [ ] ≥ 1 soat tasdiqlangan shaxsiy ma'lumot (birinchi iteratsiya)
- [ ] LoRA o'qitish 8 GB VRAM'da OOM'siz tugaydi, skript qayta ishga tushirsa bo'ladigan (config fayl + seed)
- [ ] Shaxsiy test'da WER kamida **30% nisbiy** pasaygan (masalan 20% → 14%)
- [ ] FLEURS'da regressiya ≤ 3 punkt
- [ ] Yangi model config'ga ulangan, latency TZ-03 talablariga mos
- [ ] `PROGRESS.md` yangilangan

## 5. Claude Code uchun prompt
```
CLAUDE.md, docs/tz/03_stt.md (§3.4–3.5) va docs/tz/11_uzbek_stt_finetune.md ni o'qi.
Bu bosqichni 3 ta sessiyada qilamiz: (1) §3.1–3.2 yig'ish va tuzatish vositasi; (2) §3.3–3.4 dataset va o'qitish skripti;
(3) §3.5 baholash va eksport. Hozir faqat (1) ni bajar.
```
