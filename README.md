# ULTRON — Shaxsiy lokal ovozli AI yordamchi

## Texnik topshiriq (TZ) to'plami

Bu to'plam Ultron loyihasini **Claude Code** yordamida bosqichma-bosqich qurish uchun yozilgan. Har bir bosqich alohida faylda. Shuning uchun Claude Code'ga har safar faqat kerakli qism beriladi. Bu limitni tejaydi va kontekstni toza saqlaydi.

> **Nom:** yordamchining nomi **Ultron**. Siz uni o'zbekcha talaffuzda **"Altron"** deb chaqirasiz. Tizim ikkala talaffuzni ham ("Ultron", "Altron", "Oltron", "Ультрон", "Алтрон") tanishi shart.

---

## Asosiy talablar (qisqacha)

| # | Talab |
|---|---|
| 1 | **Hammasi lokal.** STT, LLM, TTS va wake word laptopda ishlaydi. Bulutdagi AI API ishlatilmaydi |
| 2 | **LLM:** Ollama orqali Qwen oilasining 7–9B modeli |
| 3 | **Til:** siz o'zbekcha gapirasiz, Ultron o'zbekcha javob beradi |
| 4 | **Doimiy tinglash.** "Altron" deyilganda **har doim** javob berishi shart (asosiy KPI) |
| 5 | **Avtostart.** Windows 11'ga kirilganda hammasi o'zi ishga tushadi |
| 6 | **MVP amallar:** tizim boshqaruvi, fayllar, brauzer/internet, xotira va eslatmalar |

**Apparat:** RTX 4060 Laptop 8 GB VRAM · Ryzen 7 7435HS · 24 GB RAM · 1 TB SSD · Windows 11

---

## Fayllar va bajarish tartibi

| Tartib | Fayl | Nima qilinadi | Taxminiy hajm |
|---|---|---|---|
| — | `CLAUDE.md` | Repo ildiziga qo'yiladi. Claude Code har sessiyada avtomatik o'qiydi | — |
| 0 | `docs/tz/00_umumiy_arxitektura.md` | Arxitektura, papkalar, config, interfeyslar. **Kod yozilmaydi**, faqat ma'lumotnoma | — |
| 1 | `docs/tz/01_muhit_sozlash.md` | Windows muhiti, CUDA, Ollama, Python, loyiha skeleti | Kichik |
| 2 | `docs/tz/02_audio.md` | Mikrofon, VAD, ovoz chiqarish, earcon'lar | O'rta |
| 3 | `docs/tz/03_stt.md` | faster-whisper, o'zbek matnini normallashtirish, benchmark | O'rta |
| 4 | `docs/tz/04_wake_word.md` | "Altron" detektori (STT asosida + openWakeWord interfeysi) | O'rta |
| 5 | `docs/tz/05_tts.md` | O'zbekcha TTS, transliteratsiya, raqamlarni so'zga aylantirish | O'rta |
| 6 | `docs/tz/06_llm.md` | Qwen/Ollama, tool calling sikli, system prompt, text rejim | O'rta |
| 7 | `docs/tz/07_tools.md` | Tizim, fayl, brauzer, xotira va eslatma tool'lari | Katta |
| 8 | `docs/tz/08_orchestrator.md` | Hammasini bitta pipeline'ga ulash, streaming, barge-in | Katta |
| 9 | `docs/tz/09_autostart.md` | Avtostart, tray, watchdog, log'lar, barqarorlik | O'rta |
| 10 | `docs/tz/10_wake_word_training.md` | "Altron" uchun maxsus openWakeWord modelini o'qitish | O'rta (keyinroq) |
| 11 | `docs/tz/11_uzbek_stt_finetune.md` | O'zbekcha STT sifatini oshirish (dataset + LoRA) | Katta (keyinroq) |

**1–9-bosqichlar MVP hisoblanadi.** 9-bosqichdan keyin Ultron to'liq ishlaydi. 10 va 11-bosqichlar sifatni oshirish uchun, keyinroq bajariladi.

---

## Claude Code bilan ishlash tartibi (limitni tejash)

1. Repo yarating (`C:\ultron`) va `CLAUDE.md` faylini ildizga, `docs/tz/` papkasini esa ichiga ko'chiring.
2. Har bir bosqichni **yangi sessiyada** boshlang (yoki oldingi bosqichdan keyin `/clear` qiling).
3. Claude Code'ga quyidagi shablonni bering (har bir TZ faylining oxirida tayyor prompt ham bor):

```
CLAUDE.md va docs/tz/00_umumiy_arxitektura.md ni o'qi (faqat kerakli bo'limlarini).
Keyin docs/tz/0X_....md ni to'liq o'qi va FAQAT shu bosqichni bajar.
Boshqa bosqichlarning kodini o'zgartirma (faqat TZ'da aytilgan integratsiya nuqtalari bundan mustasno).
Oxirida "Qabul mezonlari" bo'limidagi har bir bandni tekshir va natijani jadval ko'rinishida hisobot qil.
```

4. Bosqich tugagach: testlar o'tganini tekshiring → `git commit` qiling → `docs/PROGRESS.md` ga bir qator yozing (Claude Code buni o'zi qiladi).
5. Agar bosqich juda katta bo'lsa (7 va 8), uni TZ ichida ko'rsatilgan **kichik qismlarga** (7A, 7B...) bo'lib, har birini alohida sessiyada bajaring.

---

## Avtostart va parolsiz kirish (ixtiyoriy)

`scripts/install_autostart.ps1` Windows Task Scheduler'ga "Ultron" vazifasini qo'shadi (login paytida + workstation unlock'da, admin **shart emas**) — batafsil: `docs/tz/09_autostart.md`, qo'lda tekshiruvlar: `docs/manual_tests_09.md`.

Bu login ekranini o'zi o'tkazib yubormaydi — laptop yoqilgach baribir parol kiritish kerak bo'ladi, shundan keyingina Ultron avtomatik ishga tushadi. Agar login ekranisiz, laptop yoqilishi bilanoq Ultron ishlashini xohlasangiz, Windows'ning **avtomatik kirish** (autologon) imkoniyatidan foydalanish mumkin:

- `netplwiz` (Boshlash menyusidan ishga tushirish) → "Users must enter a password..." belgisini olib tashlash, yoki
- Sysinternals **Autologon** (`https://learn.microsoft.com/sysinternals/downloads/autologon`) — parolni shifrlangan holda saqlaydi, `netplwiz`ga qaraganda xavfsizroq.

**Muhim:** bu laptop **xavfsizligini pasaytiradi** — kompyuterni har kim yoqib, hech qanday parolsiz Windows sessiyasiga kira oladi. TZ bunga majburlamaydi; qaror to'liq foydalanuvchi zimmasida.

---

## Muhim ochiq eslatmalar

- **O'zbek tili — eng katta xavf.** Whisper va Qwen o'zbekchada ingliz tilidagidek kuchli emas. Shuning uchun TZ'da benchmark'lar, normallashtirish qatlamlari va "fast-path" (LLM'siz tez buyruqlar) qo'shilgan. 11-bosqich aynan shu muammoni hal qilish uchun.
- **"Har doim javob berishi shart" talabi** 4-bosqichda ikki bosqichli detektor bilan, 10-bosqichda esa maxsus o'qitilgan model bilan ta'minlanadi. Qabul mezoni: 100 ta chaqiruvdan kamida 95 tasiga javob berishi, soatiga 1 tadan kam yolg'on uyg'onish.
- **Naushnik (headset) tavsiya qilinadi.** Laptop karnayidan chiqqan Ultron ovozi mikrofonga qaytib tushadi (echo). TZ'da bunga qarshi himoya bor, lekin naushnik bilan barge-in ancha yaxshi ishlaydi.
