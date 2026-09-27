# Qo'lda tekshiruvlar — 09-TZ (Avtostart, tray, watchdog, barqarorlik)

Bu ro'yxat docs/tz/09_autostart.md §5 "Qabul mezonlari"ning har bir bandiga mos keladi. Avtomatik testlar (`pytest tests/test_supervisor.py tests/test_single_instance.py`) `RestartPolicy`/mutex mantig'ini haqiqiy jarayon/GPU/audio'siz tekshiradi; bu ro'yxat esa **haqiqiy laptop**da qo'lda tasdiqlash uchun.

O'rnatish (birinchi marta, admin **shart emas**):

```powershell
powershell -ExecutionPolicy Bypass -File C:\ultron\scripts\install_autostart.ps1
```

Tozalash (hammasini o'chirish):

```powershell
powershell -ExecutionPolicy Bypass -File C:\ultron\scripts\uninstall_autostart.ps1
```

---

## Qabul mezonlari

- [ ] **Avtostart + tezlik.** `install_autostart.ps1` ishga tushirilgach, laptopni qayta yoqing (yoki `shutdown /r`), login qiling. **≤ 90 soniya** ichida Ultron "Tayyorman" deb aytadi (yoki TTS ishlamasa toast ko'rsatiladi), **hech qanday konsol oynasi ko'rinmaydi** (Task Manager'da faqat `pythonw.exe` ikkita jarayon — biri supervisor, biri asosiy — ko'rinishi kerak, konsol oynasi yo'q).
- [ ] **Majburiy o'chirish → tez tiklanish.** Task Manager'da asosiy `pythonw.exe` jarayonini (supervisor emas, ikkalasidan kattarog'i — `wmic process where "name='pythonw.exe'" get processid,commandline` bilan aniqlash mumkin) "End task" qiling. **≤ 15 soniya** ichida qayta ishga tushishi kerak (tray ikonkasi yo'qolib, qayta paydo bo'ladi).
- [ ] **Ollama qulashi → o'zi tiklanadi.** Ollama'ni to'xtating (`taskkill /IM ollama.exe /F` yoki tray'idan chiqish). Ultron tray ikonkasi **qizil**ga o'tishi kerak. **≤ 90 soniya** ichida o'zi Ollama'ni qayta ishga tushirib, tray yana normal rangga qaytishi kerak (health monitor birinchi muvaffaqiyatsiz tekshiruvdayoq `ollama serve`ni qayta ishga tushiradi).
- [ ] **Uyqu/uyg'onish.** Laptopni uyquga yuboring (Boshlash menyusi yoki `rundll32.exe powrprof.dll,SetSuspendState 0,1,0`), keyin uyg'oting. **≤ 10 soniya** ichida "Altron" chaqirig'iga javob berishi kerak.
- [ ] **Batareya.** Zaryadlovchini ajratib, batareyada ishlab turganda ham Ultron ishlashda davom etadi (avtostart sozlamalarida "faqat AC quvvatida" cheklovi o'chirilgan).
- [ ] **Tray menyusi.** Har bir band ishlaydi:
  - Mikrofonni o'chirish/yoqish — ikkalasi ham ishlaydi, ikonka rangi mosga o'zgaradi (kulrang = MUTED).
  - Matnli buyruq... — kichik oyna ochiladi, matn yozib "Yuborish" bosilganda (yoki Enter) Ultron buyruqni bajaradi/javob beradi.
  - Log papkasini ochish — Explorer `data/logs/` papkasini ochadi.
  - Config'ni ochish — `config/config.yaml` fayli ochiladi.
  - GPU'ni bo'shatish / Normal rejim — bosilganda log'da "VRAM'dan tushirilmoqda"/"qayta yuklanmoqda" yozuvi ko'rinadi (`ollama ps` bilan tasdiqlash mumkin).
  - Qayta ishga tushirish — jarayon darhol (backoff'siz) qayta ishga tushadi.
  - Chiqish — jarayon (supervisor ham) butunlay to'xtaydi, tray ikonkasi yo'qoladi.
- [ ] **Naushnik ulash/uzish.** Ishlab turganda naushnikni ulang/uzing — audio qayta ulanish avtomatik ishlaydi (02-TZ), Ultron ishlashda davom etadi.
- [ ] **7 kunlik uzluksiz ishlash.** Qo'lda aralashuv talab qilinmasligi kerak. Kuzatuv natijasi keyinroq `docs/PROGRESS.md`ga yoziladi.
- [ ] **Tozalash.** `uninstall_autostart.ps1` ishga tushirilgach, `Get-ScheduledTask -TaskName Ultron` "topilmadi" xatosi bilan qaytishi kerak.

## Qo'shimcha (ixtiyoriy, TZ tavsiyasi)

- [ ] Global hotkey (`Ctrl+Alt+U`, agar `autostart.hotkey_enabled: true`) wake word ishlamay qolganda ham LISTENING holatiga o'tkazadi.
- [ ] Parolsiz avtomatik kirish (README.md'dagi yo'riqnoma) — foydalanuvchi xohishiga ko'ra, xavfsizlikni pasaytirishini tushunib.
