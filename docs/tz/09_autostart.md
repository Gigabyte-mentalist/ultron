# TZ-09. Avtostart, tray, watchdog va barqarorlik

**Bog'liqlik:** TZ-08
**Natija:** laptop yoqilib Windows'ga kirilgach, Ultron **o'zi** ishga tushadi, konsol oynasi ko'rinmaydi, tray'da holat ko'rinadi. Qulasa, o'zi qayta turadi. Uyqu yoki uyg'onishdan keyin ishlashda davom etadi.

---

## 1. Maqsad
Foydalanuvchi talabi: *"laptopni yoqqanda hammasi o'zi ishga tushishi kerak"* va *"doimo Altron deganimda javob qaytarishi shart"*. Bu bosqich Ultron'ni **ishonchli fon xizmati**ga aylantiradi.

## 2. Qamrov
**Kiradi:** `supervisor.py`, `tray/app.py`, `scripts/install_autostart.ps1`, `scripts/uninstall_autostart.ps1`, single-instance lock, Ollama nazorati, uyqu/uyg'onish ishlovi, health monitor.
**Kirmaydi:** Windows Service (mikrofon va desktop bilan ishlash uchun service emas, **foydalanuvchi sessiyasidagi** jarayon kerak).

---

## 3. Texnik talablar

### 3.1. Nega Windows Service emas
Windows service'lar "Session 0" da ishlaydi: ular foydalanuvchi mikrofoni, oynalari, tray'i va ilovalarni ochish imkoniyatiga ega emas. Shuning uchun **Task Scheduler "At log on"** + supervisor jarayoni ishlatiladi.

### 3.2. Supervisor (`supervisor.py`)
- `pythonw.exe -m ultron.supervisor` (konsolsiz) sifatida ishga tushadi.
- Vazifalari:
  1. **Single instance:** Windows named mutex `Local\UltronSupervisor` (`win32event.CreateMutex`; `Global\` admin'siz ishlamasligi mumkin). Asosiy jarayon uchun ham alohida `Local\UltronMain` mutex. Ikkinchi nusxa darhol chiqib ketadi.
  2. `startup_delay_s` (20 s) kutadi. Shu vaqtda Windows, drayverlar va audio xizmati tayyor bo'ladi.
  3. **Ollama tekshiruvi:** `GET /api/tags`. Javob bo'lmasa va `ollama_start_if_missing=True` bo'lsa → `ollama.exe serve` ni yashirin ishga tushiradi (`CREATE_NO_WINDOW`). 60 s gacha kutadi.
  4. Asosiy jarayonni ishga tushiradi: `pythonw.exe -m ultron` (child process), stdout/stderr `data/logs/ultron_stdout.log` ga yo'naltiriladi.
  5. **Heartbeat:** asosiy jarayon har 5 s da `data/tmp/heartbeat` fayliga vaqt yozadi. 30 s davomida yangilanmasa → jarayon "osilib qolgan" hisoblanadi → kill → qayta ishga tushiriladi.
  6. **Qayta ishga tushirish siyosati:** jarayon chiqib ketsa (exit code ≠ 0) → 2, 5, 10, 30, 60 s backoff. 10 daqiqada 5 martadan ko'p qulasa → 5 daqiqa kutish + toast "Ultron qayta-qayta to'xtayapti, log'ni tekshiring".
  7. Exit code `0` (foydalanuvchi tray'dan "Chiqish" bosgan) → supervisor ham chiqadi.
  8. Exit code `10` (tray'dan "Qayta ishga tushirish") → darhol qayta ishga tushiradi.

### 3.3. Avtostart o'rnatish (`scripts/install_autostart.ps1`)
- Task Scheduler vazifasi `Ultron`:
  - Trigger: **At log on** (joriy foydalanuvchi), delay 0 (delay supervisor ichida).
  - Action: `C:\ultron\.venv\Scripts\pythonw.exe -m ultron.supervisor`, working dir `C:\ultron`.
  - Settings: "Run only when user is logged on", **"Start the task only if the computer is on AC power" = O'CHIRILGAN**, "Stop if the computer switches to battery power" = O'CHIRILGAN, "Stop the task if it runs longer than" = O'CHIRILGAN, "If the task fails, restart every 1 minute, up to 999 times".
  - Run level: **Limited** (admin shart emas).
  - Qo'shimcha trigger: **On workstation unlock** (supervisor mutex tufayli dublikat bo'lmaydi, lekin u qulagan bo'lsa, qayta tiklanadi).
- PowerShell `Register-ScheduledTask` / `New-ScheduledTaskSettingsSet` bilan. Skript idempotent (bor bo'lsa yangilaydi).
- `uninstall_autostart.ps1` vazifani o'chiradi.
- Ollama o'zining avtostart yozuvini (Startup papkasi) saqlaydi. Supervisor faqat zaxira.

> **Ixtiyoriy (foydalanuvchi qaroriga ko'ra):** Windows'ga parolsiz avtomatik kirish (`netplwiz` yoki Sysinternals Autologon). Bu laptop yoqilishi bilan login ekranisiz Ultron ishga tushishini ta'minlaydi, lekin **laptop xavfsizligini pasaytiradi**. TZ bunga majburlamaydi, faqat README'da yo'riqnoma sifatida yoziladi.

### 3.4. Tray ilovasi (`tray/app.py`)
- `pystray` alohida thread'da. Ikonka `Pillow` bilan generatsiya qilinadi (doira + holat rangi):
  - 🟢 kulrang-yashil: IDLE (tinglayapti) · 🔵 ko'k: LISTENING · 🟣 binafsha: THINKING/SPEAKING · ⚪ kulrang: MUTED · 🔴 qizil: ERROR / Ollama yo'q / mikrofon yo'q.
- Tooltip: `Ultron — tinglayapti | oxirgi javob 1.3 s`.
- Menyu (o'zbekcha): **Mikrofonni o'chirish/yoqish** · **Matnli buyruq...** (kichik `tkinter` oynasi: yozib yuborish) · **Log papkasini ochish** · **Config'ni ochish** · **Qayta ishga tushirish** · **Chiqish**.
- Tray amallari orchestrator'ga `UserCommand` hodisasi sifatida `loop.call_soon_threadsafe` orqali yuboriladi.
- **Global hotkey** (ixtiyoriy, config): `Ctrl+Alt+U` → wake event (wake word ishlamay qolgan holatlar uchun zaxira). `pynput` kutubxonasi orqali (`uv add pynput`).

### 3.5. Uyqu va uyg'onish, qurilma o'zgarishlari
- `win32api`/`win32gui` yashirin oyna orqali `WM_POWERBROADCAST` (`PBT_APMRESUMEAUTOMATIC`, `PBT_APMSUSPEND`) hodisalari tinglanadi:
  - Suspend → audio stream'lar yopiladi, holat `MUTED(sleep)`.
  - Resume → 3 s kutish → audio qayta ochiladi → Ollama health → STT warmup (CUDA konteksti ba'zan uyg'onishdan keyin sekinlashadi) → IDLE. O'tkazib yuborilgan eslatmalar aytiladi.
- Audio qurilma o'zgarishi (TZ-02 dagi qayta ulanish) va default qurilma o'zgarganda (naushnik ulandi) → yangi default'ga o'tiladi (`input_device=null` bo'lsa).

### 3.6. Health monitor
Har 30 s:
| Tekshiruv | Muammo bo'lsa |
|---|---|
| Ollama `/api/tags` va model yuklanganmi (`/api/ps`) | Qayta ishga tushirish yoki `ensure_loaded()`, tray 🔴, 3 martadan keyin ovozda "Miyam ishlamayapti, qayta ulanyapman" |
| Audio frame'lar oqimi (oxirgi 2 s) | Stream'ni qayta ochish |
| VRAM (`nvidia-smi`) > 7.8 GB | Log'da ogohlantirish (boshqa dastur GPU'ni egallagan bo'lishi mumkin, masalan o'yin) |
| Disk bo'sh joyi < 5 GB | Eski log va utterance yozuvlarini tozalash |
| RSS > 4 GB | Log + soft restart |

### 3.7. O'yin yoki og'ir GPU ishi rejimi (ixtiyoriy, lekin tavsiya)
- Tray menyusida **"GPU'ni bo'shatish"**: Ollama modeli VRAM'dan tushiriladi (`keep_alive=0`), STT CPU'ga (`small`, int8) o'tkaziladi. Ultron sekinroq, lekin ishlashda davom etadi. "Normal rejim" → qaytarish.

---

## 4. Testlar
- `test_supervisor.py`: mock child process bilan: crash → backoff; heartbeat to'xtashi → restart; exit 0 → chiqish; exit 10 → darhol restart.
- `test_single_instance.py`: ikkinchi nusxa chiqib ketadi.
- Qo'lda tekshiruvlar ro'yxati (`docs/manual_tests_09.md`): quyidagi qabul bandlari.

## 5. Qabul mezonlari
- [ ] `install_autostart.ps1` → laptopni qayta yoqish → login → **≤ 90 s** ichida "Tayyorman" deydi, konsol oynasi ko'rinmaydi
- [ ] Task Manager'da `pythonw.exe` jarayonini majburan o'chirish → ≤ 15 s ichida qayta ishga tushadi
- [ ] Ollama'ni o'chirish → Ultron 🔴 ko'rsatadi → ≤ 90 s ichida o'zi tiklanadi
- [ ] Laptopni uyquga yuborish → uyg'otish → ≤ 10 s ichida "Altron" ga javob beradi
- [ ] Batareyada ishlaganda ham avtostart va ishlash davom etadi
- [ ] Tray menyusining barcha bandlari ishlaydi. Matnli buyruq oynasi javob beradi
- [ ] Naushnikni ulash/uzish → ishlash davom etadi
- [ ] 7 kunlik real foydalanish davomida qo'lda aralashuv kerak bo'lmasligi (keyinroq tekshiriladi, `PROGRESS.md` ga yoziladi)
- [ ] `uninstall_autostart.ps1` hammasini tozalaydi

## 6. Xavflar
| Muammo | Yechim |
|---|---|
| Login'dan keyin audio xizmati hali tayyor emas | `startup_delay_s` + audio qayta ulanish logikasi |
| Windows yangilanishidan keyin CUDA yoki drayver muammosi | check_env'ni tray'dan ishga tushirish, toast bilan xabar |
| Antivirus `pythonw` fon jarayoniga shubha qiladi | Kerak bo'lsa `C:\ultron` papkasini Defender istisnolariga qo'shish (foydalanuvchi qaroriga ko'ra) |
| Uyg'onishdan keyin CUDA "osilib" qoladi | Resume'da STT warmup. Muvaffaqiyatsiz bo'lsa → exit 10 (supervisor qayta ishga tushiradi) |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md va docs/tz/09_autostart.md ni to'liq o'qi (00-TZ dan faqat §5 autostart kerak).
Faqat shu bosqichni bajar. PowerShell skriptlarni men ishga tushiraman: qaysi buyruqni qanday (oddiy yoki admin) ishga tushirishni ayt.
Oxirida docs/manual_tests_09.md ni yarat va qabul bandlarini birga tekshiramiz.
```
