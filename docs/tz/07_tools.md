# TZ-07. Tool'lar: tizim, fayllar, brauzer/internet, xotira va eslatmalar

**Bog'liqlik:** TZ-06 (`tools/base.py`, registry, text rejim)
**Natija:** MVP'dagi barcha amallar tool sifatida ishlaydi va `--text` rejimida sinab ko'rilgan.

> ⚠️ **Bu bosqich katta.** Limitni tejash uchun uni **4 ta sessiyaga** bo'ling: **7A → 7B → 7C → 7D**. Har bir sessiyada shu fayldagi §1–§2 va faqat tegishli bo'lim o'qiladi.

---

## 1. Umumiy qoidalar (barcha qismlar uchun)
- Har bir tool: `@tool` decorator, **inglizcha description**, argumentlarda `Annotated` tavsiflar, `ToolResult` qaytaradi.
- `ToolResult.speak` — tayyor, qisqa **o'zbekcha** javob ("Telegram ochildi"). LLM uni o'zgartirib aytishi ham mumkin.
- `ToolResult.data` — LLM uchun tuzilgan ma'lumot (JSON-mos).
- Xavfli tool'lar `dangerous=True`. Ular uchun tasdiqlash matni o'zbekcha ("Kompyuterni o'chirishni tasdiqlaysizmi?").
- Windows API chaqiruvlari sinxron bo'lsa → `asyncio.to_thread`. COM (pycaw) ishlatiladigan thread'da `comtypes.CoInitialize()` chaqiriladi.
- Har bir tool uchun **mock rejim** (`ULTRON_TOOLS_MOCK=1`): haqiqiy amal bajarilmaydi, faqat log yoziladi (eval va testlar uchun).
- Fast-path (TZ-06 §3.5) intentlari shu tool'larga ulanadi.
- Qo'shimcha bog'liqliklar: `uv add watchdog` (7B), `uv add --dev freezegun` (7D).

## 2. Tool ro'yxati (umumiy ko'rinish)
| Qism | Tool | Xavfli |
|---|---|---|
| 7A | `open_app`, `close_app`, `list_running_apps`, `set_volume`, `mute`, `get_volume`, `set_brightness`, `lock_screen`, `sleep_pc`, `shutdown_pc`, `restart_pc`, `cancel_shutdown`, `screenshot`, `get_system_status`, `get_time`, `get_date` | sleep/shutdown/restart ✅ |
| 7B | `search_files`, `open_file`, `open_folder`, `list_folder`, `recent_files`, `create_folder`, `rename_item`, `move_item`, `copy_item`, `delete_item` | move/delete/rename ✅ |
| 7C | `open_url`, `web_search`, `youtube_search`, `get_weather` | — |
| 7D | `remember_fact`, `recall_facts`, `forget_fact`, `set_reminder`, `set_timer`, `list_reminders`, `cancel_reminder` | forget ✅ |

---

## 7A. Tizim boshqaruvi (`system_tools.py`, `app_index.py`)

### 7A.1. Ilovalar indeksi (`app_index.py`)
- Ishga tushishda (va har 6 soatda) skanerlanadi:
  - `%ProgramData%\Microsoft\Windows\Start Menu\Programs\**\*.lnk`
  - `%AppData%\Microsoft\Windows\Start Menu\Programs\**\*.lnk`
  - Microsoft Store ilovalari: `Get-StartApps` (PowerShell) → `AppID` (`explorer.exe shell:AppsFolder\<AppID>` orqali ochiladi)
- Har bir yozuv: `display_name`, `target/app_id`, `exe_name` (yopish uchun).
- `config/app_aliases.yaml` — o'zbekcha sinonimlar (Claude Code boshlang'ich ro'yxatni yaratadi):
```yaml
brauzer: chrome
xrom: chrome
bloknot: notepad
kalkulyator: calc
telegram: telegram
fayllar: explorer
provodnik: explorer
sozlamalar: "ms-settings:"
vs kod: code
vord: winword
eksel: excel
```
- Qidiruv: alias → aniq moslik → `rapidfuzz.process.extractOne` (`WRatio`, ≥ 75). Bir nechta yaqin natija bo'lsa (farq < 5 ball) → `ok=False`, `data={"candidates": [...]}`, LLM foydalanuvchidan aniqlashtiradi.
- O'zbekcha qo'shimchalarni olib tashlash: "telegramni" → "telegram" (`-ni, -ga, -da, -dan, -ning` suffix stripping, `text/normalize.py` ga `strip_uz_suffixes()`).

### 7A.2. Tool'lar
| Tool | Amalga oshirish |
|---|---|
| `open_app(name)` | `os.startfile(lnk)` yoki `explorer shell:AppsFolder\ID`; `ms-settings:` URI'lar |
| `close_app(name)` | `psutil` → `exe_name` bo'yicha jarayonlar → avval `terminate()`, 3 s dan keyin `kill()`. **Tizim jarayonlarini** (explorer, csrss, winlogon, svchost, python/ultron o'zi, ollama) yopish taqiqlanadi |
| `list_running_apps()` | Oynasi ko'rinib turgan jarayonlar (`win32gui.EnumWindows`) |
| `set_volume(level: int\|None, delta: int\|None)` | `pycaw` `IAudioEndpointVolume.SetMasterVolumeLevelScalar`; 0–100 oralig'iga qirqiladi |
| `mute(on: bool)`, `get_volume()` | pycaw |
| `set_brightness(level\|delta)` | `screen_brightness_control` (laptop ekrani) |
| `lock_screen()` | `ctypes.windll.user32.LockWorkStation()` |
| `sleep_pc()` ⚠️ | `rundll32.exe powrprof.dll,SetSuspendState 0,1,0` (yoki `SetSuspendState` API) |
| `shutdown_pc(delay_s=30)` ⚠️ | `shutdown /s /t <delay>`; javob: "30 soniyadan keyin o'chadi, bekor qilish uchun 'bekor qil' deng" |
| `restart_pc(delay_s=30)` ⚠️ | `shutdown /r /t <delay>` |
| `cancel_shutdown()` | `shutdown /a` |
| `screenshot()` | `mss` → `~/Pictures/Ultron/screenshot_YYYYmmdd_HHMMSS.png`; `data.path` |
| `get_system_status()` | batareya (`psutil.sensors_battery`), CPU%, RAM%, GPU VRAM (`nvidia-smi`), disk bo'sh joyi |
| `get_time()`, `get_date()` | TZ-06 dagi demo'lar shu yerga ko'chiriladi. O'zbekcha oy va hafta kunlari nomlari bilan |

### 7A.3. Qabul (7A)
- [ ] `--text`: "Telegramni och", "brauzerni och", "bloknotni yop", "ovozni 40 qil", "ovozni o'chir", "yorug'likni 70 qil", "kompyuterni qulfla", "skrinshot ol", "batareya qancha?" to'g'ri ishlaydi
- [ ] "kompyuterni o'chir" tasdiq so'raydi. "yo'q" → bekor qilinadi. "ha" → 30 s taymer, "bekor qil" → `shutdown /a`
- [ ] Noma'lum ilova → "Bunday dastur topilmadi" yoki nomzodlar bilan savol
- [ ] `test_app_index.py` (mock .lnk ro'yxati bilan fuzzy va alias), `test_system_tools.py` (mock)

---

## 7B. Fayllar (`file_tools.py`)

### 7B.1. Xavfsizlik chegarasi
- Barcha yo'llar `Path.resolve()` qilinadi va `allowed_roots` ichida bo'lishi **majburiy** (`is_relative_to`). Aks holda `ok=False, error="ruxsat etilmagan joy"`.
- Tizim papkalari (`C:\Windows`, `Program Files`, `AppData`) har doim taqiqlangan.
- O'chirish **faqat Recycle Bin'ga** (`send2trash`). To'liq o'chirish yo'q.
- Papka nomlari uchun o'zbekcha sinonimlar: "ish stoli" → Desktop, "hujjatlar" → Documents, "yuklamalar/zagruzka" → Downloads, "rasmlar" → Pictures, "musiqalar" → Music, "videolar" → Videos.

### 7B.2. Tool'lar
| Tool | Tavsif |
|---|---|
| `search_files(query, extension=None, folder=None, modified_within_days=None)` | Fayl nomi bo'yicha fuzzy qidiruv. Tez ishlashi uchun **fon indeksi**: ishga tushishda `allowed_roots` skanerlanadi (fayl nomi, yo'l, o'lcham, mtime → SQLite `file_index` jadvali), `watchdog` kutubxonasi o'zgarishlarni kuzatadi. Qidiruv ≤ 300 ms. Top `max_results` natija qaytadi |
| `open_file(path_or_query)` | Yo'l bo'lsa ochadi; so'rov bo'lsa `search_files` → 1 ta aniq natija bo'lsa ochadi, bir nechta bo'lsa nomzodlarni qaytaradi |
| `open_folder(name_or_path)` | `os.startfile(folder)` |
| `list_folder(path, limit=20)` | Nomlar va turlari |
| `recent_files(folder="Downloads", limit=5)` | mtime bo'yicha |
| `create_folder(parent, name)` | |
| `rename_item(path, new_name)` ⚠️ | |
| `move_item(src, dst_folder)` ⚠️, `copy_item(src, dst_folder)` | Nom to'qnashsa `(1)` qo'shiladi |
| `delete_item(path)` ⚠️ | `send2trash` |

- LLM'ga fayl yo'llari **qisqartirilgan** holda beriladi (`~\Downloads\rezyume.pdf`). Ovozda esa faqat fayl nomi aytiladi.
- **Kontekst:** oxirgi qidiruv natijalari agent xotirasida saqlanadi. "ikkinchisini och" kabi murojaatlar ishlashi uchun natijalar raqamlangan holda qaytariladi.

### 7B.3. Qabul (7B)
- [ ] "Yuklamalar papkasini och", "rezyume degan faylni top", "ikkinchisini och", "oxirgi yuklangan faylni och", "ish stolida 'Loyiha' papkasini yarat", "shu faylni hujjatlarga ko'chir" (tasdiq bilan), "uni o'chir" (tasdiq, Recycle Bin) ishlaydi
- [ ] `allowed_roots` tashqarisidagi yo'l rad etiladi (test bilan)
- [ ] 50 000 faylli indeksda qidiruv ≤ 300 ms
- [ ] `test_file_tools.py`: `tmp_path` ichida vaqtinchalik `allowed_roots` bilan barcha operatsiyalar

---

## 7C. Brauzer va internet (`browser_tools.py`)

| Tool | Amalga oshirish |
|---|---|
| `open_url(url)` | Faqat `http/https`. Config'dagi brauzer yoki `webbrowser.open` |
| `web_search(query, engine="google")` | `https://www.google.com/search?q=...` (URL-encode). Brauzerda ochiladi, natijalar o'qilmaydi |
| `youtube_search(query, autoplay=False)` | `https://www.youtube.com/results?search_query=...` |
| `get_weather(city=None, days=1)` | `internet_tools_enabled=True` bo'lsa: **Open-Meteo** ochiq API'si (kalit kerak emas, AI emas) `httpx` bilan, timeout 5 s. Shahar nomi → koordinata uchun Open-Meteo geocoding API. Natija: harorat, holat (WMO kodi → o'zbekcha matn), yog'ingarchilik ehtimoli. Internet yo'q bo'lsa → "Internetga ulanib bo'lmadi" |

- Mashhur saytlar lug'ati (config yoki kodda): "youtube", "telegram web", "gmail", "github", "kun.uz", "google tarjimon" → URL.
- **Eslatma:** "Hammasi lokal" talabi AI modellar uchun. Brauzer va ob-havo API'si LLM'ga ma'lumot bermaydi, faqat foydalanuvchi so'roviga javob uchun ishlatiladi. `internet_tools_enabled=False` bo'lsa, `get_weather` registry'dan chiqariladi.

### Qabul (7C)
- [ ] "YouTube'da Sherali Jo'rayev qo'shiqlarini och", "Google'da Python asyncio'ni qidir", "kun.uz ni och", "ertaga Toshkentda ob-havo qanday?" ishlaydi
- [ ] Internet uzilganda `get_weather` 5 s ichida chiroyli xato qaytaradi
- [ ] `test_browser_tools.py`: URL yaratish va encoding, WMO → o'zbekcha matn

---

## 7D. Xotira va eslatmalar (`memory/db.py`, `memory/scheduler.py`, `memory_tools.py`, `reminder_tools.py`)

### 7D.1. Ma'lumotlar bazasi (`aiosqlite`, `memory.db_path`)
```sql
CREATE TABLE facts (
  id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT NOT NULL,
  category TEXT DEFAULT 'general', created_at TEXT, updated_at TEXT
);
CREATE TABLE reminders (
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL,   -- ISO, local vaqt
  repeat TEXT DEFAULT NULL,        -- NULL | 'daily' | 'weekdays' | 'weekly' | cron
  kind TEXT DEFAULT 'reminder',    -- 'reminder' | 'timer'
  status TEXT DEFAULT 'pending',   -- 'pending' | 'done' | 'cancelled' | 'missed'
  created_at TEXT
);
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY, ts TEXT, role TEXT, text TEXT, tool_name TEXT, latency_ms INTEGER
);
CREATE TABLE file_index ( ... );   -- 7B dan
```
- Migratsiyalar: oddiy `schema_version` jadvali va ketma-ket SQL fayllar.

### 7D.2. Xotira tool'lari
| Tool | Tavsif |
|---|---|
| `remember_fact(key, value, category)` | "Mening tug'ilgan kunim 15-mart" → `key="birthday", value="15-mart"`. Kalit bor bo'lsa yangilanadi |
| `recall_facts(query=None)` | Barcha faktlar yoki kalit/qiymat bo'yicha qidiruv |
| `forget_fact(key)` ⚠️ | |
- System prompt'ga (`{facts}`) oxirgi yangilangan `max_facts_in_prompt` (40) fakt `key: value` ko'rinishida qo'shiladi.

### 7D.3. Eslatma va taymerlar
| Tool | Tavsif |
|---|---|
| `set_reminder(text, due_at_iso, repeat=None)` | Nisbiy vaqtni ("ertaga soat 9 da", "yarim soatdan keyin") **LLM** system prompt'dagi `now_iso` asosida ISO'ga aylantiradi. Tool vaqt o'tmishda emasligini tekshiradi |
| `set_timer(seconds, label=None)` | Fast-path: "10 daqiqaga taymer qo'y" |
| `list_reminders()` | Kelgusi 10 tasi |
| `cancel_reminder(id_or_text)` | Fuzzy matn bo'yicha ham |

**Scheduler (`scheduler.py`):**
- APScheduler 3.x `AsyncIOScheduler`, jobstore = xotirada. Manba — SQLite: ishga tushishda barcha `pending` eslatmalar qayta rejalashtiriladi.
- **O'tkazib yuborilganlar:** laptop o'chiq yoki uyquda bo'lgan paytdagi eslatmalar ishga tushishda "O'tkazib yuborilgan eslatma: ..." deb aytiladi (24 soatdan eskilari `missed` deb belgilanadi va faqat toast'da ko'rsatiladi).
- Vaqti kelganda: `ReminderDue` hodisasi → orchestrator: `confirm` earcon + TTS "Eslatma: {text}" + Windows toast (`win11toast`). Takroriy bo'lsa, keyingi vaqt hisoblanadi.
- Agar shu paytda Ultron gapirayotgan bo'lsa → navbatga qo'yiladi, gap tugagach aytiladi.

### Qabul (7D)
- [ ] "Mening ismim <ismingiz>, eslab qol" → keyingi sessiyada "mening ismim nima?" → to'g'ri javob
- [ ] "5 daqiqadan keyin choyni eslat" → 5 daqiqada ovoz + toast
- [ ] "Har kuni soat 8 da sport qilishni eslat" → takroriy ishlaydi, `list_reminders` da ko'rinadi
- [ ] Jarayon qayta ishga tushirilganda eslatmalar yo'qolmaydi
- [ ] `test_memory_db.py`, `test_scheduler.py` (vaqtni `freezegun` yoki qo'lda soat bilan)

---

## 3. Yakuniy qabul (butun TZ-07)
- [ ] 7A–7D qabul bandlari bajarilgan
- [ ] `eval_llm_tools.py` haqiqiy tool sxemalari bilan (mock rejimda) **≥ 85%** tool aniqligi. `llm_eval_uz.yaml` kamida 60 ta holatgacha kengaytirilgan
- [ ] Audit log'da barcha chaqiruvlar bor
- [ ] `PROGRESS.md` da 7A/7B/7C/7D alohida qatorlar

---

## 4. Claude Code uchun promptlar
```
[7A] CLAUDE.md va docs/tz/07_tools.md ning §1, §2 va "7A" bo'limini o'qi (00-TZ dan faqat §6 tools/base kerak bo'lsa).
     Faqat 7A ni bajar, qabul bandlarini tekshir.
[7B] ... §1, §2 va "7B" bo'limini o'qi. Faqat 7B ni bajar.
[7C] ... §1, §2 va "7C" bo'limini o'qi. Faqat 7C ni bajar.
[7D] ... §1, §2 va "7D" bo'limini o'qi. Faqat 7D ni bajar. Oxirida §3 yakuniy qabulni tekshir.
```
