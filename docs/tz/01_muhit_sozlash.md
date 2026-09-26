# TZ-01. Muhitni sozlash va loyiha skeleti

**Bog'liqlik:** yo'q (birinchi bosqich)
**Natija:** barcha tizim dasturlari o'rnatilgan, `C:\ultron` repo skeleti tayyor, `scripts/check_env.py` hamma bandni ✅ ko'rsatadi.

---

## 1. Maqsad
Ultron uchun Windows 11 muhitini tayyorlash: NVIDIA drayver, Python 3.11, uv, Git, Ollama, Qwen modeli, CUDA kutubxonalari, loyiha skeleti, config tizimi va log'lar.

## 2. Qamrov
**Kiradi:** tizim dasturlarini o'rnatish bo'yicha yo'riqnoma, `pyproject.toml`, papka skeleti, `config.py`, `logging_setup.py`, `_cuda_dlls.py`, `__main__.py` (bo'sh CLI), `check_env.py`, `download_models.py` (skelet), `docs/PROGRESS.md`.
**Kirmaydi:** audio, STT, LLM logikasi (keyingi bosqichlar).

---

## 3. Qo'lda bajariladigan qadamlar (foydalanuvchi, PowerShell)

> Claude Code bu qadamlarni **foydalanuvchiga ro'yxat qilib beradi va tekshiradi**. Admin huquqi yoki GUI kerak bo'lganlarini o'zi bajarishga urinmaydi.

### 3.1. Windows sozlamalari
1. **Settings → Privacy & security → Microphone:** "Microphone access" = On, "Let desktop apps access your microphone" = On.
2. **NVIDIA Control Panel → Manage 3D settings → Program settings:** `python.exe` va `ollama.exe` uchun "High-performance NVIDIA processor" tanlanadi.
3. (Tavsiya) Laptop ekrani integrated Radeon'da qolsin (MUX/Advanced Optimus bo'lsa "Hybrid/Optimus" rejimi). Shunda RTX'ning VRAM'i AI uchun bo'sh qoladi.
4. **Power:** zaryadlovchiga ulangan holatda "Best performance" rejimi.
5. Windows foydalanuvchi nomida kirill harflari yoki bo'shliq bo'lsa, barcha kesh papkalari `C:\ultron\data` ichiga yo'naltiriladi (quyidagi env o'zgaruvchilari).

### 3.2. Dasturlarni o'rnatish
```powershell
# NVIDIA drayver: GeForce Experience / NVIDIA App orqali eng so'nggi Game Ready yoki Studio drayver
nvidia-smi                                   # CUDA Version: 12.x yoki undan yuqori chiqishi kerak

winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Ollama.Ollama -e
winget install --id Microsoft.VCRedist.2015+.x64 -e

# Yangi PowerShell oynasini oching, keyin:
uv python install 3.11
git --version ; uv --version ; ollama --version
```

### 3.3. Muhit o'zgaruvchilari (User darajasida)
```powershell
setx HF_HOME "C:\ultron\data\hf_cache"
setx OLLAMA_FLASH_ATTENTION "1"
setx OLLAMA_KV_CACHE_TYPE "q8_0"
setx OLLAMA_MODELS "C:\ultron\data\ollama_models"     # ixtiyoriy: modellarni bitta joyda saqlash
setx OLLAMA_HOST "127.0.0.1:11434"                   # faqat lokal
```
Ollama'ni tray'dan **to'liq yopib**, qayta oching (env o'zgaruvchilari shundan keyin kuchga kiradi).

### 3.4. LLM modelini yuklab olish
```powershell
ollama pull qwen3:8b
ollama run qwen3:8b "Salom, o'zingni qisqacha tanishtir. /no_think"
ollama ps                                            # PROCESSOR = 100% GPU bo'lishi kerak
```
> Agar `ollama ps` CPU/GPU aralash ko'rsatsa (masalan `20%/80% CPU/GPU`), VRAM yetmayapti. Ekran NVIDIA'da emasligini tekshiring yoki `num_ctx`ni kamaytiring.
> Model nomi config'da turadi (`llm.model`). Ollama'da yangiroq Qwen 7–9B tool-calling modeli chiqsa, faqat config o'zgartiriladi.

---

## 4. Claude Code bajaradigan ishlar

### 4.1. Repo va `pyproject.toml`
- `C:\ultron` ichida `git init`, `.gitignore` (`data/`, `.venv/`, `config/config.local.yaml`, `__pycache__`, `*.onnx`, `*.wav`).
- `uv init --package --name ultron` (src layout), `.python-version` = `3.11`.
- Asosiy bog'liqliklar (bu bosqichda hammasi o'rnatiladi, keyingi bosqichlar faqat qo'shimcha qo'shadi):

```toml
[project]
name = "ultron"
requires-python = ">=3.11,<3.12"
dependencies = [
  "pydantic>=2", "pyyaml", "loguru", "numpy<2.3",
  "sounddevice", "soundfile", "scipy",
  "onnxruntime",                 # Silero VAD, openWakeWord (CPU)
  "faster-whisper",
  "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10",   # CTranslate2 uchun CUDA DLL'lar
  "torch", "transformers", "accelerate",               # MMS-TTS
  "ollama",
  "rapidfuzz", "aiosqlite", "apscheduler>=3.10,<4",
  "psutil", "pycaw", "comtypes", "screen-brightness-control", "send2trash",
  "pywin32", "pystray", "pillow", "win11toast", "httpx", "mss",
]

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "ruff", "jiwer"]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }

[project.scripts]
ultron = "ultron.__main__:main"
```
> Versiyalar bo'yicha: agar biror paket Python 3.11 + Windows uchun o'rnatilmasa, Claude Code muqobil versiyani tanlaydi va hisobotda yozadi. `torch` CUDA build'i shart emas (TTS CPU'da ishlaydi), lekin kelajakda TTS'ni GPU'ga o'tkazish uchun CUDA build o'rnatiladi.

### 4.2. `src/ultron/_cuda_dlls.py`
Windows'da CTranslate2 `cublas64_12.dll` va `cudnn*_9.dll` fayllarini topa olmasligi — eng ko'p uchraydigan muammo. Bu modul `site-packages/nvidia/*/bin` papkalarini `os.add_dll_directory()` orqali qo'shadi va `PATH`ga ham yozadi. U `ultron/__init__.py` ichida **eng birinchi** import qilinadi.

### 4.3. `config.py`
- 00-TZ §5 dagi sxema asosida pydantic v2 modellari (`AppConfig`, `AudioConfig`, ... `RootConfig`).
- `load_config(path="config/config.yaml")`: YAML'ni o'qiydi, keyin `config.local.yaml` bo'lsa deep-merge qiladi. `~` va nisbiy yo'llarni `data_dir`ga nisbatan absolyut yo'lga aylantiradi.
- `ULTRON_CONFIG` env o'zgaruvchisi config yo'lini almashtirishga imkon beradi.
- Noto'g'ri qiymatda aniq xato xabari chiqadi (qaysi kalit, qanday xato).

### 4.4. `logging_setup.py`
- loguru: konsolga (rangli) va `data/logs/ultron.log` fayliga (rotation va retention config'dan).
- Alohida sink: `data/logs/tools_audit.log` (faqat `extra["audit"]=True` yozuvlar uchun).
- `pythonw` bilan ishga tushirilganda (konsol yo'q) konsol sink qo'shilmaydi.

### 4.5. `__main__.py`
```
python -m ultron            # (hozircha) config va log'ni yuklab, "Ultron skeleti ishlayapti" deb log yozadi
python -m ultron --text     # keyingi bosqichlar uchun joy
python -m ultron --debug
python -m ultron --check    # check_env'ni chaqiradi
```
`argparse` ishlatiladi. Asosiy funksiya `main()` → `asyncio.run(run(args))`.

### 4.6. `scripts/check_env.py`
Har bir tekshiruvni ✅ / ❌ bilan va tuzatish bo'yicha maslahat bilan chiqaradi:

| # | Tekshiruv | Qanday |
|---|---|---|
| 1 | Python 3.11 | `sys.version_info` |
| 2 | NVIDIA GPU, drayver, VRAM | `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv` |
| 3 | CTranslate2 CUDA | `ctranslate2.get_cuda_device_count() > 0` |
| 4 | cuBLAS/cuDNN DLL | `ctypes.WinDLL("cublas64_12.dll")`, `cudnn_ops64_9.dll` |
| 5 | onnxruntime | versiya va provayderlar |
| 6 | torch | versiya, `torch.cuda.is_available()` (ma'lumot uchun) |
| 7 | Ollama server | `GET http://127.0.0.1:11434/api/tags` |
| 8 | Qwen modeli bor | `llm.model` tags ichida |
| 9 | Qwen tool calling | bitta sinov `get_time` tool bilan `ollama.chat`, javobda `tool_calls` bor |
| 10 | Mikrofon | `sounddevice.query_devices()`, 2 s yozib RMS > 0 |
| 11 | Karnay | 0.5 s 440 Hz sinov tovushi (`--no-sound` bilan o'tkazib yuboriladi) |
| 12 | Disk | `data_dir` diskida ≥ 30 GB bo'sh joy |
| 13 | Yo'llar | `data_dir` yozish mumkin, yo'lda kirill harflari yo'q |

Natija oxirida umumiy xulosa chiqadi. ❌ bo'lsa, exit code = 1.

### 4.7. `scripts/download_models.py` (skelet)
Keyingi bosqichlar o'z modellarini shu yerga qo'shadi (`--stt`, `--tts`, `--vad`, `--all` flag'lari). Hozircha faqat `data/models/` papkalarini yaratadi.

### 4.8. `docs/PROGRESS.md`
```markdown
| Bosqich | Holat | Sana | Izoh |
|---|---|---|---|
| 01 Muhit | ✅ | YYYY-MM-DD | ... |
```

### 4.9. Testlar
- `tests/test_config.py`: default yuklash, local override merge, noto'g'ri qiymat xatosi.
- `tests/test_logging.py`: audit sink faqat audit yozuvlarni oladi.

---

## 5. Qabul mezonlari
- [ ] `uv sync` xatosiz tugaydi
- [ ] `uv run python scripts/check_env.py` barcha 13 bandda ✅ ko'rsatadi
- [ ] `ollama ps` bo'yicha Qwen modeli **100% GPU**da ishlaydi
- [ ] `uv run python -m ultron` config va log'ni yuklaydi, `data/logs/ultron.log` yaratiladi
- [ ] `uv run pytest -m "not hardware"` o'tadi, `ruff check` toza
- [ ] `docs/PROGRESS.md` yangilangan, git commit qilingan

## 6. Xavflar va yechimlar
| Muammo | Yechim |
|---|---|
| `Could not locate cudnn_ops64_9.dll` | `_cuda_dlls.py` birinchi import qilinganini tekshirish; `nvidia-cudnn-cu12` 9.x versiyasi o'rnatilganini tekshirish |
| Ollama CPU'da ishlayapti | NVIDIA Control Panel sozlamasi, drayverni yangilash, `ollama ps` |
| `numpy` 2.x bilan mos kelmaslik | `numpy<2.3` cheklovi; kerak bo'lsa `<2` |
| HF kesh `C:\Users\<kirill>\...` ichida | `HF_HOME` env o'zgaruvchisi |

---

## 7. Claude Code uchun prompt
```
CLAUDE.md va docs/tz/00_umumiy_arxitektura.md (§4, §5, §7) ni o'qi.
Keyin docs/tz/01_muhit_sozlash.md ni to'liq o'qi va shu bosqichni bajar.
§3 dagi qo'lda bajariladigan qadamlarni menga ro'yxat qilib ber va har birini tekshirishimni kut;
§4 dagi hamma narsani o'zing qil. Oxirida check_env natijasi va qabul mezonlari jadvalini ko'rsat.
```
