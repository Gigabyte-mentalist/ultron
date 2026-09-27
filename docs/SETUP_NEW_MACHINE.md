# Ultron'ni yangi kompyuterda o'rnatish (masalan RTX 3060 12GB)

Bu hujjat GitHub'dagi **kod**dan tashqari, yangi mashinada Ultron'ni ishga
tushirish uchun kerak bo'lgan **hamma narsa**ni tushuntiradi: nima avtomatik
yuklab olinadi, nima qo'lda sozlanadi, va shaxsiy ma'lumotlar (ovoz yozuvlari,
xotira) bilan nima qilish kerak.

---

## 0. Qisqacha: GitHub'da nima bor, nima yo'q

| Bor (git orqali keladi) | Yo'q (qayta yuklanadi yoki qo'lda ko'chiriladi) |
|---|---|
| Butun kod (`src/ultron/`, `scripts/`, `training/`) | `.venv/` (Python muhiti) |
| Testlar (`tests/`) | `data/models/` (STT/TTS/VAD/wake modellari, GB'larcha) |
| TZ hujjatlari, PROGRESS.md | `data/recordings/` (**shaxsiy ovoz yozuvlaringiz**) |
| `config/config.yaml` (standart sozlamalar) | `data/db/ultron.sqlite` (eslatmalar, xotiradagi fakts) |
| `config/app_aliases.yaml` | `data/logs/`, `data/reports/` |
| `uv.lock` (aniq paket versiyalari) | `config/config.local.yaml` (mashinaga xos sozlamalar) |
| | `training/stt/checkpoints/`, `open_datasets/` (o'qitish natijalari) |

Sabab: modellar va shaxsiy ma'lumotlar juda katta (ba'zilari GB'larcha) va/yoki
shaxsiy — GitHub'ga (ayniqsa **private bo'lsa ham**) yuklash noto'g'ri.
Ularning har biri pastda tushuntirilgan.

---

## 1. Talablar

| # | Talab | Tekshirish |
|---|---|---|
| 1 | Windows 11 | — |
| 2 | NVIDIA GPU, drayver CUDA 12.x+ | `nvidia-smi` (CUDA Version qatorini tekshiring) |
| 3 | Python 3.11 (aynan shu, `uv` orqali o'rnatiladi) | `uv python install 3.11` |
| 4 | `uv` paket menejeri | `winget install --id astral-sh.uv -e` |
| 5 | Ollama | `winget install --id Ollama.Ollama -e` |
| 6 | Git | odatda allaqachon bor |

**RTX 3060 12GB vs RTX 4060 Laptop 8GB:** loyihaning VRAM byudjeti 8 GB'ga
mo'ljallangan (STT ≤2GB + LLM ≤5.5GB), 12 GB'da bemalol sig'adi, hech qanday
config o'zgarishi shart emas. Aksincha, 12 GB bilan kelajakda kattaroq
STT/LLM modelini sinash imkoniyati ham bor.

---

## 2. Klonlash (private repo)

```powershell
git clone https://github.com/Gigabyte-mentalist/ultron.git C:\ultron
cd C:\ultron
```

Repo **private** — birinchi `git clone`/`git push` paytida Git Credential
Manager oynasi ochiladi, brauzer orqali GitHub'ga kirib tasdiqlaysiz (bir
marta, keyin eslab qoladi). Agar `nurmuhammad1160` akkaunti bilan kirsangiz,
taklifni avval https://github.com/Gigabyte-mentalist/ultron/invitations
sahifasida qabul qilishi kerak.

---

## 3. Python muhiti

```powershell
cd C:\ultron
uv python install 3.11
uv sync
```

`uv sync` `pyproject.toml`+`uv.lock`dagi barcha paketlarni (`torch` CUDA
build'i bilan, `faster-whisper`, `transformers` va h.k.) `.venv\`ga
o'rnatadi — internetdan ~5-10 GB yuklanadi (asosan `torch`/CUDA
kutubxonalari), 10-20 daqiqa.

---

## 4. Ollama + LLM modeli

```powershell
ollama pull qwen3:8b
ollama run qwen3:8b "Salom, o'zingni qisqacha tanishtir. /no_think"
```

`ollama pull` ~5 GB yuklaydi. Model nomi `config/config.yaml`'ning
`llm.model`sida — agar keyinchalik boshqa Qwen versiyasini sinasangiz, faqat
shu qatorni o'zgartirasiz.

---

## 5. `data_dir`ni sozlash (mashinaga xos)

`config/config.local.yaml` git'da yo'q (har bir mashina o'zinikini yaratadi).
C diskda joy tanqisligi bo'lsa (bu loyihada shunday bo'lgan edi), boshqa
diskka ko'rsating:

```yaml
# config/config.local.yaml
app:
  data_dir: "D:/ultron_data"   # yoki istalgan diskdagi papka
```

Agar C diskda joy yetarli bo'lsa, bu faylni yaratmasangiz ham bo'ladi —
standart `config.yaml`'dagi `C:/ultron/data` ishlatiladi.

---

## 6. Modellarni yuklab olish

```powershell
uv run python scripts/download_models.py --all
```

Bu quyidagilarni `<data_dir>/models/`ga yuklaydi (internetdan, Hugging
Face'dan):
- **VAD**: Silero VAD (ONNX, kichik)
- **STT**: standart faster-whisper modellar + `docs/PROGRESS.md`'ning
  03-bosqich yozuvida ko'rsatilgan tanlangan o'zbekcha fine-tune
  (`islomov/rubaistt_v2_medium`, CTranslate2'ga avtomatik o'giriladi)
- **TTS**: `facebook/mms-tts-uzb-script_cyrillic`
- **Wake word**: openWakeWord'ning umumiy feature modellari

Alohida ham yuklash mumkin: `--vad`, `--stt`, `--tts`, `--oww`.

**Eslatma:** agar 10/11-bosqichlarda (`training/wake/`, `training/stt/`)
shaxsiy o'qitilgan modellar (`altron.onnx`, `ultron-uz-vN-ct2`) bo'lsa va
ularni ham shu mashinada ishlatmoqchi bo'lsangiz, ularni **eski
mashinadan qo'lda ko'chiring** (USB/tashqi disk/cloud orqali) —
`data/models/wake/altron.onnx` va `data/models/stt/ultron-uz-vN-ct2/`
mos ravishda `config.yaml`'ning `wake.oww_model_path`/`stt.model`siga mos
kelishi kerak. Bu sessiyada bunday shaxsiy modellar hali **o'qitilmagan**
(faqat infratuzilma tayyor), shuning uchun hozircha bu qadam kerak emas.

---

## 7. Tekshirish

```powershell
uv run python scripts/check_env.py
```

13 bandning barchasi ✅ bo'lishi kerak (Python versiyasi, GPU/drayver,
CTranslate2 CUDA, Ollama server va h.k.). Muammo chiqsa, skriptning o'zi
sabab va yechimni ko'rsatadi.

```powershell
uv run pytest -m "not hardware"
uv run ruff check .
```

745 ta test o'tishi, `ruff check` toza chiqishi kerak — agar shunday bo'lsa,
kod tomondan hammasi joyida.

---

## 8. Ishga tushirish

```powershell
uv run python -m ultron --text        # matnli debug (mikrofonsiz) -- birinchi sinov uchun tavsiya
uv run python -m ultron                # to'liq ovozli rejim
uv run python -m ultron --debug        # + batafsil holat logi (mic/VAD/VRAM)
```

Birinchi marta `--text` bilan sinab, LLM/tool'lar javob berayotganini
tasdiqlang, keyin to'liq ovozli rejimga o'ting.

**Avtostart (ixtiyoriy):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1
```
Bekor qilish: `scripts\uninstall_autostart.ps1`. Batafsil: `docs/manual_tests_09.md`.

---

## 9. Shaxsiy ma'lumotlar (ixtiyoriy ko'chirish)

Bular git'da yo'q, chunki katta va/yoki shaxsiy. Agar eski mashinada
to'plangan narsalarni davom ettirmoqchi bo'lsangiz, `<data_dir>` ostidagi
quyidagi papkalarni USB/tashqi disk/shaxsiy cloud (Google Drive va h.k. —
**hech qachon git/GitHub emas**) orqali ko'chiring:

| Papka | Nima | Ko'chirish shartmi? |
|---|---|---|
| `recordings/utterances/` | Shaxsiy STT dataset (11-bosqich, `label_tool.py` uchun) | Yo'q — ikkala mashinada ham mustaqil to'planaveradi, keyin ikkalasini ham `build_personal_dataset.py`ga berish mumkin |
| `recordings/wake_auto/`, `wake/`, `wake_neg/` | Wake word namunalari (10-bosqich) | Yo'q, xuddi shu sababdan |
| `db/ultron.sqlite` | Eslatmalar, xotiradagi fakts (07D-bosqich) | Xohishga ko'ra — ko'chirilmasa, yangi mashinada bo'sh xotiradan boshlaydi |
| `datasets/personal/` | `build_personal_dataset.py`ning tayyor train/dev/test'i | Faqat shu mashinada fine-tune qilmoqchi bo'lsangiz kerak |

**Muhim:** bu jarayonda `data_dir`ning o'zi butunlay git tashqarisida
(mashinaning o'z diskida) — shuning uchun ko'chirish shart emas, faqat
**ikkala mashinada alohida davom etishni** yoki **bittasini ikkinchisiga
ko'chirib bitta joyda davom ettirishni** tanlaysiz.

---

## 10. Keyingi safar: yangilanishlarni olish / yuborish

```powershell
git pull                              # boshqa mashinadagi o'zgarishlarni olish
# ... o'zgarish kiritilgach ...
git add -A
git commit -m "izoh"
git push                              # bu mashinadagi o'zgarishlarni GitHub'ga yuborish
```

Ikkala mashinada **parallel** ishlasangiz (masalan ikkalasida ham Claude
Code orqali kod o'zgartirilsa), `git pull`ni ishni boshlashdan oldin doim
qiling — aks holda `git push` to'qnashuv (conflict) beradi.

---

## 11. Muammolarni bartaraf etish

`docs/tz/01_muhit_sozlash.md`ning §7 jadvali eng ko'p uchraydigan muammolarni
qamrab oladi (masalan `cudnn_ops64_9.dll topilmadi`, Ollama CPU'da ishlashi).
Audio bilan bog'liq g'alati xatolar (masalan karnaydan ovoz chiqmasligi)
chiqsa, `docs/PROGRESS.md`ning 13-bosqich yozuviga qarang — bitta shunday
xato allaqachon topilib tuzatilgan (`audio/_device_utils.py`).
