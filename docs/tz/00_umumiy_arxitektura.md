# TZ-00. Umumiy arxitektura va kelishuvlar

> Bu fayl **ma'lumotnoma**. Unda kod yozish vazifasi yo'q. Qolgan barcha TZ'lar shu yerdagi interfeys, papka tuzilishi va config'ga tayanadi.

---

## 1. Maqsad

Windows 11 laptopda doimiy ishlaydigan, **"Altron"** so'zi bilan uyg'onadigan, o'zbek tilida tushunadigan va gapiradigan, kompyuterni boshqaradigan lokal AI yordamchi yaratish.

### 1.1. Asosiy KPI'lar

| Ko'rsatkich | Maqsad (MVP) | Keyingi maqsad |
|---|---|---|
| Wake word'ni tanish (1–3 m masofa, tinch xona) | ≥ 95% | ≥ 99% |
| Yolg'on uyg'onish (fonda gap yoki TV) | ≤ 1 marta/soat | ≤ 1 marta/8 soat |
| Gap tugashidan birinchi javob tovushigacha | ≤ 2.0 s | ≤ 1.2 s |
| Oddiy buyruq (fast-path) bajarilishi | ≤ 1.0 s | ≤ 0.6 s |
| O'zbekcha buyruqlarda STT WER (o'z test to'plami) | ≤ 25% | ≤ 12% |
| Tool tanlash aniqligi (30 ta test buyruq) | ≥ 85% | ≥ 95% |
| Uzluksiz ishlash | 24 soat qulashsiz | 7 kun |
| Laptop yoqilgandan tayyor holatgacha | ≤ 90 s | ≤ 45 s |

---

## 2. Yuqori darajadagi arxitektura

```
                         ┌───────────────────────── Ultron jarayoni (Python, asyncio) ─────────────────────────┐
                         │                                                                                     │
 [Mikrofon] ─► AudioCapture ─► RingBuffer(pre-roll 2s) ─► VAD (Silero, CPU) ─► WakeDetector (CPU/GPU)       │
   16 kHz mono        (thread)                                   │                    │                         │
                                                                 ▼                    ▼ WakeEvent               │
                                                          Endpointing ◄──────── Orchestrator (state machine)    │
                                                                 │ Utterance            │                       │
                                                                 ▼                      │                       │
                                                        STT: faster-whisper (GPU) ──► TextNormalizer (kir→lat)  │
                                                                                        │                       │
                                              ┌── FastPath (regex/fuzzy intent) ◄───────┤                       │
                                              │          │ topildi                      │ topilmadi             │
                                              │          ▼                              ▼                       │
                                              │     ToolRegistry ◄──── tool_calls ── LLM Agent (Ollama/Qwen)    │
                                              │          │                              │ stream (gaplar)       │
                                              │          ▼ natija                       ▼                       │
                                              │     Memory (SQLite)            SentenceSplitter                 │
                                              │     Scheduler (eslatmalar)            │                         │
                                              │                                       ▼                         │
                                              └──────────────────────────► TTS (MMS-VITS, CPU) ─► AudioPlayer ─► [Karnay]
                         │                                                                                     │
                         │  Tray icon (pystray) · Logger · Metrics · Health check                              │
                         └─────────────────────────────────────────────────────────────────────────────────────┘
          Tashqi jarayon: Ollama server (localhost:11434)      Supervisor (watchdog) ─► Ultron jarayonini qayta ishga tushiradi
```

### 2.1. Holat mashinasi (Orchestrator)

```
          ┌──────────── timeout / "rahmat" / "bo'ldi" ──────────────┐
          ▼                                                          │
 ┌──────────┐  WakeEvent   ┌───────────┐  Utterance  ┌──────────┐  javob  ┌──────────┐
 │  IDLE    │ ───────────► │ LISTENING │ ──────────► │ THINKING │ ──────► │ SPEAKING │
 └──────────┘              └───────────┘             └──────────┘         └──────────┘
      ▲   ▲                     ▲  ▲ jimlik (5s)          │ xato               │   │
      │   └─────────────────────┘  │                      ▼                    │   │ WakeEvent (barge-in)
      │                            │               "Kechirasiz, ..."           │   ▼
      │                            └──────── FOLLOW_UP (8 s, wake word'siz) ◄──┘  to'xtatish → LISTENING
      └──────────────────────── FOLLOW_UP timeout ────────────────────────────────┘
   + CONFIRMING holati: xavfli amal oldidan "ha / yo'q" kutiladi (8 s, keyin bekor qilinadi)
```

### 2.2. Ikki xil chaqiruv ssenariysi (ikkalasi ham ishlashi shart)
1. **Alohida:** "Altron" → *(earcon "ding")* → "brauzerni och" → javob.
2. **Bir nafasda:** "Altron, brauzerni och" → wake word'dan keyingi qism buyruq sifatida olinadi, qayta so'ralmaydi.

---

## 3. VRAM va resurs byudjeti

| Komponent | Qurilma | VRAM / RAM | Izoh |
|---|---|---|---|
| Ollama + Qwen 8B Q4_K_M | GPU | ~5.0–5.5 GB VRAM | `num_ctx=8192`, KV cache q8_0, flash attention |
| faster-whisper (turbo, int8_float16) | GPU | ~1.5–2.0 GB VRAM | Doim yuklangan holda turadi |
| KWS uchun Whisper `small` (stt_kws rejimi) | GPU | ~0.5 GB VRAM | `stt.kws_model` = asosiy model bo'lsa, qo'shimcha VRAM kerak emas |
| Silero VAD | CPU | ~50 MB RAM | ONNX |
| openWakeWord / STT-KWS | CPU (+GPU STT) | ~100 MB RAM | |
| MMS-TTS (VITS) | CPU | ~300–500 MB RAM | |
| Python jarayoni jami | — | ~2–3 GB RAM | |
| **VRAM jami** | | **~7.0–7.5 GB** | Windows ekrani uchun iGPU tavsiya qilinadi |

**Qoida:** agar `nvidia-smi` 7.6 GB'dan ko'p VRAM ishlatilayotganini ko'rsatsa, STT modeli `small` yoki `medium`ga tushiriladi, yoki `num_ctx` kamaytiriladi. Bu config orqali qilinadi, kod o'zgarmaydi.

---

## 4. Papka tuzilishi (yakuniy)

```
C:\ultron\
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version                  # 3.11
├── config\
│   ├── config.yaml                  # asosiy config
│   ├── config.local.yaml            # (ixtiyoriy) shaxsiy override, git'ga kirmaydi
│   └── app_aliases.yaml             # "brauzer" → chrome.exe va h.k.
├── docs\
│   ├── PROGRESS.md
│   └── tz\ ...                      # shu TZ fayllari
├── src\ultron\
│   ├── __init__.py
│   ├── __main__.py                  # CLI: python -m ultron [--text] [--debug]
│   ├── _cuda_dlls.py                # Windows'da CUDA DLL yo'llarini qo'shish (birinchi import)
│   ├── config.py                    # pydantic modellari + yuklash
│   ├── logging_setup.py
│   ├── events.py                    # Event dataclass'lari + EventBus
│   ├── i18n\uz.py                   # foydalanuvchiga ko'rinadigan barcha o'zbekcha matnlar
│   ├── audio\
│   │   ├── capture.py               # AudioCapture (sounddevice InputStream)
│   │   ├── ring_buffer.py
│   │   ├── vad.py                   # SileroVAD + Endpointer
│   │   ├── player.py                # AudioPlayer (to'xtatsa bo'ladigan)
│   │   └── earcons.py               # ding/dong/xato tovushlari
│   ├── stt\
│   │   ├── base.py                  # STTEngine Protocol
│   │   ├── whisper_engine.py
│   │   └── filters.py               # hallucination filtrlari
│   ├── text\
│   │   ├── translit.py              # lotin ↔ kirill (o'zbek)
│   │   ├── normalize.py             # STT chiqishini tozalash
│   │   └── numbers_uz.py            # raqam → so'z
│   ├── wake\
│   │   ├── base.py                  # WakeDetector Protocol
│   │   ├── stt_kws.py               # STT asosidagi kalit so'z detektori
│   │   ├── oww_detector.py          # openWakeWord detektori
│   │   └── matcher.py               # "altron" variantlari bilan fuzzy moslashtirish
│   ├── tts\
│   │   ├── base.py                  # TTSEngine Protocol
│   │   ├── mms_engine.py
│   │   └── preprocess.py
│   ├── llm\
│   │   ├── client.py                # Ollama async wrapper
│   │   ├── agent.py                 # tool-calling sikli
│   │   ├── prompts.py               # system prompt (o'zbekcha)
│   │   ├── fastpath.py              # LLM'siz tez intentlar
│   │   └── sentence_splitter.py
│   ├── tools\
│   │   ├── base.py                  # Tool, ToolResult, @tool decorator, ToolRegistry
│   │   ├── system_tools.py
│   │   ├── file_tools.py
│   │   ├── browser_tools.py
│   │   ├── memory_tools.py
│   │   ├── reminder_tools.py
│   │   └── app_index.py             # Start Menu .lnk indekslash
│   ├── memory\
│   │   ├── db.py                    # SQLite (aiosqlite)
│   │   └── scheduler.py             # eslatma/taymer scheduler
│   ├── core\
│   │   ├── orchestrator.py
│   │   ├── state.py
│   │   ├── confirm.py               # "ha/yo'q" tasdiqlash
│   │   └── metrics.py               # latency o'lchash
│   ├── tray\app.py
│   └── supervisor.py                # watchdog (alohida jarayon)
├── scripts\
│   ├── check_env.py
│   ├── download_models.py
│   ├── record_samples.py            # wake word va test to'plam yozish
│   ├── benchmark_stt.py
│   ├── benchmark_wake.py
│   ├── eval_llm_tools.py
│   ├── install_autostart.ps1
│   └── uninstall_autostart.ps1
├── tests\ ...
└── data\                            # git'ga kirmaydi
    ├── models\ (whisper, tts, wake)
    ├── hf_cache\
    ├── db\ultron.sqlite
    ├── logs\
    ├── recordings\ (wake_pos, wake_neg, testset, utterances)
    └── tmp\
```

---

## 5. Config (`config/config.yaml`) — to'liq sxema

Har bir bosqich o'z bo'limini shu sxema bo'yicha to'ldiradi. Kalit nomlari o'zgartirilmaydi.

```yaml
app:
  name: "Ultron"
  language: "uz"
  data_dir: "C:/ultron/data"
  debug: false

audio:
  sample_rate: 16000
  channels: 1
  block_size: 512            # 32 ms, Silero VAD uchun
  input_device: null         # null = default; yoki qurilma nomining bir qismi, masalan "Headset"
  output_device: null
  preroll_seconds: 2.0
  output_volume: 0.9

vad:
  threshold: 0.5
  min_speech_ms: 250
  end_silence_ms: 700        # gap tugadi deb hisoblash uchun jimlik
  max_utterance_s: 15
  listen_timeout_s: 6        # wake word'dan keyin gap boshlanmasa

wake:
  mode: "stt_kws"            # "stt_kws" | "oww" | "hybrid"
  phrases: ["altron", "ultron", "oltron", "altiron", "ultiron", "алтрон", "ультрон", "олтрон"]
  fuzzy_threshold: 82        # rapidfuzz ratio, 0–100
  max_prefix_words: 2        # wake word gapning birinchi N so'zida bo'lishi kerak
  oww_model_path: "data/models/wake/altron.onnx"
  oww_threshold: 0.35
  oww_verify_with_stt: true  # oww trigger'ini STT bilan tasdiqlash
  cooldown_ms: 1200
  barge_in_enabled: true
  barge_in_threshold_boost: 0.15

stt:
  engine: "faster_whisper"
  model: "large-v3-turbo"    # yoki lokal CT2 papka yo'li (o'zbekcha fine-tune)
  device: "cuda"
  compute_type: "int8_float16"
  language: "uz"
  beam_size: 1
  kws_model: "small"         # stt_kws rejimi uchun yengil model (cpu yoki cuda)
  kws_device: "cuda"
  initial_prompt: "Altron, brauzerni och. Ovozni balandlat. Telegramni och. Eslatma qo'y."
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  output_script: "latin"     # har doim lotinga normallashtiriladi

llm:
  host: "http://127.0.0.1:11434"
  model: "qwen3:8b"          # Ollama'dagi mavjud Qwen 7–9B tool-calling modeli
  think: false               # Qwen3 "thinking" rejimi o'chirilgan (latency uchun)
  temperature: 0.3
  num_ctx: 8192
  keep_alive: "-1"           # modelni VRAM'da doim ushlab turish
  max_tool_iterations: 5
  history_turns: 8
  request_timeout_s: 60

fastpath:
  enabled: true

tts:
  engine: "mms"
  model: "facebook/mms-tts-uzb-script_cyrillic"   # 05-bosqichda HF'da tekshiriladi
  model_script: "cyrillic"   # model qaysi yozuvda o'qitilgan: "cyrillic" | "latin"
  device: "cpu"
  speed: 1.0
  sentence_pause_ms: 150

tools:
  confirm_dangerous: true
  confirm_timeout_s: 8
  files:
    allowed_roots: ["~/Desktop", "~/Documents", "~/Downloads", "~/Pictures", "~/Music", "~/Videos", "D:/"]
    max_results: 10
  browser:
    default: "chrome"        # "chrome" | "edge" | "default"
    search_engine: "google"
  internet_tools_enabled: true   # ob-havo kabi ochiq API'lar (AI emas)
  weather:
    city: "Tashkent"
    lat: 41.2995
    lon: 69.2401

memory:
  db_path: "data/db/ultron.sqlite"
  max_facts_in_prompt: 40
  save_utterances: true      # 11-bosqich uchun audio + transkript saqlash

conversation:
  follow_up_window_s: 8
  end_phrases: ["rahmat", "bo'ldi", "kerak emas", "yaxshi bo'ldi"]
  cancel_phrases: ["to'xta", "bekor qil", "jim", "stop"]

autostart:
  startup_delay_s: 20
  announce_ready: true
  ollama_start_if_missing: true

logging:
  level: "INFO"
  rotation: "10 MB"
  retention: "14 days"
```

---

## 6. Asosiy interfeyslar (barcha bosqichlar uchun majburiy)

```python
# events.py
from dataclasses import dataclass, field
import numpy as np, time

@dataclass
class AudioFrame:
    pcm: np.ndarray            # int16, shape (block_size,)
    ts: float = field(default_factory=time.monotonic)

@dataclass
class Utterance:
    pcm: np.ndarray            # int16, 16 kHz, pre-roll bilan
    started_at: float
    ended_at: float
    reason: str                # "silence" | "max_length" | "cancelled"

@dataclass
class WakeEvent:
    ts: float
    score: float               # 0..1
    source: str                # "stt_kws" | "oww"
    trailing_text: str | None  # "Altron, brauzerni och" → "brauzerni och"

@dataclass
class Transcript:
    text: str                  # normallashtirilgan, lotin yozuvida
    raw_text: str
    language: str
    confidence: float
    duration_s: float
    latency_ms: int
```

```python
# stt/base.py
class STTEngine(Protocol):
    def load(self) -> None: ...
    def transcribe(self, pcm: np.ndarray, *, initial_prompt: str | None = None) -> Transcript: ...

# wake/base.py
class WakeDetector(Protocol):
    def load(self) -> None: ...
    def process_frame(self, frame: AudioFrame) -> WakeEvent | None: ...        # oww uchun
    def process_utterance(self, utt: Utterance) -> WakeEvent | None: ...       # stt_kws uchun
    def reset(self) -> None: ...

# tts/base.py
class TTSEngine(Protocol):
    sample_rate: int
    def load(self) -> None: ...
    def synthesize(self, text: str) -> np.ndarray: ...   # float32 [-1, 1]

# tools/base.py
@dataclass
class ToolResult:
    ok: bool
    data: dict | str | None = None
    speak: str | None = None          # foydalanuvchiga aytiladigan qisqa matn (ixtiyoriy)
    error: str | None = None

# @tool(name="open_app", description="...", dangerous=False) decorator
# ToolRegistry.schemas() -> list[dict]   (Ollama/OpenAI tools formati)
# await ToolRegistry.call(name, args) -> ToolResult
```

---

## 7. Umumiy qoidalar
1. **Lokallik.** Faqat `localhost` (Ollama) va brauzer/ixtiyoriy ochiq API'larga tarmoq so'rovi yuboriladi. Modellar faqat o'rnatish vaqtida yuklab olinadi.
2. **Til.** STT chiqishi har doim o'zbek **lotin** yozuviga normallashtiriladi. LLM lotinda ishlaydi. TTS model talab qilsa, kirillga o'tkazadi.
3. **Xavfsizlik.** `dangerous=True` tool'lar tasdiqsiz ishlamaydi. Barcha tool chaqiruvlari `data/logs/tools_audit.log` fayliga yoziladi.
4. **Chidamlilik.** Har bir komponent xatosi butun jarayonni yiqitmasligi kerak. Xato log'ga yoziladi, foydalanuvchiga qisqa aytiladi va holat IDLE'ga qaytadi.
5. **O'lchash.** Har bir so'rov uchun `vad_end → stt_done → llm_first_token → tts_first_audio` vaqtlari `metrics` orqali log'ga yoziladi.
6. **Ultron o'z ismini aytmaydi.** TTS matnidan "Ultron/Altron" so'zlari olib tashlanadi, aks holda o'zini o'zi uyg'otadi (self-trigger).
