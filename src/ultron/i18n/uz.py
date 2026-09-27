"""User-facing Uzbek strings (docs/tz/00_umumiy_arxitektura.md §4, §7.6).

Every string spoken or shown to the user is meant to live here rather than
inline in the module that uses it, so later stages (LLM/orchestrator) reuse
the same wording instead of re-typing it. ``CACHED_PHRASES`` lists exactly
the short, frequently-spoken replies that docs/tz/05_tts.md §3.2 requires
the TTS engine to pre-synthesize at startup so they play with ~0 ms latency.
"""

from __future__ import annotations

ACKNOWLEDGE = "Ha, xo'jayin"
DONE = "Bajarildi"
DIDNT_UNDERSTAND = "Kechirasiz, tushunmadim"
CONFIRM_PROMPT = "Tasdiqlaysizmi?"
LLM_UNAVAILABLE = "Hozir o'ylay olmayapman, bir oz kuting"
TOOL_ITERATIONS_EXCEEDED = "Kechirasiz, buni bajara olmadim"

# docs/tz/08_orchestrator.md: orchestrator-level canned phrases.
READY = "Tayyorman"
GENERIC_ERROR = "Kechirasiz, xatolik yuz berdi"
LISTEN_RETRY = "Tushunmadim, qaytaring"
WAIT_A_MOMENT = "Bir daqiqa..."
FOLLOW_UP_CLOSING = "Xizmatingizdaman"
REMINDER_PREFIX = "Eslatma: "
REMINDER_OVERDUE_PREFIX = "O'tkazib yuborilgan eslatma: "

# docs/tz/09_autostart.md §3.6: health monitor's spoken/toast messages.
OLLAMA_RECONNECTING = "Miyam ishlamayapti, qayta ulanyapman"
REPEATED_CRASH_TOAST = "Ultron qayta-qayta to'xtayapti, log'ni tekshiring"

# docs/tz/09_autostart.md §3.4: tray menu (o'zbekcha).
TRAY_MUTE_MIC = "Mikrofonni o'chirish"
TRAY_UNMUTE_MIC = "Mikrofonni yoqish"
TRAY_TEXT_COMMAND = "Matnli buyruq..."
TRAY_OPEN_LOGS = "Log papkasini ochish"
TRAY_OPEN_CONFIG = "Config'ni ochish"
TRAY_GPU_SAVER_ON = "GPU'ni bo'shatish"
TRAY_GPU_SAVER_OFF = "Normal rejim"
TRAY_RESTART = "Qayta ishga tushirish"
TRAY_QUIT = "Chiqish"
TRAY_TEXT_COMMAND_TITLE = "Ultron — matnli buyruq"
TRAY_TEXT_COMMAND_LABEL = "Buyruqni yozing:"
TRAY_TEXT_COMMAND_SUBMIT = "Yuborish"

# ultron.companion: vizual hamroh oynasi tray menyusi.
TRAY_COMPANION_SHOW = "Hamrohni ko'rsatish"
TRAY_COMPANION_HIDE = "Hamrohni yashirish"

CACHED_PHRASES: tuple[str, ...] = (
    ACKNOWLEDGE,
    DONE,
    DIDNT_UNDERSTAND,
    CONFIRM_PROMPT,
    READY,
    GENERIC_ERROR,
    LISTEN_RETRY,
    WAIT_A_MOMENT,
    FOLLOW_UP_CLOSING,
    OLLAMA_RECONNECTING,
)

# Monday-first, matching ``datetime.date.weekday()`` (docs/tz/06_llm.md §3.3,
# also used by ``tools/demo_tools.py``'s ``get_date``).
WEEKDAYS_UZ: tuple[str, ...] = (
    "dushanba",
    "seshanba",
    "chorshanba",
    "payshanba",
    "juma",
    "shanba",
    "yakshanba",
)
