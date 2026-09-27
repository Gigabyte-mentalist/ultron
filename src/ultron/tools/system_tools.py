"""System control tools: apps, volume, brightness, power, screenshot, status
(docs/tz/07_tools.md §7A).

Every state-changing tool checks :func:`ultron.tools.base.is_mock_mode` first
and, when set, only logs what it would have done -- see that function's
docstring for why. Windows-COM-backed calls (pycaw) run inside
``asyncio.to_thread`` with ``comtypes.CoInitialize()``/``CoUninitialize()``
bracketing the call, since COM apartments are per-thread and the thread pool
otherwise leaves them uninitialized (docs/tz/07_tools.md §1).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import datetime
import os
import subprocess
from pathlib import Path
from typing import Annotated, Any

import psutil
from loguru import logger
from rapidfuzz import fuzz, process

from ultron.config import RootConfig
from ultron.i18n.uz import WEEKDAYS_UZ
from ultron.text.normalize import strip_uz_suffixes
from ultron.tools.app_index import AppEntry, AppIndex
from ultron.tools.base import ToolResult, is_mock_mode, tool

# Never terminated by close_app, whatever the user asks for.
_PROTECTED_PROCESSES: frozenset[str] = frozenset(
    {
        "explorer.exe",
        "csrss.exe",
        "winlogon.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "svchost.exe",
        "smss.exe",
        "system",
        "registry",
        "python.exe",
        "pythonw.exe",
        "ultron.exe",
        "ollama.exe",
        "ollama app.exe",
    }
)

_DEFAULT_ALIASES_PATH = Path("config/app_aliases.yaml")

_config: RootConfig | None = None
_app_index: AppIndex | None = None


def configure(config: RootConfig) -> None:
    """Called once at startup so tool bodies (whose signatures are entirely
    LLM-facing) can still reach config values."""
    global _config, _app_index
    _config = config
    _app_index = AppIndex(config.tools.app_aliases_path)


def _get_app_index() -> AppIndex:
    global _app_index
    if _app_index is None:
        _app_index = AppIndex(_DEFAULT_ALIASES_PATH)
    return _app_index


async def _ensure_app_index_scanned() -> AppIndex:
    index = _get_app_index()
    if not index.scanned:
        await asyncio.to_thread(index.scan)
    return index


async def ensure_app_index_scanned() -> None:
    """Public entry point for docs/tz/08_orchestrator.md's startup warmup
    (§8A.1's "app indeksi" parallel-load step): scans the Start Menu/Store
    app index proactively so the *first* real "... ni och" command doesn't
    pay the PowerShell/``.lnk``-walk cost. A no-op if already scanned (e.g.
    a prior ``open_app``/``close_app`` call, or a repeated warmup call)."""
    await _ensure_app_index_scanned()


def _with_com(func: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import comtypes

        comtypes.CoInitialize()
        try:
            return func(*args, **kwargs)
        finally:
            comtypes.CoUninitialize()

    return wrapper


# -- apps ---------------------------------------------------------------


@tool(
    name="open_app",
    description=(
        "Open an installed application, a Windows Settings page, or a UWP/Store"
        " app by the name the user spoke (Uzbek, may include grammatical suffixes)."
    ),
)
async def open_app(
    name: Annotated[str, "Application name as the user said it, e.g. 'telegram', 'brauzer'"],
) -> ToolResult:
    index = await _ensure_app_index_scanned()
    match = index.find(name)

    if match is None:
        return ToolResult(ok=False, error=f"'{name}' nomli dastur topilmadi")
    if isinstance(match, list):
        candidates = [entry.display_name for entry in match]
        return ToolResult(
            ok=False,
            data={"candidates": candidates},
            speak="Bir nechta mos dastur topildi, qaysi birini demoqchisiz?",
        )

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] open_app -> {} ({})", match.display_name, match.target)
        return ToolResult(
            ok=True, data={"opened": match.display_name}, speak=f"{match.display_name} ochildi."
        )

    try:
        if match.kind == "store":
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{match.target}"])  # noqa: S603
        else:
            os.startfile(match.target)  # type: ignore[attr-defined]
    except OSError as exc:
        return ToolResult(ok=False, error=f"'{match.display_name}'ni ochib bo'lmadi: {exc}")

    return ToolResult(
        ok=True, data={"opened": match.display_name}, speak=f"{match.display_name} ochildi."
    )


def _resolve_exe_name_for_close(name: str, match: AppEntry | list[AppEntry] | None) -> str | None:
    if isinstance(match, AppEntry) and match.exe_name:
        return match.exe_name

    target = strip_uz_suffixes(name)
    running_names = {
        proc.info["name"].lower() for proc in psutil.process_iter(["name"]) if proc.info.get("name")
    }
    if not running_names:
        return None
    best = process.extractOne(target, running_names, scorer=fuzz.WRatio, score_cutoff=75)
    return best[0] if best else None


def _is_protected(exe_name: str) -> bool:
    if exe_name in _PROTECTED_PROCESSES:
        return True
    with contextlib.suppress(psutil.Error):
        if exe_name == psutil.Process().name().lower():
            return True
    return False


@tool(name="close_app", description="Close a running application by the name the user spoke.")
async def close_app(name: Annotated[str, "Application name as the user said it"]) -> ToolResult:
    index = await _ensure_app_index_scanned()
    match = index.find(name)
    display_name = match.display_name if isinstance(match, AppEntry) else name

    exe_name = await asyncio.to_thread(_resolve_exe_name_for_close, name, match)
    if exe_name is None:
        return ToolResult(ok=False, error=f"'{display_name}' hozir ishlamayapti")

    if _is_protected(exe_name):
        return ToolResult(ok=False, error="Bu tizim jarayonini yopib bo'lmaydi")

    matched = [
        proc
        for proc in psutil.process_iter(["name"])
        if (proc.info.get("name") or "").lower() == exe_name
    ]
    if not matched:
        return ToolResult(ok=False, error=f"'{display_name}' hozir ishlamayapti")

    if is_mock_mode():
        logger.bind(audit=True).info(
            "[MOCK] close_app -> {} ({} jarayon)", display_name, len(matched)
        )
        return ToolResult(ok=True, data={"closed": display_name}, speak=f"{display_name} yopildi.")

    for proc in matched:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.terminate()
    _gone, alive = await asyncio.to_thread(psutil.wait_procs, matched, 3)
    for proc in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.kill()

    return ToolResult(ok=True, data={"closed": display_name}, speak=f"{display_name} yopildi.")


def _visible_window_process_names() -> list[str]:
    import win32gui
    import win32process

    names: set[str] = set()

    def _callback(hwnd: int, _extra: Any) -> bool:
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                names.add(psutil.Process(pid).name())
        return True

    win32gui.EnumWindows(_callback, None)
    return sorted(names)


@tool(
    name="list_running_apps",
    description="List currently running applications with a visible window.",
)
async def list_running_apps() -> ToolResult:
    names = await asyncio.to_thread(_visible_window_process_names)
    if not names:
        return ToolResult(ok=True, data={"apps": []}, speak="Hozir ochiq oyna yo'q.")
    return ToolResult(ok=True, data={"apps": names}, speak="Ochiq dasturlar: " + ", ".join(names))


# -- volume / brightness -------------------------------------------------


def _volume_interface() -> Any:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _do_set_volume(level: int | None, delta: int | None) -> int:
    volume = _volume_interface()
    if delta is not None:
        target = volume.GetMasterVolumeLevelScalar() * 100 + delta
    else:
        target = float(level)  # type: ignore[arg-type]
    target = max(0, min(100, target))
    volume.SetMasterVolumeLevelScalar(target / 100.0, None)
    return int(round(target))


def _do_mute(muted: bool) -> None:
    _volume_interface().SetMute(1 if muted else 0, None)


def _do_get_volume() -> tuple[int, bool]:
    volume = _volume_interface()
    pct = int(round(volume.GetMasterVolumeLevelScalar() * 100))
    return pct, bool(volume.GetMute())


@tool(
    name="set_volume",
    description="Set system speaker volume to an absolute percent (0-100), or adjust it"
    " by a relative delta.",
)
async def set_volume(
    level: Annotated[int, "Absolute volume percent, 0-100"] | None = None,
    delta: Annotated[int, "Relative change in percent, e.g. -10 or 10"] | None = None,
) -> ToolResult:
    if level is None and delta is None:
        return ToolResult(ok=False, error="level yoki delta berilishi kerak")

    if is_mock_mode():
        pct = max(0, min(100, level if level is not None else 50))
        logger.bind(audit=True).info("[MOCK] set_volume level={} delta={}", level, delta)
        return ToolResult(ok=True, data={"volume": pct}, speak=f"Ovoz {pct} foizga o'rnatildi.")

    try:
        new_pct = await asyncio.to_thread(_with_com(_do_set_volume), level, delta)
    except Exception as exc:  # noqa: BLE001 -- COM/driver errors vary by hardware
        return ToolResult(ok=False, error=f"Ovozni o'zgartirib bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"volume": new_pct}, speak=f"Ovoz {new_pct} foizga o'rnatildi.")


@tool(name="mute", description="Mute or unmute system audio output.")
async def mute(muted: Annotated[bool, "True to mute, False to unmute"]) -> ToolResult:
    speak = "Ovoz o'chirildi." if muted else "Ovoz yoqildi."
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] mute muted={}", muted)
        return ToolResult(ok=True, data={"muted": muted}, speak=speak)

    try:
        await asyncio.to_thread(_with_com(_do_mute), muted)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"Ovozni o'zgartirib bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"muted": muted}, speak=speak)


@tool(name="get_volume", description="Return the current speaker volume percent and mute state.")
async def get_volume() -> ToolResult:
    if is_mock_mode():
        return ToolResult(ok=True, data={"volume": 50, "muted": False}, speak="Ovoz 50 foiz.")

    try:
        pct, muted = await asyncio.to_thread(_with_com(_do_get_volume))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"Ovozni o'qib bo'lmadi: {exc}")
    speak = "Ovoz o'chirilgan." if muted else f"Ovoz {pct} foiz."
    return ToolResult(ok=True, data={"volume": pct, "muted": muted}, speak=speak)


def _do_set_brightness(level: int | None, delta: int | None) -> int:
    import screen_brightness_control as sbc

    if delta is not None:
        current = sbc.get_brightness(display=0)[0]
        target = current + delta
    else:
        target = level  # type: ignore[assignment]
    target = max(0, min(100, target))
    sbc.set_brightness(target)
    return target


@tool(
    name="set_brightness",
    description="Set laptop screen brightness to an absolute percent (0-100), or adjust it"
    " by a relative delta.",
)
async def set_brightness(
    level: Annotated[int, "Absolute brightness percent, 0-100"] | None = None,
    delta: Annotated[int, "Relative change in percent"] | None = None,
) -> ToolResult:
    if level is None and delta is None:
        return ToolResult(ok=False, error="level yoki delta berilishi kerak")

    if is_mock_mode():
        pct = max(0, min(100, level if level is not None else 50))
        logger.bind(audit=True).info("[MOCK] set_brightness level={} delta={}", level, delta)
        return ToolResult(
            ok=True, data={"brightness": pct}, speak=f"Yorug'lik {pct} foizga o'rnatildi."
        )

    try:
        new_pct = await asyncio.to_thread(_do_set_brightness, level, delta)
    except Exception as exc:  # noqa: BLE001 -- no brightness control on some displays
        return ToolResult(ok=False, error=f"Yorug'likni o'zgartirib bo'lmadi: {exc}")
    return ToolResult(
        ok=True, data={"brightness": new_pct}, speak=f"Yorug'lik {new_pct} foizga o'rnatildi."
    )


# -- power ----------------------------------------------------------------


@tool(name="lock_screen", description="Lock the Windows session immediately.")
async def lock_screen() -> ToolResult:
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] lock_screen")
        return ToolResult(ok=True, speak="Kompyuter qulflandi.")
    await asyncio.to_thread(ctypes.windll.user32.LockWorkStation)
    return ToolResult(ok=True, speak="Kompyuter qulflandi.")


@tool(
    name="sleep_pc",
    description="Put the computer to sleep.",
    dangerous=True,
    confirm_question_uz="Kompyuterni uyquga yuborishni tasdiqlaysizmi?",
)
async def sleep_pc() -> ToolResult:
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] sleep_pc")
        return ToolResult(ok=True, speak="Kompyuter uyquga ketmoqda.")
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return ToolResult(ok=False, error=f"Uyquga ketolmadi: {exc}")
    return ToolResult(ok=True, speak="Kompyuter uyquga ketmoqda.")


@tool(
    name="shutdown_pc",
    description="Shut down the computer after a delay in seconds (default 30);"
    " cancelable via cancel_shutdown.",
    dangerous=True,
    confirm_question_uz="Kompyuterni o'chirishni tasdiqlaysizmi?",
)
async def shutdown_pc(
    delay_s: Annotated[int, "Seconds to wait before shutting down"] = 30,
) -> ToolResult:
    delay_s = max(0, delay_s)
    speak = f"{delay_s} soniyadan keyin o'chadi, bekor qilish uchun 'bekor qil' deng."
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] shutdown_pc delay_s={}", delay_s)
        return ToolResult(ok=True, data={"delay_s": delay_s}, speak=speak)
    try:
        await asyncio.to_thread(subprocess.run, ["shutdown", "/s", "/t", str(delay_s)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return ToolResult(ok=False, error=f"O'chirib bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"delay_s": delay_s}, speak=speak)


@tool(
    name="restart_pc",
    description="Restart the computer after a delay in seconds (default 30);"
    " cancelable via cancel_shutdown.",
    dangerous=True,
    confirm_question_uz="Kompyuterni qayta yuklashni tasdiqlaysizmi?",
)
async def restart_pc(
    delay_s: Annotated[int, "Seconds to wait before restarting"] = 30,
) -> ToolResult:
    delay_s = max(0, delay_s)
    speak = f"{delay_s} soniyadan keyin qayta yuklanadi, bekor qilish uchun 'bekor qil' deng."
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] restart_pc delay_s={}", delay_s)
        return ToolResult(ok=True, data={"delay_s": delay_s}, speak=speak)
    try:
        await asyncio.to_thread(subprocess.run, ["shutdown", "/r", "/t", str(delay_s)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return ToolResult(ok=False, error=f"Qayta yuklab bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"delay_s": delay_s}, speak=speak)


@tool(name="cancel_shutdown", description="Cancel a pending shutdown or restart.")
async def cancel_shutdown() -> ToolResult:
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] cancel_shutdown")
        return ToolResult(ok=True, speak="O'chirish bekor qilindi.")
    proc = await asyncio.to_thread(subprocess.run, ["shutdown", "/a"], capture_output=True)
    if proc.returncode != 0:
        return ToolResult(ok=False, error="Bekor qilinadigan o'chirish topilmadi")
    return ToolResult(ok=True, speak="O'chirish bekor qilindi.")


# -- screenshot / status ----------------------------------------------------


def _do_screenshot(out_dir: Path) -> Path:
    import mss
    import mss.tools

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
        mss.tools.to_png(shot.rgb, shot.size, output=str(path))
    return path


@tool(name="screenshot", description="Take a screenshot of the whole desktop and save it to disk.")
async def screenshot() -> ToolResult:
    out_dir = Path.home() / "Pictures" / "Ultron"
    if is_mock_mode():
        path = out_dir / f"screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
        logger.bind(audit=True).info("[MOCK] screenshot -> {}", path)
        return ToolResult(ok=True, data={"path": str(path)}, speak="Skrinshot olindi.")

    try:
        path = await asyncio.to_thread(_do_screenshot, out_dir)
    except Exception as exc:  # noqa: BLE001 -- mss backend errors vary by display setup
        return ToolResult(ok=False, error=f"Skrinshot olib bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"path": str(path)}, speak="Skrinshot olindi.")


def _gpu_vram_used_mb() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return int(proc.stdout.strip().splitlines()[0])
    except ValueError:
        return None


def _collect_system_status() -> dict[str, Any]:
    battery = psutil.sensors_battery()
    disk = psutil.disk_usage(str(Path.home().anchor or "C:\\"))
    status: dict[str, Any] = {
        "battery_percent": int(battery.percent) if battery else None,
        "battery_plugged": bool(battery.power_plugged) if battery else None,
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_free_gb": round(disk.free / 1e9, 1),
    }
    vram = _gpu_vram_used_mb()
    if vram is not None:
        status["gpu_vram_used_mb"] = vram
    return status


@tool(
    name="get_system_status",
    description="Return battery, CPU, RAM, GPU VRAM, and free disk space status.",
)
async def get_system_status() -> ToolResult:
    status = await asyncio.to_thread(_collect_system_status)
    parts = []
    if status["battery_percent"] is not None:
        parts.append(f"batareya {status['battery_percent']} foiz")
    parts.append(f"protsessor {status['cpu_percent']:.0f} foiz")
    parts.append(f"xotira {status['ram_percent']:.0f} foiz")
    return ToolResult(ok=True, data=status, speak="Holat: " + ", ".join(parts) + ".")


# -- time / date (moved from tools/demo_tools.py, docs/tz/07_tools.md §7A.2) --


@tool(name="get_time", description="Return the current local time (hour and minute).")
async def get_time() -> ToolResult:
    now = datetime.datetime.now()
    return ToolResult(
        ok=True,
        data={"hour": now.hour, "minute": now.minute},
        speak=f"Hozir soat {now.strftime('%H:%M')}.",
    )


@tool(name="get_date", description="Return today's calendar date and weekday name.")
async def get_date() -> ToolResult:
    today = datetime.date.today()
    weekday_uz = WEEKDAYS_UZ[today.weekday()]
    return ToolResult(
        ok=True,
        data={"day": today.day, "month": today.month, "year": today.year, "weekday": weekday_uz},
        speak=f"Bugun {today.day}.{today.month}.{today.year}, {weekday_uz}.",
    )
