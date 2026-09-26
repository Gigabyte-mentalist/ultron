"""Environment readiness check for Ultron (docs/tz/01_muhit_sozlash.md §4.6).

Run:
    uv run python scripts/check_env.py
    uv run python scripts/check_env.py --no-sound     # skip the audible speaker test
"""

from __future__ import annotations

import argparse
import ctypes
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron import _cuda_dlls  # noqa: F401  # must precede ctranslate2/torch imports
from ultron.config import ConfigError, RootConfig, load_config

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str | None = None


def check_python() -> CheckResult:
    major, minor = sys.version_info[:2]
    ok = (major, minor) == (3, 11)
    return CheckResult(
        "Python 3.11",
        ok,
        sys.version.split()[0],
        None if ok else "uv python install 3.11, keyin uv sync",
    )


def check_gpu() -> CheckResult:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        return CheckResult("NVIDIA GPU / drayver / VRAM", True, line)
    except Exception as exc:
        return CheckResult(
            "NVIDIA GPU / drayver / VRAM",
            False,
            str(exc),
            "NVIDIA drayverini o'rnating/yangilang (GeForce Experience / NVIDIA App)",
        )


def check_ctranslate2_cuda() -> CheckResult:
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        return CheckResult("CTranslate2 CUDA", count > 0, f"{count} ta CUDA qurilma")
    except ImportError:
        return CheckResult("CTranslate2 CUDA", False, "ctranslate2 o'rnatilmagan", "uv sync")
    except Exception as exc:
        return CheckResult(
            "CTranslate2 CUDA",
            False,
            str(exc),
            "nvidia-cublas-cu12 / nvidia-cudnn-cu12 o'rnatilganini tekshiring",
        )


def check_cuda_dlls() -> CheckResult:
    if sys.platform != "win32":
        return CheckResult("cuBLAS/cuDNN DLL", True, "Windows emas, o'tkazib yuborildi")
    missing = []
    for dll_name in ("cublas64_12.dll", "cudnn_ops64_9.dll"):
        try:
            ctypes.WinDLL(dll_name)
        except OSError:
            missing.append(dll_name)
    ok = not missing
    hint = (
        "_cuda_dlls.py birinchi import qilinganini va "
        "nvidia-cudnn-cu12>=9,<10 o'rnatilganini tekshiring"
    )
    return CheckResult(
        "cuBLAS/cuDNN DLL",
        ok,
        "topildi" if ok else f"topilmadi: {', '.join(missing)}",
        None if ok else hint,
    )


def check_onnxruntime() -> CheckResult:
    try:
        import onnxruntime as ort

        return CheckResult(
            "onnxruntime",
            True,
            f"{ort.__version__}, providers={ort.get_available_providers()}",
        )
    except ImportError:
        return CheckResult("onnxruntime", False, "o'rnatilmagan", "uv sync")


def check_torch() -> CheckResult:
    try:
        import torch

        return CheckResult(
            "torch",
            True,
            f"{torch.__version__}, cuda_available={torch.cuda.is_available()} "
            "(ma'lumot uchun, TTS CPU'da ishlaydi)",
        )
    except ImportError:
        return CheckResult("torch", False, "o'rnatilmagan", "uv sync")


def check_ollama_server(config: RootConfig) -> CheckResult:
    try:
        import httpx

        resp = httpx.get(f"{config.llm.host}/api/tags", timeout=5)
        resp.raise_for_status()
        return CheckResult("Ollama server", True, f"{config.llm.host} javob berdi")
    except ImportError:
        return CheckResult("Ollama server", False, "httpx o'rnatilmagan", "uv sync")
    except Exception as exc:
        return CheckResult(
            "Ollama server",
            False,
            str(exc),
            "Ollama'ni o'rnating (winget install Ollama.Ollama) va tray'dan ishga tushiring",
        )


def check_qwen_model(config: RootConfig) -> CheckResult:
    try:
        import httpx

        resp = httpx.get(f"{config.llm.host}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
        ok = config.llm.model in tags or any(tag.startswith(f"{config.llm.model}:") for tag in tags)
        return CheckResult(
            "Qwen modeli mavjud",
            ok,
            f"kerak: {config.llm.model}; mavjud: {tags}",
            None if ok else f"ollama pull {config.llm.model}",
        )
    except Exception as exc:
        return CheckResult(
            "Qwen modeli mavjud", False, str(exc), "Avval Ollama serverni ishga tushiring"
        )


def check_qwen_tool_calling(config: RootConfig) -> CheckResult:
    try:
        import ollama

        client = ollama.Client(host=config.llm.host)
        response = client.chat(
            model=config.llm.model,
            messages=[{"role": "user", "content": "Hozir soat nechchi? /no_think"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Joriy vaqtni qaytaradi",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        message = response.get("message") if isinstance(response, dict) else response.message
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None)
        ok = bool(tool_calls)
        return CheckResult(
            "Qwen tool calling",
            ok,
            f"tool_calls={tool_calls}",
            None
            if ok
            else "Model tool calling'ni qo'llamasligi mumkin; config'dagi llm.model'ni tekshiring",
        )
    except ImportError:
        return CheckResult("Qwen tool calling", False, "ollama paketi o'rnatilmagan", "uv sync")
    except Exception as exc:
        return CheckResult(
            "Qwen tool calling", False, str(exc), "Ollama server va model tayyorligini tekshiring"
        )


def check_microphone() -> CheckResult:
    try:
        import numpy as np
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            return CheckResult(
                "Mikrofon",
                False,
                "kirish qurilmasi topilmadi",
                "Settings > Privacy & security > Microphone'ni yoqing",
            )
        recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        rms = float(np.sqrt(np.mean(recording.astype(np.float64) ** 2)))
        ok = rms > 0
        return CheckResult(
            "Mikrofon",
            ok,
            f"RMS={rms:.2f}",
            None if ok else "Mikrofon ruxsatlarini va default qurilmani tekshiring",
        )
    except ImportError:
        return CheckResult("Mikrofon", False, "sounddevice o'rnatilmagan", "uv sync")
    except Exception as exc:
        return CheckResult(
            "Mikrofon", False, str(exc), "Settings > Privacy & security > Microphone'ni tekshiring"
        )


def check_speaker(play_sound: bool) -> CheckResult:
    if not play_sound:
        return CheckResult("Karnay", True, "--no-sound bilan o'tkazib yuborildi")
    try:
        import numpy as np
        import sounddevice as sd

        t = np.linspace(0, 0.5, int(0.5 * 16000), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sd.play(tone, samplerate=16000)
        sd.wait()
        return CheckResult("Karnay", True, "440 Hz sinov tovushi chalindi")
    except ImportError:
        return CheckResult("Karnay", False, "sounddevice o'rnatilmagan", "uv sync")
    except Exception as exc:
        return CheckResult("Karnay", False, str(exc), "Chiqish qurilmasini tekshiring")


def check_disk(config: RootConfig) -> CheckResult:
    try:
        data_dir = Path(config.app.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(data_dir)
        free_gb = usage.free / (1024**3)
        ok = free_gb >= 30
        return CheckResult(
            "Disk (>= 30 GB bo'sh)",
            ok,
            f"{free_gb:.1f} GB bo'sh ({data_dir})",
            None if ok else "data_dir joylashgan diskda joy bo'shating",
        )
    except Exception as exc:
        return CheckResult("Disk (>= 30 GB bo'sh)", False, str(exc))


def check_paths(config: RootConfig) -> CheckResult:
    try:
        data_dir = Path(config.app.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        has_cyrillic = bool(_CYRILLIC_RE.search(str(data_dir)))
        ok = not has_cyrillic
        return CheckResult(
            "Yo'llar (yozish mumkin, kirill yo'q)",
            ok,
            str(data_dir),
            None if ok else "HF_HOME va app.data_dir'ni kirill harflarisiz yo'lga o'zgartiring",
        )
    except Exception as exc:
        return CheckResult("Yo'llar (yozish mumkin, kirill yo'q)", False, str(exc))


def run_checks(config: RootConfig, play_sound: bool) -> list[CheckResult]:
    return [
        check_python(),
        check_gpu(),
        check_ctranslate2_cuda(),
        check_cuda_dlls(),
        check_onnxruntime(),
        check_torch(),
        check_ollama_server(config),
        check_qwen_model(config),
        check_qwen_tool_calling(config),
        check_microphone(),
        check_speaker(play_sound),
        check_disk(config),
        check_paths(config),
    ]


def report(results: list[CheckResult]) -> bool:
    all_ok = True
    for i, result in enumerate(results, start=1):
        mark = "✅" if result.ok else "❌"
        logger.info("{:2d}. {} {}: {}", i, mark, result.name, result.detail)
        if not result.ok:
            all_ok = False
            if result.hint:
                logger.info("      -> {}", result.hint)
    logger.info("")
    if all_ok:
        logger.info("✅ Hammasi tayyor!")
    else:
        logger.info("❌ Ba'zi tekshiruvlar o'tmadi, yuqoridagi maslahatlarga qarang.")
    return all_ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron muhit tekshiruvi")
    parser.add_argument("--no-sound", action="store_true", help="Karnay sinovini o'tkazib yuborish")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        logger.info("❌ Config yuklab bo'lmadi:\n{}", exc)
        return 1

    results = run_checks(config, play_sound=not args.no_sound)
    ok = report(results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
