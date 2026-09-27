"""Tests for ultron.tools.system_tools (docs/tz/07_tools.md §7A).

Nothing here ever touches the real system: state-changing tools are tested
in mock mode (``ULTRON_TOOLS_MOCK=1``, per docs/tz/07_tools.md §1), and the
one "real mode" test per tool monkeypatches the underlying OS/COM/subprocess
call it would otherwise make, so the wiring is verified without ever
shutting down, locking, or muting this machine.
"""

from __future__ import annotations

import subprocess

import pytest

from ultron.tools import system_tools
from ultron.tools.app_index import AppEntry, AppIndex


def _index_with(entries: list[AppEntry]) -> AppIndex:
    idx = AppIndex(__import__("pathlib").Path("unused.yaml"))
    idx.load_entries(entries)
    return idx


class _FakeProc:
    def __init__(self, name: str, pid: int = 4242) -> None:
        self.info = {"name": name}
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


@pytest.fixture(autouse=True)
def _reset_app_index() -> None:
    system_tools._app_index = None


# -- open_app / close_app -------------------------------------------------


async def test_open_app_mock_mode_does_not_call_startfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    entry = AppEntry(display_name="Telegram", kind="lnk", target="C:/x/Telegram.lnk")
    system_tools._app_index = _index_with([entry])
    calls: list[str] = []
    monkeypatch.setattr(system_tools.os, "startfile", lambda path: calls.append(path))

    result = await system_tools.open_app(name="telegramni")

    assert result.ok is True
    assert calls == []


async def test_open_app_real_mode_calls_startfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    entry = AppEntry(display_name="Telegram", kind="lnk", target="C:/x/Telegram.lnk")
    system_tools._app_index = _index_with([entry])
    calls: list[str] = []
    monkeypatch.setattr(system_tools.os, "startfile", lambda path: calls.append(path))

    result = await system_tools.open_app(name="Telegram")

    assert result.ok is True
    assert calls == ["C:/x/Telegram.lnk"]


async def test_open_app_store_kind_uses_explorer_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    entry = AppEntry(display_name="Calculator", kind="store", target="Microsoft.Calc_abc!App")
    system_tools._app_index = _index_with([entry])
    calls: list[list[str]] = []
    monkeypatch.setattr(system_tools.subprocess, "Popen", lambda args: calls.append(args))

    result = await system_tools.open_app(name="Calculator")

    assert result.ok is True
    assert calls == [["explorer.exe", "shell:AppsFolder\\Microsoft.Calc_abc!App"]]


async def test_open_app_not_found_returns_error() -> None:
    system_tools._app_index = _index_with([])

    result = await system_tools.open_app(name="mavjud emas dastur nomi")

    assert result.ok is False


async def test_open_app_ambiguous_returns_candidates() -> None:
    a = AppEntry(display_name="Adobe Reader", kind="lnk", target="a.lnk")
    b = AppEntry(display_name="Adobe Acrobat", kind="lnk", target="b.lnk")
    system_tools._app_index = _index_with([a, b])

    result = await system_tools.open_app(name="Adobe")

    assert result.ok is False
    assert set(result.data["candidates"]) == {"Adobe Reader", "Adobe Acrobat"}


async def test_close_app_refuses_protected_process() -> None:
    system_tools._app_index = _index_with([])
    system_tools._resolve_exe_name_for_close = lambda name, match: "explorer.exe"  # type: ignore[assignment]

    result = await system_tools.close_app(name="explorer")

    assert result.ok is False


async def test_close_app_not_running_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    system_tools._app_index = _index_with([])
    monkeypatch.setattr(system_tools, "_resolve_exe_name_for_close", lambda name, match: None)

    result = await system_tools.close_app(name="notepad")

    assert result.ok is False


async def test_close_app_mock_mode_does_not_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    system_tools._app_index = _index_with([])
    fake = _FakeProc("fakeapp.exe")
    monkeypatch.setattr(
        system_tools, "_resolve_exe_name_for_close", lambda name, match: "fakeapp.exe"
    )
    monkeypatch.setattr(system_tools.psutil, "process_iter", lambda attrs=None: [fake])

    result = await system_tools.close_app(name="fakeapp")

    assert result.ok is True
    assert fake.terminated is False


async def test_close_app_real_mode_terminates_then_confirms_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    system_tools._app_index = _index_with([])
    fake = _FakeProc("fakeapp.exe")
    monkeypatch.setattr(
        system_tools, "_resolve_exe_name_for_close", lambda name, match: "fakeapp.exe"
    )
    monkeypatch.setattr(system_tools.psutil, "process_iter", lambda attrs=None: [fake])
    monkeypatch.setattr(system_tools.psutil, "wait_procs", lambda procs, timeout=None: ([fake], []))

    result = await system_tools.close_app(name="fakeapp")

    assert result.ok is True
    assert fake.terminated is True
    assert fake.killed is False


async def test_close_app_real_mode_kills_survivor_after_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    system_tools._app_index = _index_with([])
    fake = _FakeProc("fakeapp.exe")
    monkeypatch.setattr(
        system_tools, "_resolve_exe_name_for_close", lambda name, match: "fakeapp.exe"
    )
    monkeypatch.setattr(system_tools.psutil, "process_iter", lambda attrs=None: [fake])
    monkeypatch.setattr(system_tools.psutil, "wait_procs", lambda procs, timeout=None: ([], [fake]))

    result = await system_tools.close_app(name="fakeapp")

    assert result.ok is True
    assert fake.terminated is True
    assert fake.killed is True


def test_is_protected_checks_static_list_and_current_process() -> None:
    assert system_tools._is_protected("explorer.exe") is True
    assert system_tools._is_protected("svchost.exe") is True
    assert system_tools._is_protected("some_random_app.exe") is False


async def test_list_running_apps_returns_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_tools, "_visible_window_process_names", lambda: ["chrome.exe", "telegram.exe"]
    )

    result = await system_tools.list_running_apps()

    assert result.ok is True
    assert result.data["apps"] == ["chrome.exe", "telegram.exe"]


async def test_list_running_apps_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_tools, "_visible_window_process_names", lambda: [])

    result = await system_tools.list_running_apps()

    assert result.ok is True
    assert result.data["apps"] == []


# -- volume / brightness --------------------------------------------------


async def test_set_volume_requires_level_or_delta() -> None:
    result = await system_tools.set_volume()

    assert result.ok is False


async def test_set_volume_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.set_volume(level=40)

    assert result.ok is True
    assert result.data == {"volume": 40}


async def test_set_volume_real_mode_calls_do_set_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(system_tools, "_with_com", lambda func: func)
    monkeypatch.setattr(system_tools, "_do_set_volume", lambda level, delta: 77)

    result = await system_tools.set_volume(level=77)

    assert result.ok is True
    assert result.data == {"volume": 77}


async def test_mute_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.mute(muted=True)

    assert result.ok is True
    assert result.data == {"muted": True}


async def test_mute_real_mode_calls_do_mute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(system_tools, "_with_com", lambda func: func)
    calls = []
    monkeypatch.setattr(system_tools, "_do_mute", lambda muted: calls.append(muted))

    result = await system_tools.mute(muted=False)

    assert result.ok is True
    assert calls == [False]


async def test_get_volume_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.get_volume()

    assert result.ok is True
    assert result.data == {"volume": 50, "muted": False}


async def test_get_volume_real_mode_calls_do_get_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(system_tools, "_with_com", lambda func: func)
    monkeypatch.setattr(system_tools, "_do_get_volume", lambda: (65, True))

    result = await system_tools.get_volume()

    assert result.ok is True
    assert result.data == {"volume": 65, "muted": True}


def test_with_com_initializes_and_uninitializes(monkeypatch: pytest.MonkeyPatch) -> None:
    import comtypes

    calls: list[str] = []
    monkeypatch.setattr(comtypes, "CoInitialize", lambda: calls.append("init"))
    monkeypatch.setattr(comtypes, "CoUninitialize", lambda: calls.append("uninit"))

    wrapped = system_tools._with_com(lambda x: x * 2)
    result = wrapped(21)

    assert result == 42
    assert calls == ["init", "uninit"]


async def test_set_brightness_requires_level_or_delta() -> None:
    result = await system_tools.set_brightness()

    assert result.ok is False


async def test_set_brightness_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.set_brightness(delta=10)

    assert result.ok is True


async def test_set_brightness_real_mode_calls_do_set_brightness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(system_tools, "_do_set_brightness", lambda level, delta: 60)

    result = await system_tools.set_brightness(level=60)

    assert result.ok is True
    assert result.data == {"brightness": 60}


# -- power ------------------------------------------------------------------


async def test_lock_screen_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.lock_screen()

    assert result.ok is True


async def test_lock_screen_real_mode_calls_user32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    calls: list[bool] = []
    monkeypatch.setattr(
        system_tools.ctypes.windll.user32, "LockWorkStation", lambda: calls.append(True)
    )

    result = await system_tools.lock_screen()

    assert result.ok is True
    assert calls == [True]


async def test_sleep_pc_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.sleep_pc()

    assert result.ok is True


async def test_sleep_pc_real_mode_calls_rundll32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    calls = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(system_tools.subprocess, "run", fake_run)

    result = await system_tools.sleep_pc()

    assert result.ok is True
    assert calls == [["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]]


async def test_shutdown_pc_mock_mode_does_not_call_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    calls = []
    monkeypatch.setattr(system_tools.subprocess, "run", lambda *a, **k: calls.append(a))

    result = await system_tools.shutdown_pc(delay_s=30)

    assert result.ok is True
    assert result.data == {"delay_s": 30}
    assert calls == []


async def test_shutdown_pc_real_mode_calls_shutdown_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    calls = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(system_tools.subprocess, "run", fake_run)

    result = await system_tools.shutdown_pc(delay_s=45)

    assert result.ok is True
    assert calls == [["shutdown", "/s", "/t", "45"]]


async def test_shutdown_pc_failure_returns_ok_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(system_tools.subprocess, "run", fake_run)

    result = await system_tools.shutdown_pc()

    assert result.ok is False


async def test_restart_pc_real_mode_calls_shutdown_r(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    calls = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(system_tools.subprocess, "run", fake_run)

    result = await system_tools.restart_pc(delay_s=20)

    assert result.ok is True
    assert calls == [["shutdown", "/r", "/t", "20"]]


async def test_cancel_shutdown_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    calls = []
    monkeypatch.setattr(system_tools.subprocess, "run", lambda *a, **k: calls.append(a))

    result = await system_tools.cancel_shutdown()

    assert result.ok is True
    assert calls == []


async def test_cancel_shutdown_real_mode_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        lambda args, **k: subprocess.CompletedProcess(args, 0),
    )

    result = await system_tools.cancel_shutdown()

    assert result.ok is True


async def test_cancel_shutdown_real_mode_nothing_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        lambda args, **k: subprocess.CompletedProcess(args, 1),
    )

    result = await system_tools.cancel_shutdown()

    assert result.ok is False


# -- screenshot / status ----------------------------------------------------


async def test_screenshot_mock_mode_does_not_call_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await system_tools.screenshot()

    assert result.ok is True
    assert "path" in result.data


async def test_screenshot_real_mode_calls_do_screenshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    fake_path = tmp_path / "shot.png"  # type: ignore[operator]
    monkeypatch.setattr(system_tools, "_do_screenshot", lambda out_dir: fake_path)

    result = await system_tools.screenshot()

    assert result.ok is True
    assert result.data["path"] == str(fake_path)


async def test_get_system_status_returns_collected_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_tools,
        "_collect_system_status",
        lambda: {
            "battery_percent": 80,
            "battery_plugged": True,
            "cpu_percent": 12.0,
            "ram_percent": 55.0,
            "disk_free_gb": 120.5,
        },
    )

    result = await system_tools.get_system_status()

    assert result.ok is True
    assert result.data["battery_percent"] == 80
    assert "80" in result.speak


async def test_get_time_returns_hour_and_minute() -> None:
    result = await system_tools.get_time()

    assert result.ok is True
    assert "hour" in result.data
    assert "minute" in result.data


async def test_get_date_returns_weekday() -> None:
    result = await system_tools.get_date()

    assert result.ok is True
    assert result.data["weekday"] in {
        "dushanba",
        "seshanba",
        "chorshanba",
        "payshanba",
        "juma",
        "shanba",
        "yakshanba",
    }
