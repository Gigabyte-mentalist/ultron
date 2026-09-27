"""Tests for ultron.tools.app_index.AppIndex (docs/tz/07_tools.md §7A.1, §7A.3).

Uses ``load_entries`` to inject a fixed, mock ``.lnk``/Store list instead of
scanning the real Start Menu, per the TZ's own acceptance note ("mock .lnk
ro'yxati bilan").
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ultron.tools.app_index import AppEntry, AppIndex, scan_lnk_apps


def _index(entries: list[AppEntry], aliases: dict[str, str] | None = None) -> AppIndex:
    idx = AppIndex(Path("unused/app_aliases.yaml"))
    idx.load_entries(entries, aliases or {})
    return idx


_TELEGRAM = AppEntry(
    display_name="Telegram", kind="lnk", target="C:/lnk/Telegram.lnk", exe_name="telegram.exe"
)
_CHROME = AppEntry(
    display_name="Google Chrome", kind="lnk", target="C:/lnk/Chrome.lnk", exe_name="chrome.exe"
)
_NOTEPAD = AppEntry(
    display_name="Notepad", kind="lnk", target="C:/lnk/Notepad.lnk", exe_name="notepad.exe"
)
_SETTINGS = AppEntry(display_name="Sozlamalar", kind="uri", target="ms-settings:")


def test_exact_display_name_match_is_case_insensitive() -> None:
    idx = _index([_TELEGRAM, _CHROME])

    assert idx.find("Telegram") == _TELEGRAM
    assert idx.find("telegram") == _TELEGRAM


def test_uz_suffix_is_stripped_before_matching() -> None:
    idx = _index([_TELEGRAM])

    assert idx.find("telegramni") == _TELEGRAM
    assert idx.find("telegramga") == _TELEGRAM


def test_alias_resolves_to_canonical_app_name() -> None:
    idx = _index([_CHROME], aliases={"brauzer": "google chrome", "xrom": "google chrome"})

    assert idx.find("brauzerni") == _CHROME
    assert idx.find("xrom") == _CHROME


def test_fuzzy_match_within_threshold() -> None:
    idx = _index([_TELEGRAM, _CHROME, _NOTEPAD])

    match = idx.find("telegran")  # one-letter typo

    assert match == _TELEGRAM


def test_no_match_returns_none() -> None:
    idx = _index([_TELEGRAM, _CHROME])

    assert idx.find("bunday dastur yoq umuman") is None


def test_empty_index_returns_none() -> None:
    idx = _index([])

    assert idx.find("telegram") is None


def test_ambiguous_close_scores_return_candidate_list() -> None:
    # Both share the same prefix/length -- close enough that WRatio scores
    # land within AMBIGUOUS_GAP of each other for a partial query.
    a = AppEntry(display_name="Adobe Reader", kind="lnk", target="a.lnk")
    b = AppEntry(display_name="Adobe Acrobat", kind="lnk", target="b.lnk")
    idx = _index([a, b])

    match = idx.find("Adobe")

    assert isinstance(match, list)
    assert set(match) == {a, b}


def test_find_by_exe_matches_lowercase_basename() -> None:
    idx = _index([_TELEGRAM, _CHROME])

    assert idx.find_by_exe("chrome.exe") == _CHROME
    assert idx.find_by_exe("CHROME.EXE") == _CHROME
    assert idx.find_by_exe("nope.exe") is None


def test_uri_app_matches_exact_alias_target() -> None:
    idx = _index([_SETTINGS], aliases={"sozlamalar": "sozlamalar"})

    assert idx.find("sozlamalarni") == _SETTINGS


async def test_scan_lnk_apps_from_a_worker_thread_does_not_raise_com_error() -> None:
    """Regression: ``win32com.client.Dispatch`` needs ``pythoncom.CoInitialize()``
    on the calling thread, which a bare ``asyncio.to_thread`` pool thread
    never has by default (caught via manual ``--text`` smoke testing, not by
    the mocked-entries tests above -- see docs/PROGRESS.md's 07 entry).
    Runs against the real Start Menu roots; asserts no exception, regardless
    of how many/few real shortcuts this machine happens to have."""
    entries = await asyncio.to_thread(scan_lnk_apps)

    assert isinstance(entries, list)
