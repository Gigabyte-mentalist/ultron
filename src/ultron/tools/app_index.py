"""Installed-app index for ``open_app``/``close_app`` fuzzy lookup
(docs/tz/07_tools.md §7A.1).

Scans Start Menu ``.lnk`` shortcuts (both the per-machine and per-user
folders) plus Microsoft Store apps (via ``Get-StartApps``), loads the Uzbek
alias table from ``config/app_aliases.yaml``, and resolves a spoken name to
one confident :class:`AppEntry`, a list of ambiguous candidates, or ``None``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from rapidfuzz import fuzz, process

from ultron.text.normalize import strip_uz_suffixes

_LNK_ROOTS: tuple[Path, ...] = (
    Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"),
    Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs",
)

FUZZY_THRESHOLD = 75
AMBIGUOUS_GAP = 5
_SUBPROCESS_TIMEOUT_S = 15

# ``ms-settings:``-style URIs the TZ calls out explicitly, keyed by the
# canonical name they're looked up under (an alias in app_aliases.yaml maps
# the Uzbek word to this key).
_BUILTIN_URI_APPS: dict[str, str] = {
    "sozlamalar": "ms-settings:",
}


@dataclass(frozen=True)
class AppEntry:
    display_name: str
    kind: str  # "lnk" | "store" | "uri"
    target: str  # .lnk path | AppID | URI
    exe_name: str | None = None  # lowercase exe basename, used by close_app


def _resolve_lnk_target(lnk_path: Path, shell: object) -> str | None:
    try:
        shortcut = shell.CreateShortcut(str(lnk_path))  # type: ignore[attr-defined]
        target = shortcut.Targetpath
        return target or None
    except Exception as exc:  # noqa: BLE001 -- a single unreadable shortcut must not abort the scan
        logger.debug("Shortcut o'qib bo'lmadi ({}): {}", lnk_path, exc)
        return None


def scan_lnk_apps(roots: tuple[Path, ...] = _LNK_ROOTS) -> list[AppEntry]:
    """Walk the Start Menu ``.lnk`` folders and resolve each shortcut's target
    executable (needed later so ``close_app`` can match it via ``psutil``).

    Runs inside ``asyncio.to_thread`` (see ``system_tools._ensure_app_index_scanned``),
    a thread-pool thread with no COM apartment of its own -- ``pythoncom.CoInitialize()``
    brackets the ``win32com.client.Dispatch`` call the same way
    ``system_tools._with_com`` does for pycaw.
    """
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    try:
        entries: list[AppEntry] = []
        seen: set[str] = set()
        shell = win32com.client.Dispatch("WScript.Shell")
        for root in roots:
            if not root.is_dir():
                continue
            for lnk in root.rglob("*.lnk"):
                key = lnk.stem.lower()
                if key in seen:
                    continue
                target = _resolve_lnk_target(lnk, shell)
                exe_name = Path(target).name.lower() if target else None
                entries.append(
                    AppEntry(display_name=lnk.stem, kind="lnk", target=str(lnk), exe_name=exe_name)
                )
                seen.add(key)
        return entries
    finally:
        pythoncom.CoUninitialize()


def scan_store_apps() -> list[AppEntry]:
    """``Get-StartApps`` lists Microsoft Store (UWP) apps by display name +
    ``AppID``; opened via ``explorer.exe shell:AppsFolder\\<AppID>``."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json -Compress"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Get-StartApps ishga tushmadi: {}", exc)
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Get-StartApps chiqishi JSON emas: {}", exc)
        return []
    if isinstance(raw, dict):
        raw = [raw]

    entries: list[AppEntry] = []
    for item in raw:
        name = item.get("Name")
        app_id = item.get("AppID")
        if not name or not app_id:
            continue
        entries.append(AppEntry(display_name=name, kind="store", target=app_id))
    return entries


class AppIndex:
    """Fuzzy-searchable, in-memory index of installed apps."""

    def __init__(self, aliases_path: Path) -> None:
        self._aliases_path = aliases_path
        self._entries: list[AppEntry] = []
        self._aliases: dict[str, str] = {}
        self._scanned = False

    def scan(self) -> None:
        """Rebuild the index. Safe to call repeatedly (e.g. every 6h)."""
        entries = [
            AppEntry(display_name=name, kind="uri", target=uri)
            for name, uri in _BUILTIN_URI_APPS.items()
        ]
        entries.extend(scan_lnk_apps())
        entries.extend(scan_store_apps())
        self._entries = entries
        self._aliases = self._load_aliases()
        self._scanned = True
        logger.info("App indeks yangilandi: {} yozuv, {} alias", len(entries), len(self._aliases))

    def load_entries(self, entries: list[AppEntry], aliases: dict[str, str] | None = None) -> None:
        """Inject entries directly (tests; also usable for a cached snapshot).
        Marks the index as scanned, so ``system_tools`` won't trigger a real
        scan on top of injected test data."""
        self._entries = list(entries)
        self._aliases = aliases if aliases is not None else self._load_aliases()
        self._scanned = True

    @property
    def scanned(self) -> bool:
        return self._scanned

    def _load_aliases(self) -> dict[str, str]:
        if not self._aliases_path.is_file():
            return {}
        try:
            raw = yaml.safe_load(self._aliases_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            logger.warning("app_aliases.yaml o'qib bo'lmadi ({}): {}", self._aliases_path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}

    @property
    def entries(self) -> list[AppEntry]:
        return list(self._entries)

    def find(self, query: str) -> AppEntry | list[AppEntry] | None:
        """Resolve a spoken app name: exact alias/name match first, else a
        fuzzy ``rapidfuzz`` search (``>= FUZZY_THRESHOLD``). Returns a single
        entry, a list of ambiguous candidates (scores within
        ``AMBIGUOUS_GAP`` of the best), or ``None`` if nothing matches."""
        normalized = strip_uz_suffixes(query)
        if not normalized:
            return None

        search_term = self._aliases.get(normalized, normalized)

        for entry in self._entries:
            if entry.display_name.strip().lower() == search_term:
                return entry
        for entry in self._entries:
            if entry.exe_name and Path(entry.exe_name).stem == search_term:
                return entry

        if not self._entries:
            return None

        choices = {i: entry.display_name.lower() for i, entry in enumerate(self._entries)}
        matches = process.extract(
            search_term,
            choices,
            scorer=fuzz.WRatio,
            limit=5,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if not matches:
            return None

        best_score = matches[0][1]
        top = [m for m in matches if best_score - m[1] < AMBIGUOUS_GAP]
        if len(top) == 1:
            return self._entries[top[0][2]]
        return [self._entries[m[2]] for m in top]

    def find_by_exe(self, exe_name: str) -> AppEntry | None:
        target = exe_name.strip().lower()
        for entry in self._entries:
            if entry.exe_name == target:
                return entry
        return None
