"""File system tools: search, open, browse, and manage files/folders, all
bounded to ``config.tools.files.allowed_roots`` (docs/tz/07_tools.md §7B).

Safety boundary (§7B.1): every path is resolved with
:func:`resolve_within_allowed_roots`, which rejects anything outside the
configured roots or inside a blocked system folder (Windows/Program
Files/AppData), before any tool below touches disk. Deletion never bypasses
the Recycle Bin (:func:`send2trash.send2trash`).

The background name index (SQLite ``file_index`` table, kept in sync by
``watchdog``) is *not* built automatically by importing this module or by
calling any tool here -- ``allowed_roots`` includes an entire drive
(``D:/``) in this project's config, so a first-use auto-scan would be an
unbounded, surprising disk walk. The orchestrator (docs/tz/08_orchestrator.md)
is expected to call :meth:`FileIndexer.build_index` at startup (and every 6h)
and :meth:`FileIndexer.start_watching` once; until that happens,
``search_files`` simply finds nothing.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Annotated, Any

import send2trash
from loguru import logger
from rapidfuzz import fuzz, process
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ultron.config import RootConfig
from ultron.memory import db as memory_db
from ultron.memory.db import Database
from ultron.text.normalize import strip_uz_suffixes
from ultron.tools.base import ToolResult, is_mock_mode, tool

_DEFAULT_ALLOWED_ROOTS: tuple[str, ...] = (
    "~/Desktop",
    "~/Documents",
    "~/Downloads",
    "~/Pictures",
    "~/Music",
    "~/Videos",
)
_DEFAULT_MAX_RESULTS = 10
_FUZZY_SCORE_CUTOFF = 60

_FOLDER_ALIASES: dict[str, str] = {
    "ish stoli": "Desktop",
    "desktop": "Desktop",
    "hujjatlar": "Documents",
    "documents": "Documents",
    "yuklamalar": "Downloads",
    "zagruzka": "Downloads",
    "downloads": "Downloads",
    "rasmlar": "Pictures",
    "pictures": "Pictures",
    "musiqalar": "Music",
    "music": "Music",
    "videolar": "Videos",
    "videos": "Videos",
}

# Ordinal references to the last search_files/recent_files result list
# ("ikkinchisini och" -> index 1), docs/tz/07_tools.md §7B.2's "Kontekst" note.
_ORDINAL_WORDS: dict[str, int] = {
    "birinchi": 1,
    "ikkinchi": 2,
    "uchinchi": 3,
    "tortinchi": 4,
    "to'rtinchi": 4,
    "beshinchi": 5,
    "oxirgi": -1,
}

_config: RootConfig | None = None
_last_results: list[dict[str, Any]] = []

_UPSERT_FILE_INDEX_SQL = (
    "INSERT OR REPLACE INTO file_index (path, name, ext, size, mtime) VALUES (?, ?, ?, ?, ?)"
)


class FileAccessError(Exception):
    """Raised when a path is outside ``allowed_roots`` or in a blocked folder."""


def configure(config: RootConfig) -> None:
    global _config
    _config = config


def _allowed_roots() -> list[Path]:
    if _config is None:
        return [Path(p).expanduser() for p in _DEFAULT_ALLOWED_ROOTS]
    return [Path(p) for p in _config.tools.files.allowed_roots]


def _max_results() -> int:
    return _config.tools.files.max_results if _config is not None else _DEFAULT_MAX_RESULTS


def _blocked_roots() -> list[Path]:
    names = (
        "WINDIR",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
        "APPDATA",
        "LOCALAPPDATA",
    )
    roots = []
    for name in names:
        value = os.environ.get(name)
        if value:
            roots.append(Path(value).resolve())
    return roots


def resolve_within_allowed_roots(raw_path: str, allowed_roots: list[Path]) -> Path:
    """Resolve ``raw_path`` and enforce the §7B.1 safety boundary.

    Raises :class:`FileAccessError` (never returns a path outside the
    boundary) if ``raw_path`` resolves outside every root in
    ``allowed_roots``, or inside a blocked system folder.
    """
    candidate = Path(raw_path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileAccessError(f"Yo'lni o'qib bo'lmadi: {exc}") from exc

    for blocked in _blocked_roots():
        if resolved == blocked or resolved.is_relative_to(blocked):
            raise FileAccessError("ruxsat etilmagan joy")

    for root in allowed_roots:
        root_resolved = Path(root).expanduser().resolve()
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return resolved

    raise FileAccessError("ruxsat etilmagan joy")


def _resolve_named_folder(name: str) -> Path | None:
    words = name.strip().lower().split()
    if words and words[-1].startswith("papka"):
        words = words[:-1]  # "yuklamalar papkasini" -> "yuklamalar"
    key = strip_uz_suffixes(" ".join(words)) if words else ""
    folder = _FOLDER_ALIASES.get(key)
    return (Path.home() / folder) if folder else None


def is_folder_reference(name: str) -> bool:
    """True if ``name`` looks like one of the known Uzbek folder aliases,
    with or without an explicit "papka" word. ``llm/fastpath.py`` uses this
    to route "Yuklamalar papkasini och" to ``open_folder`` instead of the
    generic ``open_app`` pattern it would otherwise match."""
    if "papka" in name.strip().lower():
        return True
    return _resolve_named_folder(name) is not None


def _short_path(path_str: str) -> str:
    """LLM-facing form of a path (§7B.2: "LLM'ga fayl yo'llari qisqartirilgan
    holda beriladi"), e.g. ``~\\Downloads\\rezyume.pdf``."""
    path = Path(path_str)
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _dedupe_name(target: Path) -> Path:
    """§7B.2: "Nom to'qnashsa (1) qo'shiladi.\""""
    if not target.exists():
        return target
    n = 1
    while True:
        candidate = target.with_name(f"{target.stem} ({n}){target.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _resolve_ordinal_index(text: str) -> int | None:
    normalized = text.strip().lower()
    if normalized.isdigit():
        return int(normalized)
    # Uzbek stacks possessive+case onto the bare ordinal ("ikkinchi" + "si"
    # + "ni" -> "ikkinchisini"); strip_uz_suffixes only removes one case
    # suffix, so the combined "sini" ending is checked first.
    for suffix in ("sini", "ni", "si"):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 3:
            normalized = normalized[: -len(suffix)]
            break
    return _ORDINAL_WORDS.get(normalized)


# -- file index (SQLite, kept in sync by watchdog) -------------------------


class FileIndexer:
    """Background name index over ``allowed_roots``, stored in the shared
    SQLite ``file_index`` table (docs/tz/07_tools.md §7B.2)."""

    def __init__(self, db: Database, allowed_roots: list[Path]) -> None:
        self._db = db
        self._allowed_roots = allowed_roots
        self._observer: Observer | None = None

    async def build_index(self) -> int:
        """Full rescan of ``allowed_roots``. Blocking I/O runs in a thread."""
        rows = await asyncio.to_thread(self._scan_all)
        conn = self._db.conn
        await conn.execute("DELETE FROM file_index")
        await conn.executemany(_UPSERT_FILE_INDEX_SQL, rows)
        await conn.commit()
        return len(rows)

    def _scan_all(self) -> list[tuple[str, str, str, int, float]]:
        """Walks every allowed root with ``os.walk`` rather than
        ``Path.rglob`` -- a single unreadable/vanished subdirectory (a
        OneDrive placeholder, a broken junction, ...) raising mid-walk kills
        ``rglob``'s one recursive generator for the *entire* root once the
        exception unwinds it, silently truncating the whole scan. ``os.walk``
        takes an ``onerror`` callback specifically so one bad directory is
        skipped without abandoning its siblings."""
        rows: list[tuple[str, str, str, int, float]] = []
        for root in self._allowed_roots:
            if not root.is_dir():
                continue
            for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _exc: None):
                for filename in filenames:
                    entry = Path(dirpath) / filename
                    try:
                        stat = entry.stat()
                    except OSError:
                        continue
                    rows.append(
                        (
                            str(entry),
                            entry.name,
                            entry.suffix.lower().lstrip("."),
                            stat.st_size,
                            stat.st_mtime,
                        )
                    )
        return rows

    async def upsert_path(self, path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        await self._db.conn.execute(
            _UPSERT_FILE_INDEX_SQL,
            (str(path), path.name, path.suffix.lower().lstrip("."), stat.st_size, stat.st_mtime),
        )
        await self._db.conn.commit()

    async def remove_path(self, path: Path) -> None:
        await self._db.conn.execute("DELETE FROM file_index WHERE path = ?", (str(path),))
        await self._db.conn.commit()

    def start_watching(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._observer is not None:
            return
        loop = loop or asyncio.get_running_loop()
        handler = _IndexEventHandler(self, loop)
        observer = Observer()
        for root in self._allowed_roots:
            if root.is_dir():
                observer.schedule(handler, str(root), recursive=True)
        observer.start()
        self._observer = observer

    def stop_watching(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None


class _IndexEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: FileIndexer, loop: asyncio.AbstractEventLoop) -> None:
        self._indexer = indexer
        self._loop = loop

    def on_created(self, event: Any) -> None:
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self._indexer.upsert_path(Path(event.src_path)), self._loop
            )

    on_modified = on_created

    def on_deleted(self, event: Any) -> None:
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self._indexer.remove_path(Path(event.src_path)), self._loop
            )

    def on_moved(self, event: Any) -> None:
        if event.is_directory:
            return
        asyncio.run_coroutine_threadsafe(
            self._indexer.remove_path(Path(event.src_path)), self._loop
        )
        asyncio.run_coroutine_threadsafe(
            self._indexer.upsert_path(Path(event.dest_path)), self._loop
        )


_indexer: FileIndexer | None = None


async def _get_indexer() -> FileIndexer:
    global _indexer
    if _indexer is None:
        db = await memory_db.get_db()
        _indexer = FileIndexer(db, _allowed_roots())
    return _indexer


async def search_files_index(
    db: Database,
    query: str,
    *,
    extension: str | None,
    folder: str | None,
    modified_within_days: int | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Query the ``file_index`` table, then fuzzy-rank by name in Python
    (rapidfuzz over an already SQL-filtered candidate set stays well under
    the §7B.3 300ms budget even at 50k rows)."""
    sql = "SELECT path, name, ext, size, mtime FROM file_index WHERE 1=1"
    params: list[Any] = []
    if extension:
        sql += " AND ext = ?"
        params.append(extension.lower().lstrip("."))
    if folder:
        sql += " AND path LIKE ?"
        params.append(f"%{folder}%")
    if modified_within_days is not None:
        params.append(time.time() - modified_within_days * 86400)
        sql += " AND mtime >= ?"

    cursor = await db.conn.execute(sql, params)
    rows = await cursor.fetchall()
    if not rows:
        return []

    choices = {i: row[1] for i, row in enumerate(rows)}
    matches = process.extract(
        query, choices, scorer=fuzz.WRatio, limit=max_results, score_cutoff=_FUZZY_SCORE_CUTOFF
    )
    return [
        {
            "path": rows[idx][0],
            "name": rows[idx][1],
            "size": rows[idx][3],
            "mtime": rows[idx][4],
        }
        for _choice, _score, idx in matches
    ]


# -- tools ------------------------------------------------------------------


@tool(
    name="search_files",
    description=(
        "Fuzzy-search indexed files by name, optionally filtered by extension,"
        " folder, or how recently modified."
    ),
)
async def search_files(
    query: Annotated[str, "File name or partial name to search for"],
    extension: Annotated[str, "File extension filter, e.g. 'pdf' (no dot)"] | None = None,
    folder: Annotated[str, "Restrict results to paths containing this folder name"] | None = None,
    modified_within_days: Annotated[int, "Only files modified within this many days"] | None = None,
) -> ToolResult:
    db = await memory_db.get_db()
    results = await search_files_index(
        db,
        query,
        extension=extension,
        folder=folder,
        modified_within_days=modified_within_days,
        max_results=_max_results(),
    )
    if not results:
        return ToolResult(ok=False, error=f"'{query}' bo'yicha fayl topilmadi")

    _last_results[:] = results
    display = [{"name": r["name"], "path": _short_path(r["path"])} for r in results]
    return ToolResult(
        ok=True,
        data={"results": display},
        speak=f"{len(results)} ta fayl topildi, birinchisi: {results[0]['name']}.",
    )


async def _open_resolved_path(path: Path, display_name: str) -> ToolResult:
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] open_file -> {}", path)
        return ToolResult(
            ok=True, data={"opened": _short_path(str(path))}, speak=f"{display_name} ochildi."
        )
    try:
        await asyncio.to_thread(os.startfile, str(path))  # type: ignore[attr-defined]
    except OSError as exc:
        return ToolResult(ok=False, error=f"Faylni ochib bo'lmadi: {exc}")
    return ToolResult(
        ok=True, data={"opened": _short_path(str(path))}, speak=f"{display_name} ochildi."
    )


@tool(
    name="open_file",
    description=(
        "Open a file by its full path, by a fuzzy name search, or by its"
        " ordinal position ('ikkinchisi'/'2') in the last search_files/recent_files result."
    ),
)
async def open_file(
    path_or_query: Annotated[str, "Full path, search text, or ordinal reference"],
) -> ToolResult:
    ordinal = _resolve_ordinal_index(path_or_query)
    if ordinal is not None and _last_results:
        idx = ordinal - 1 if ordinal > 0 else ordinal
        if -len(_last_results) <= idx < len(_last_results):
            chosen = _last_results[idx]
            return await _open_resolved_path(Path(chosen["path"]), chosen["name"])
        return ToolResult(ok=False, error="Bunday tartib raqamli natija yo'q")

    direct = Path(path_or_query).expanduser()
    if direct.is_file():
        try:
            resolved = resolve_within_allowed_roots(str(direct), _allowed_roots())
        except FileAccessError as exc:
            return ToolResult(ok=False, error=str(exc))
        return await _open_resolved_path(resolved, resolved.name)

    db = await memory_db.get_db()
    results = await search_files_index(
        db,
        path_or_query,
        extension=None,
        folder=None,
        modified_within_days=None,
        max_results=_max_results(),
    )
    if not results:
        return ToolResult(ok=False, error=f"'{path_or_query}' nomli fayl topilmadi")

    _last_results[:] = results
    if len(results) == 1:
        chosen = results[0]
        return await _open_resolved_path(Path(chosen["path"]), chosen["name"])

    candidates = [{"name": r["name"], "path": _short_path(r["path"])} for r in results]
    return ToolResult(
        ok=False,
        data={"candidates": candidates},
        speak="Bir nechta fayl topildi, qaysi birini demoqchisiz?",
    )


@tool(
    name="open_folder",
    description="Open a folder in File Explorer, by Uzbek name ('ish stoli') or path.",
)
async def open_folder(
    name_or_path: Annotated[str, "Folder name (may be Uzbek) or full path"],
) -> ToolResult:
    candidate = _resolve_named_folder(name_or_path) or Path(name_or_path).expanduser()
    try:
        resolved = resolve_within_allowed_roots(str(candidate), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.is_dir():
        return ToolResult(ok=False, error=f"'{name_or_path}' papkasi topilmadi")

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] open_folder -> {}", resolved)
        return ToolResult(
            ok=True, data={"opened": _short_path(str(resolved))}, speak="Papka ochildi."
        )
    try:
        await asyncio.to_thread(os.startfile, str(resolved))  # type: ignore[attr-defined]
    except OSError as exc:
        return ToolResult(ok=False, error=f"Papkani ochib bo'lmadi: {exc}")
    return ToolResult(ok=True, data={"opened": _short_path(str(resolved))}, speak="Papka ochildi.")


def _list_dir_entries(folder: Path, limit: int) -> list[dict[str, str]]:
    children = sorted(folder.iterdir(), key=lambda p: p.name.lower())[:limit]
    return [{"name": c.name, "type": "folder" if c.is_dir() else "file"} for c in children]


@tool(name="list_folder", description="List the contents (names and types) of a folder.")
async def list_folder(
    path: Annotated[str, "Folder name (may be Uzbek) or full path"],
    limit: Annotated[int, "Max number of entries to return"] = 20,
) -> ToolResult:
    candidate = _resolve_named_folder(path) or Path(path).expanduser()
    try:
        resolved = resolve_within_allowed_roots(str(candidate), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.is_dir():
        return ToolResult(ok=False, error=f"'{path}' papkasi topilmadi")

    entries = await asyncio.to_thread(_list_dir_entries, resolved, limit)
    if not entries:
        return ToolResult(ok=True, data={"entries": []}, speak="Papka bo'sh.")
    names = ", ".join(e["name"] for e in entries)
    return ToolResult(
        ok=True, data={"entries": entries}, speak=f"{len(entries)} ta element: {names}"
    )


def _recent_files_in(folder: Path, limit: int) -> list[dict[str, Any]]:
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in files[:limit]:
        stat = p.stat()
        result.append(
            {"path": str(p), "name": p.name, "size": stat.st_size, "mtime": stat.st_mtime}
        )
    return result


@tool(name="recent_files", description="List the most recently modified files in a folder.")
async def recent_files(
    folder: Annotated[
        str, "Folder name, e.g. 'Downloads' (default) or Uzbek equivalent"
    ] = "Downloads",
    limit: Annotated[int, "Max number of files to return"] = 5,
) -> ToolResult:
    candidate = _resolve_named_folder(folder) or Path(folder).expanduser()
    try:
        resolved = resolve_within_allowed_roots(str(candidate), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.is_dir():
        return ToolResult(ok=False, error=f"'{folder}' papkasi topilmadi")

    files = await asyncio.to_thread(_recent_files_in, resolved, limit)
    if not files:
        return ToolResult(ok=True, data={"files": []}, speak="Bu papkada fayl topilmadi.")

    _last_results[:] = files
    display = [{"name": f["name"], "path": _short_path(f["path"])} for f in files]
    names = ", ".join(f["name"] for f in files)
    return ToolResult(ok=True, data={"files": display}, speak=f"Oxirgi fayllar: {names}")


@tool(name="create_folder", description="Create a new folder inside an existing parent folder.")
async def create_folder(
    parent: Annotated[str, "Parent folder name (may be Uzbek) or path"],
    name: Annotated[str, "New folder's name"],
) -> ToolResult:
    candidate = _resolve_named_folder(parent) or Path(parent).expanduser()
    try:
        resolved_parent = resolve_within_allowed_roots(str(candidate), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    try:
        target = resolve_within_allowed_roots(str(resolved_parent / name), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] create_folder -> {}", target)
        return ToolResult(
            ok=True, data={"path": _short_path(str(target))}, speak=f"'{name}' papkasi yaratildi."
        )

    try:
        await asyncio.to_thread(lambda: target.mkdir(parents=False, exist_ok=False))
    except FileExistsError:
        return ToolResult(ok=False, error="Bu nomli papka allaqachon mavjud")
    except OSError as exc:
        return ToolResult(ok=False, error=f"Papka yaratib bo'lmadi: {exc}")
    return ToolResult(
        ok=True, data={"path": _short_path(str(target))}, speak=f"'{name}' papkasi yaratildi."
    )


@tool(
    name="rename_item",
    description="Rename a file or folder in place.",
    dangerous=True,
    confirm_question_uz="Nomini o'zgartirishni tasdiqlaysizmi?",
)
async def rename_item(
    path: Annotated[str, "Full path of the file/folder to rename"],
    new_name: Annotated[str, "New name (no path)"],
) -> ToolResult:
    try:
        resolved = resolve_within_allowed_roots(path, _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.exists():
        return ToolResult(ok=False, error="Manba topilmadi")
    try:
        target = resolve_within_allowed_roots(str(resolved.parent / new_name), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] rename_item {} -> {}", resolved, target)
        return ToolResult(
            ok=True, data={"path": _short_path(str(target))}, speak=f"'{new_name}' deb nomlandi."
        )

    target = _dedupe_name(target)
    try:
        await asyncio.to_thread(resolved.rename, target)
    except OSError as exc:
        return ToolResult(ok=False, error=f"Nomini o'zgartirib bo'lmadi: {exc}")
    return ToolResult(
        ok=True, data={"path": _short_path(str(target))}, speak=f"'{target.name}' deb nomlandi."
    )


async def _resolve_dst_folder(dst_folder: str) -> Path | ToolResult:
    candidate = _resolve_named_folder(dst_folder) or Path(dst_folder).expanduser()
    try:
        resolved = resolve_within_allowed_roots(str(candidate), _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.is_dir():
        return ToolResult(ok=False, error=f"'{dst_folder}' papkasi topilmadi")
    return resolved


@tool(
    name="move_item",
    description="Move a file or folder into a destination folder.",
    dangerous=True,
    confirm_question_uz="Ko'chirishni tasdiqlaysizmi?",
)
async def move_item(
    src: Annotated[str, "Full path of the file/folder to move"],
    dst_folder: Annotated[str, "Destination folder name (may be Uzbek) or path"],
) -> ToolResult:
    try:
        resolved_src = resolve_within_allowed_roots(src, _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved_src.exists():
        return ToolResult(ok=False, error="Manba topilmadi")

    dst = await _resolve_dst_folder(dst_folder)
    if isinstance(dst, ToolResult):
        return dst

    target = _dedupe_name(dst / resolved_src.name)
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] move_item {} -> {}", resolved_src, target)
        return ToolResult(
            ok=True,
            data={"path": _short_path(str(target))},
            speak=f"'{resolved_src.name}' ko'chirildi.",
        )

    try:
        await asyncio.to_thread(shutil.move, str(resolved_src), str(target))
    except OSError as exc:
        return ToolResult(ok=False, error=f"Ko'chirib bo'lmadi: {exc}")
    return ToolResult(
        ok=True,
        data={"path": _short_path(str(target))},
        speak=f"'{resolved_src.name}' ko'chirildi.",
    )


def _copy_path(src: Path, target: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, target)
    else:
        shutil.copy2(src, target)


@tool(name="copy_item", description="Copy a file or folder into a destination folder.")
async def copy_item(
    src: Annotated[str, "Full path of the file/folder to copy"],
    dst_folder: Annotated[str, "Destination folder name (may be Uzbek) or path"],
) -> ToolResult:
    try:
        resolved_src = resolve_within_allowed_roots(src, _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved_src.exists():
        return ToolResult(ok=False, error="Manba topilmadi")

    dst = await _resolve_dst_folder(dst_folder)
    if isinstance(dst, ToolResult):
        return dst

    target = _dedupe_name(dst / resolved_src.name)
    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] copy_item {} -> {}", resolved_src, target)
        return ToolResult(
            ok=True,
            data={"path": _short_path(str(target))},
            speak=f"'{resolved_src.name}' nusxalandi.",
        )

    try:
        await asyncio.to_thread(_copy_path, resolved_src, target)
    except OSError as exc:
        return ToolResult(ok=False, error=f"Nusxalab bo'lmadi: {exc}")
    return ToolResult(
        ok=True, data={"path": _short_path(str(target))}, speak=f"'{resolved_src.name}' nusxalandi."
    )


@tool(
    name="delete_item",
    description="Delete a file or folder by moving it to the Recycle Bin.",
    dangerous=True,
    confirm_question_uz="O'chirishni tasdiqlaysizmi?",
)
async def delete_item(path: Annotated[str, "Full path of the file/folder to delete"]) -> ToolResult:
    try:
        resolved = resolve_within_allowed_roots(path, _allowed_roots())
    except FileAccessError as exc:
        return ToolResult(ok=False, error=str(exc))
    if not resolved.exists():
        return ToolResult(ok=False, error="Manba topilmadi")

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] delete_item -> {}", resolved)
        return ToolResult(
            ok=True,
            data={"deleted": _short_path(str(resolved))},
            speak=f"'{resolved.name}' savatchaga tashlandi.",
        )

    try:
        await asyncio.to_thread(send2trash.send2trash, str(resolved))
    except OSError as exc:
        return ToolResult(ok=False, error=f"O'chirib bo'lmadi: {exc}")
    return ToolResult(
        ok=True,
        data={"deleted": _short_path(str(resolved))},
        speak=f"'{resolved.name}' savatchaga tashlandi.",
    )
