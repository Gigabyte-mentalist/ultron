"""Tests for ultron.tools.file_tools (docs/tz/07_tools.md §7B).

Every mutating tool is exercised inside a temporary ``allowed_roots`` (never
the real Desktop/Documents/etc.), per the TZ's own acceptance note: "tmp_path
ichida vaqtinchalik allowed_roots bilan barcha operatsiyalar". Deletion tests
monkeypatch ``send2trash.send2trash`` so no test ever touches the real
Recycle Bin.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultron.config import RootConfig
from ultron.memory import db as memory_db
from ultron.tools import file_tools

# Captured before any fixture monkeypatches ``_blocked_roots`` below, so
# ``test_blocked_roots_reads_expected_env_vars`` can restore the real thing.
_REAL_BLOCKED_ROOTS = file_tools._blocked_roots


@pytest.fixture(autouse=True)
async def _reset_module_state(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    # Pytest's own tmp_path lives under %LOCALAPPDATA%\Temp, which is itself
    # one of the §7B.1 blocked system folders -- neutralize the check by
    # default so tests can use tmp_path as a stand-in allowed_root without
    # every mutating call being rejected for a reason unrelated to what it's
    # testing. The two tests that exercise the block itself re-patch this.
    monkeypatch.setattr(file_tools, "_blocked_roots", lambda: [])

    db_path = tmp_path_factory.mktemp("ultron_db") / "test.sqlite"
    memory_db.configure(db_path)
    file_tools._indexer = None
    file_tools._last_results = []
    yield
    await memory_db.reset_for_tests()
    file_tools._config = None
    file_tools._indexer = None
    file_tools._last_results = []


@pytest.fixture
def configured(tmp_path: Path) -> RootConfig:
    config = RootConfig.model_validate(
        {"tools": {"files": {"allowed_roots": [str(tmp_path)], "max_results": 5}}}
    )
    file_tools.configure(config)
    return config


async def _seed_index(tmp_path: Path) -> None:
    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])
    await indexer.build_index()


# -- safety boundary (§7B.1) -------------------------------------------------


def test_path_inside_allowed_root_is_accepted(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    resolved = file_tools.resolve_within_allowed_roots(str(allowed_dir / "x.txt"), [allowed_dir])

    assert resolved == (allowed_dir / "x.txt").resolve()


def test_path_outside_allowed_roots_is_rejected(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(file_tools.FileAccessError):
        file_tools.resolve_within_allowed_roots(str(outside), [allowed_dir])


def test_blocked_system_folder_rejected_even_inside_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "Windows"
    blocked.mkdir()
    monkeypatch.setattr(file_tools, "_blocked_roots", lambda: [blocked.resolve()])

    with pytest.raises(file_tools.FileAccessError):
        file_tools.resolve_within_allowed_roots(str(blocked / "system32"), [tmp_path])


def test_blocked_roots_reads_expected_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_tools, "_blocked_roots", _REAL_BLOCKED_ROOTS)
    monkeypatch.setenv("WINDIR", str(Path.cwd()))
    for name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)

    roots = file_tools._blocked_roots()

    assert roots == [Path.cwd().resolve()]


def test_resolve_named_folder_recognizes_uz_aliases() -> None:
    assert file_tools._resolve_named_folder("ish stoli") == Path.home() / "Desktop"
    assert file_tools._resolve_named_folder("hujjatlarga") == Path.home() / "Documents"
    assert file_tools._resolve_named_folder("yuklamalarni") == Path.home() / "Downloads"
    assert file_tools._resolve_named_folder("completely_unknown_name") is None


def test_dedupe_name_appends_counter(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")

    assert file_tools._dedupe_name(target) == tmp_path / "file (1).txt"


def test_dedupe_name_returns_as_is_when_free(tmp_path: Path) -> None:
    target = tmp_path / "free.txt"

    assert file_tools._dedupe_name(target) == target


def test_short_path_uses_home_tilde() -> None:
    home_file = Path.home() / "Downloads" / "resume.pdf"

    assert file_tools._short_path(str(home_file)) == str(Path("~") / "Downloads" / "resume.pdf")


def test_short_path_outside_home_returns_as_is() -> None:
    outside = Path(Path.home().drive or "C:") / "definitely_not_home_xyz" / "x.txt"

    assert file_tools._short_path(str(outside)) == str(outside)


# -- FileIndexer --------------------------------------------------------------


async def test_build_index_scans_tmp_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_text("hi")

    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])

    count = await indexer.build_index()

    assert count == 2
    rows = await (await db.conn.execute("SELECT name FROM file_index ORDER BY name")).fetchall()
    assert {r[0] for r in rows} == {"a.txt", "b.pdf"}


async def test_build_index_skips_unreadable_subdirectory_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A OneDrive-placeholder-style directory that lists as a name but
    raises when actually scanned must not abort indexing the rest of the
    tree -- ``Path.rglob`` used to do exactly that (one real crash found
    while dogfooding: `Documents/мои рисунки` raised ``FileNotFoundError``
    partway through a real scan and silently truncated the whole index)."""
    (tmp_path / "a.txt").write_text("hi")
    broken = tmp_path / "broken"
    broken.mkdir()
    good = tmp_path / "good"
    good.mkdir()
    (good / "b.txt").write_text("hi")

    real_scandir = file_tools.os.scandir

    def fake_scandir(path="."):
        if Path(path) == broken:
            raise FileNotFoundError("simulated vanished directory")
        return real_scandir(path)

    monkeypatch.setattr(file_tools.os, "scandir", fake_scandir)

    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])

    count = await indexer.build_index()

    assert count == 2  # a.txt + good/b.txt -- "broken" skipped, not fatal
    rows = await (await db.conn.execute("SELECT name FROM file_index ORDER BY name")).fetchall()
    assert {r[0] for r in rows} == {"a.txt", "b.txt"}


async def test_upsert_and_remove_path(tmp_path: Path) -> None:
    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])
    f = tmp_path / "new.txt"
    f.write_text("x")

    await indexer.upsert_path(f)
    row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(f),))
    ).fetchone()
    assert row[0] == 1

    await indexer.remove_path(f)
    row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(f),))
    ).fetchone()
    assert row[0] == 0


async def test_index_event_handler_schedules_upsert_on_create(tmp_path: Path) -> None:
    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])
    handler = file_tools._IndexEventHandler(indexer, asyncio.get_running_loop())
    f = tmp_path / "watched.txt"
    f.write_text("x")

    handler.on_created(SimpleNamespace(is_directory=False, src_path=str(f)))
    await asyncio.sleep(0.05)

    row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(f),))
    ).fetchone()
    assert row[0] == 1


async def test_index_event_handler_schedules_remove_on_delete(tmp_path: Path) -> None:
    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])
    f = tmp_path / "gone.txt"
    f.write_text("x")
    await indexer.upsert_path(f)
    handler = file_tools._IndexEventHandler(indexer, asyncio.get_running_loop())

    handler.on_deleted(SimpleNamespace(is_directory=False, src_path=str(f)))
    await asyncio.sleep(0.05)

    row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(f),))
    ).fetchone()
    assert row[0] == 0


async def test_index_event_handler_schedules_move(tmp_path: Path) -> None:
    db = await memory_db.get_db()
    indexer = file_tools.FileIndexer(db, [tmp_path])
    src = tmp_path / "old.txt"
    src.write_text("x")
    await indexer.upsert_path(src)
    dst = tmp_path / "renamed.txt"
    src.rename(dst)
    handler = file_tools._IndexEventHandler(indexer, asyncio.get_running_loop())

    handler.on_moved(SimpleNamespace(is_directory=False, src_path=str(src), dest_path=str(dst)))
    await asyncio.sleep(0.05)

    old_row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(src),))
    ).fetchone()
    new_row = await (
        await db.conn.execute("SELECT COUNT(*) FROM file_index WHERE path = ?", (str(dst),))
    ).fetchone()
    assert old_row[0] == 0
    assert new_row[0] == 1


async def test_search_files_index_performance_50k_rows() -> None:
    db = await memory_db.get_db()
    rows = [(f"C:/data/file_{i}.txt", f"file_{i}.txt", "txt", 100, 0.0) for i in range(50_000)]
    rows.append(("C:/data/rezyume_maxsus.pdf", "rezyume_maxsus.pdf", "pdf", 100, 0.0))
    await db.conn.executemany(file_tools._UPSERT_FILE_INDEX_SQL, rows)
    await db.conn.commit()

    start = time.monotonic()
    results = await file_tools.search_files_index(
        db, "rezyume", extension=None, folder=None, modified_within_days=None, max_results=10
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    assert elapsed_ms <= 300
    assert any("rezyume" in r["name"] for r in results)


# -- search_files / open_file tools ------------------------------------------


async def test_search_files_returns_results(configured: RootConfig, tmp_path: Path) -> None:
    (tmp_path / "resume_2024.pdf").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    await _seed_index(tmp_path)

    result = await file_tools.search_files(query="resume")

    assert result.ok is True
    assert result.data["results"][0]["name"] == "resume_2024.pdf"


async def test_search_files_no_match(configured: RootConfig, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x")
    await _seed_index(tmp_path)

    result = await file_tools.search_files(query="zzz_no_such_thing_zzz")

    assert result.ok is False


async def test_search_files_extension_filter(configured: RootConfig, tmp_path: Path) -> None:
    (tmp_path / "resume.pdf").write_text("x")
    (tmp_path / "resume.txt").write_text("x")
    await _seed_index(tmp_path)

    result = await file_tools.search_files(query="resume", extension="pdf")

    names = {r["name"] for r in result.data["results"]}
    assert names == {"resume.pdf"}


async def test_open_file_direct_path_mock_mode(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    f = tmp_path / "doc.txt"
    f.write_text("x")

    result = await file_tools.open_file(path_or_query=str(f))

    assert result.ok is True


async def test_open_file_by_search_single_match(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    (tmp_path / "uniquename123.txt").write_text("x")
    await _seed_index(tmp_path)

    result = await file_tools.open_file(path_or_query="uniquename123")

    assert result.ok is True


async def test_open_file_ordinal_reference_after_search(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    (tmp_path / "report_alpha.txt").write_text("x")
    (tmp_path / "report_beta.txt").write_text("x")
    await _seed_index(tmp_path)
    assert (await file_tools.search_files(query="report")).ok is True

    result = await file_tools.open_file(path_or_query="ikkinchisini")

    assert result.ok is True
    assert "report" in result.data["opened"]


async def test_open_file_not_found(configured: RootConfig, tmp_path: Path) -> None:
    await _seed_index(tmp_path)

    result = await file_tools.open_file(path_or_query="zzz_nope_zzz")

    assert result.ok is False


# -- open_folder / list_folder / recent_files --------------------------------


async def test_open_folder_named_alias_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    config = RootConfig.model_validate(
        {"tools": {"files": {"allowed_roots": [str(Path.home() / "Desktop")]}}}
    )
    file_tools.configure(config)

    result = await file_tools.open_folder(name_or_path="ish stoli")

    assert result.ok is True


async def test_open_folder_outside_allowed_roots(configured: RootConfig, tmp_path: Path) -> None:
    result = await file_tools.open_folder(name_or_path=str(tmp_path.parent))

    assert result.ok is False


async def test_list_folder_returns_entries(configured: RootConfig, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()

    result = await file_tools.list_folder(path=str(tmp_path))

    assert result.ok is True
    assert {e["name"] for e in result.data["entries"]} == {"a.txt", "sub"}


async def test_list_folder_not_a_directory(configured: RootConfig, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")

    result = await file_tools.list_folder(path=str(f))

    assert result.ok is False


async def test_recent_files_orders_by_mtime_desc(configured: RootConfig, tmp_path: Path) -> None:
    old = tmp_path / "old.txt"
    old.write_text("x")
    time.sleep(0.01)
    new = tmp_path / "new.txt"
    new.write_text("x")

    result = await file_tools.recent_files(folder=str(tmp_path), limit=5)

    assert result.ok is True
    assert result.data["files"][0]["name"] == "new.txt"


async def test_recent_files_literal_path_is_respected(
    configured: RootConfig, tmp_path: Path
) -> None:
    """Regression: a literal folder path (not a Uz alias) must not silently
    fall back to ~/Downloads."""
    sub = tmp_path / "custom"
    sub.mkdir()
    (sub / "only.txt").write_text("x")

    result = await file_tools.recent_files(folder=str(sub))

    assert result.ok is True
    assert result.data["files"][0]["name"] == "only.txt"


# -- create_folder / rename_item / move_item / copy_item / delete_item ------


async def test_create_folder_real_mode(configured: RootConfig, tmp_path: Path) -> None:
    result = await file_tools.create_folder(parent=str(tmp_path), name="NewFolder")

    assert result.ok is True
    assert (tmp_path / "NewFolder").is_dir()


async def test_create_folder_mock_mode_does_not_touch_disk(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await file_tools.create_folder(parent=str(tmp_path), name="Ghost")

    assert result.ok is True
    assert not (tmp_path / "Ghost").exists()


async def test_create_folder_already_exists(configured: RootConfig, tmp_path: Path) -> None:
    (tmp_path / "Existing").mkdir()

    result = await file_tools.create_folder(parent=str(tmp_path), name="Existing")

    assert result.ok is False


async def test_create_folder_outside_allowed_roots_rejected(
    configured: RootConfig, tmp_path: Path
) -> None:
    result = await file_tools.create_folder(parent=str(tmp_path.parent), name="x")

    assert result.ok is False


async def test_rename_item_real_mode(configured: RootConfig, tmp_path: Path) -> None:
    f = tmp_path / "old.txt"
    f.write_text("hi")

    result = await file_tools.rename_item(path=str(f), new_name="new.txt")

    assert result.ok is True
    assert not f.exists()
    assert (tmp_path / "new.txt").is_file()


async def test_rename_item_dedupes_on_collision(configured: RootConfig, tmp_path: Path) -> None:
    f = tmp_path / "old.txt"
    f.write_text("hi")
    (tmp_path / "new.txt").write_text("existing")

    result = await file_tools.rename_item(path=str(f), new_name="new.txt")

    assert result.ok is True
    assert (tmp_path / "new (1).txt").is_file()


async def test_rename_item_missing_source(configured: RootConfig, tmp_path: Path) -> None:
    result = await file_tools.rename_item(path=str(tmp_path / "nope.txt"), new_name="x.txt")

    assert result.ok is False


async def test_rename_item_mock_mode_does_not_touch_disk(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    f = tmp_path / "old.txt"
    f.write_text("hi")

    result = await file_tools.rename_item(path=str(f), new_name="new.txt")

    assert result.ok is True
    assert f.exists()
    assert not (tmp_path / "new.txt").exists()


async def test_move_item_real_mode(configured: RootConfig, tmp_path: Path) -> None:
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    f = src_dir / "a.txt"
    f.write_text("hi")

    result = await file_tools.move_item(src=str(f), dst_folder=str(dst_dir))

    assert result.ok is True
    assert not f.exists()
    assert (dst_dir / "a.txt").is_file()


async def test_move_item_dedupes_on_collision(configured: RootConfig, tmp_path: Path) -> None:
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    f = src_dir / "a.txt"
    f.write_text("hi")
    (dst_dir / "a.txt").write_text("existing")

    result = await file_tools.move_item(src=str(f), dst_folder=str(dst_dir))

    assert result.ok is True
    assert (dst_dir / "a (1).txt").is_file()


async def test_move_item_mock_mode_does_not_touch_disk(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    f = src_dir / "a.txt"
    f.write_text("hi")

    result = await file_tools.move_item(src=str(f), dst_folder=str(dst_dir))

    assert result.ok is True
    assert f.exists()
    assert not (dst_dir / "a.txt").exists()


async def test_copy_item_file(configured: RootConfig, tmp_path: Path) -> None:
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    f = src_dir / "a.txt"
    f.write_text("hi")

    result = await file_tools.copy_item(src=str(f), dst_folder=str(dst_dir))

    assert result.ok is True
    assert f.exists()
    assert (dst_dir / "a.txt").read_text() == "hi"


async def test_copy_item_directory(configured: RootConfig, tmp_path: Path) -> None:
    src_dir, dst_dir = tmp_path / "srcdir", tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()
    (src_dir / "inner.txt").write_text("hi")

    result = await file_tools.copy_item(src=str(src_dir), dst_folder=str(dst_dir))

    assert result.ok is True
    assert (dst_dir / "srcdir" / "inner.txt").is_file()


async def test_move_and_copy_reject_paths_outside_allowed_roots(
    configured: RootConfig, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "definitely_outside_xyz_move"
    outside.mkdir(exist_ok=True)
    f = outside / "x.txt"
    f.write_text("hi")
    try:
        move_result = await file_tools.move_item(src=str(f), dst_folder=str(tmp_path))
        copy_result = await file_tools.copy_item(src=str(f), dst_folder=str(tmp_path))
        assert move_result.ok is False
        assert copy_result.ok is False
    finally:
        f.unlink()
        outside.rmdir()


async def test_delete_item_real_mode_calls_send2trash(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "gone.txt"
    f.write_text("hi")
    calls: list[str] = []
    monkeypatch.setattr(file_tools.send2trash, "send2trash", lambda p: calls.append(p))

    result = await file_tools.delete_item(path=str(f))

    assert result.ok is True
    assert calls == [str(f)]


async def test_delete_item_mock_mode_does_not_call_send2trash(
    configured: RootConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    f = tmp_path / "gone.txt"
    f.write_text("hi")
    calls: list[str] = []
    monkeypatch.setattr(file_tools.send2trash, "send2trash", lambda p: calls.append(p))

    result = await file_tools.delete_item(path=str(f))

    assert result.ok is True
    assert calls == []
    assert f.exists()


async def test_delete_item_missing_source(configured: RootConfig, tmp_path: Path) -> None:
    result = await file_tools.delete_item(path=str(tmp_path / "nope.txt"))

    assert result.ok is False


async def test_delete_item_outside_allowed_roots_rejected(
    configured: RootConfig, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "definitely_outside_xyz_delete"
    outside.mkdir(exist_ok=True)
    f = outside / "x.txt"
    f.write_text("hi")
    try:
        result = await file_tools.delete_item(path=str(f))
        assert result.ok is False
    finally:
        f.unlink()
        outside.rmdir()
