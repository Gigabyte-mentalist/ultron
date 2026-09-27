"""Personal STT dataset labeling tool (docs/tz/11_uzbek_stt_finetune.md §3.2).

    uv run python scripts/label_tool.py

Local-only web UI (stdlib ``http.server`` -- no new dependency for a tool
that never leaves ``127.0.0.1``) over the utterances ``core/orchestrator.py``
already captured into ``data/recordings/utterances/YYYY-MM-DD/index.jsonl``
(``ultron.memory.utterance_log``, docs/tz/11 §3.1). For each still-unreviewed
recording (label "unlabeled" or "silver") it plays the audio and shows the
STT hypothesis, prioritizing failed-tool-call and low-confidence recordings
first (§3.2: "avval past confidence va muvaffaqiyatsiz tool chaqiruvlari") --
those are the ones most likely to be wrong, and most valuable to fix.

Three actions per item, each written back into that day's ``index.jsonl`` in
place (the single source of truth -- ``scripts/build_personal_dataset.py``
reads it later to build the actual train/dev/test split):
  - correct   ("✅"): the STT hypothesis is used as-is.
  - corrected ("✏️"): the edited text in the textarea is used instead.
  - discard   ("🗑"): the audio is deleted immediately (frees disk right away)
    and the row is marked "discarded" (``UtteranceLog.cleanup_stale()``
    removes the row itself on the next process start).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.config import ConfigError, load_config

_DAY_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ID_RE = re.compile(r"[0-9a-f]{32}")
_REVIEWABLE_LABELS = frozenset({"unlabeled", "silver"})


def _is_valid_day(value: str) -> bool:
    return _DAY_RE.fullmatch(value) is not None


def _is_valid_id(value: str) -> bool:
    return _ID_RE.fullmatch(value) is not None


class Queue:
    """Rebuilds the reviewable-item list from disk on every call -- this tool
    is single-user/single-tab and the index files are small, so there's no
    need for an in-memory cache that could drift from what other processes
    (label_tool restarted, cleanup_stale) do to the same files."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _day_dirs(self) -> list[Path]:
        if not self.base_dir.is_dir():
            return []
        return sorted(p for p in self.base_dir.iterdir() if p.is_dir())

    def next_item(self) -> dict | None:
        candidates: list[dict] = []
        for day_dir in self._day_dirs():
            index_path = day_dir / "index.jsonl"
            if not index_path.is_file():
                continue
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("label", "unlabeled") not in _REVIEWABLE_LABELS:
                    continue
                row["day"] = day_dir.name
                candidates.append(row)
        if not candidates:
            return None

        def sort_key(row: dict) -> tuple[int, float]:
            tool_failed = row.get("tool_called") and row.get("tool_ok") is False
            return (0 if tool_failed else 1, row.get("confidence") or 0.0)

        candidates.sort(key=sort_key)
        best = candidates[0]
        best["remaining"] = len(candidates)
        return best

    def apply_label(self, *, day: str, item_id: str, action: str, text: str | None) -> None:
        day_dir = self.base_dir / day
        index_path = day_dir / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(day)

        lines = index_path.read_text(encoding="utf-8").splitlines()
        updated_lines: list[str] = []
        found = False
        target_path: Path | None = None
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == item_id:
                found = True
                target_path = day_dir / row["path"]
                if action == "correct":
                    row["label"] = "correct"
                    row["final_text"] = row.get("stt_text", "")
                elif action == "corrected":
                    row["label"] = "corrected"
                    row["final_text"] = text or ""
                elif action == "discard":
                    row["label"] = "discarded"
                else:
                    raise ValueError(f"Noma'lum amal: {action}")
            updated_lines.append(json.dumps(row, ensure_ascii=False))
        if not found:
            raise KeyError(item_id)

        index_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        if action == "discard" and target_path is not None:
            target_path.unlink(missing_ok=True)


_PAGE = """<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8">
<title>Ultron -- STT belgilash</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 1.3rem; }
  .meta { color: #555; font-size: 0.9rem; margin-bottom: 12px; }
  .meta span { margin-right: 14px; }
  .bad { color: #b00020; font-weight: 600; }
  textarea { width: 100%; height: 90px; font-size: 1.1rem; padding: 8px; box-sizing: border-box; }
  .buttons { margin-top: 14px; display: flex; gap: 10px; }
  button { font-size: 1rem; padding: 10px 16px; cursor: pointer; }
  #correct { background: #d7f5da; border: 1px solid #4caf50; }
  #corrected { background: #fff4cc; border: 1px solid #d9a400; }
  #discard { background: #fbd7d7; border: 1px solid #b00020; }
  #done { text-align: center; margin-top: 60px; color: #4caf50; font-size: 1.3rem; }
  .hint { color: #888; font-size: 0.85rem; margin-top: 18px; }
  audio { width: 100%; margin-top: 10px; }
</style>
</head>
<body>
<h1>Ultron -- shaxsiy STT ma'lumotlarini belgilash</h1>
<div id="app">
  <div class="meta" id="meta"></div>
  <audio id="player" controls autoplay></audio>
  <textarea id="text"></textarea>
  <div class="buttons">
    <button id="correct">✅ To'g'ri</button>
    <button id="corrected">✏️ Tuzatildi</button>
    <button id="discard">🗑 Tashlash</button>
  </div>
  <div class="hint">Enter = to'g'ri/tuzatildi (matn o'zgargan bo'lsa avtomatik "tuzatildi"),
    Ctrl+Delete = tashlash, Bo'sh joy (matn maydonidan tashqarida) = qayta eshittirish.</div>
</div>
<div id="done" style="display:none">Hammasi ko'rib chiqildi. 🎉</div>
<script>
let current = null;

async function loadNext() {
  const res = await fetch('/api/next');
  const data = await res.json();
  if (!data.item) {
    document.getElementById('app').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    return;
  }
  current = data.item;
  const meta = document.getElementById('meta');
  const toolBad = current.tool_called && current.tool_ok === false;
  meta.innerHTML =
    '<span>Qolgan: ' + current.remaining + '</span>' +
    '<span>Ishonch: ' + (current.confidence ?? '?') + '</span>' +
    '<span>Manba: ' + current.wake_source + '</span>' +
    (toolBad ? '<span class="bad">Tool muvaffaqiyatsiz</span>' : '');
  document.getElementById('text').value = current.stt_text || '';
  const player = document.getElementById('player');
  player.src = '/audio/' + current.day + '/' + current.id + '.flac';
  player.play().catch(() => {});
  document.getElementById('text').focus();
}

async function sendLabel(action) {
  if (!current) return;
  const text = document.getElementById('text').value;
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({day: current.day, id: current.id, action, text}),
  });
  loadNext();
}

document.getElementById('correct').onclick = () => sendLabel('correct');
document.getElementById('corrected').onclick = () => sendLabel('corrected');
document.getElementById('discard').onclick = () => sendLabel('discard');

document.addEventListener('keydown', (e) => {
  const inText = document.activeElement === document.getElementById('text');
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const edited = document.getElementById('text').value.trim();
    const original = (current && current.stt_text || '').trim();
    sendLabel(edited === original ? 'correct' : 'corrected');
  } else if (e.key === 'Delete' && e.ctrlKey) {
    e.preventDefault();
    sendLabel('discard');
  } else if (e.key === ' ' && !inText) {
    e.preventDefault();
    document.getElementById('player').play();
  }
});

loadNext();
</script>
</body>
</html>
"""


def _make_handler(queue: Queue) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            logger.debug("{}", format % args)

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's own naming
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_PAGE)
            elif parsed.path == "/api/next":
                self._send_json({"item": queue.next_item()})
            elif parsed.path.startswith("/audio/"):
                self._serve_audio(parsed.path)
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/label":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length))
                day, item_id, action = body["day"], body["id"], body["action"]
                text = body.get("text")
                if not _is_valid_day(day) or not _is_valid_id(item_id):
                    raise ValueError("noto'g'ri day/id")
                if action == "corrected" and not (text and text.strip()):
                    raise ValueError("tuzatilgan matn bo'sh bo'lishi mumkin emas")
                queue.apply_label(day=day, item_id=item_id, action=action, text=text)
            except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"ok": True})

        def _serve_audio(self, path: str) -> None:
            parts = path.split("/")
            if len(parts) != 4 or not parts[3].endswith(".flac"):
                self.send_error(404)
                return
            _, _, day, filename = parts
            item_id = filename[: -len(".flac")]
            if not _is_valid_day(day) or not _is_valid_id(item_id):
                self.send_error(400)
                return
            audio_path = queue.base_dir / day / filename
            if not audio_path.is_file():
                self.send_error(404)
                return
            data = audio_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/flac")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: dict, *, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron shaxsiy STT ma'lumotlarini belgilash")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        try:
            config = load_config()
        except ConfigError as exc:
            logger.info("Config yuklab bo'lmadi:\n{}", exc)
            return 1
        data_dir = Path(config.app.data_dir)

    base_dir = data_dir / "recordings" / "utterances"
    if not base_dir.is_dir():
        logger.info(
            "{} topilmadi -- hali hech qanday utterance yozilmagan "
            "(config.yaml'da memory.save_utterances=true ekanligini tekshiring).",
            base_dir,
        )
        return 1

    queue = Queue(base_dir)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(queue))
    url = f"http://{args.host}:{args.port}/"
    logger.info("Belgilash vositasi: {}", url)
    logger.info("To'xtatish uchun Ctrl+C bosing.")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
