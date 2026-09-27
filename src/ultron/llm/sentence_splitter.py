"""Incremental sentence splitter for streaming LLM output (docs/tz/06_llm.md §3.6).

Feed token deltas as they arrive from ``OllamaClient.chat_stream``; a
complete sentence is yielded the moment it's recognized, so TTS can start on
it long before the model finishes the whole reply. ``<think>`` blocks are
stripped even if a delta splits the tag itself, since stripping runs on the
whole accumulated buffer on every ``feed()`` call, and text after an
unclosed ``<think>`` is held back until its ``</think>`` arrives.

Sentence boundaries require actual trailing whitespace after ``.!?`` (not
just "end of buffer so far") so a decimal like "5.5" or a number that hasn't
finished streaming in never gets mistaken for a sentence end mid-stream --
only ``flush()`` (called once the stream is actually done) treats a trailing
fragment as final regardless. Known abbreviations ("t.b.", "va h.k.") are
recognized by their exact suffix and skipped as false sentence ends.
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Only paired markdown markers are stripped (bold/code spans unwrapped to
# their content, leading '#' headers dropped) -- a *lone* "*" is left alone,
# since Qwen sometimes writes it as a multiplication sign ("17 * 23 = 391")
# and blanket-stripping every "*" turned that into the unreadable "17  23 = 391".
_MARKDOWN_RE = re.compile(r"\*\*(.+?)\*\*|`(.+?)`|^#{1,6}\s*", re.MULTILINE)
# Same ranges as tts/preprocess.py's emoji stripper (docs/tz/05_tts.md §3.3) --
# kept as its own copy since llm/ has no business depending on tts/.
_EMOJI_RE = re.compile("[\U0001f300-\U0001fadf\U0001f1e6-\U0001f1ff☀-➿←-⇿️]+")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s)")
_ABBREVIATION_SUFFIXES = ("t.b.", "h.k.")
_EARLY_EXIT_MIN_CHARS = 60


class SentenceSplitter:
    """Stateful: call ``feed()`` per streamed delta, ``flush()`` once at the end."""

    def __init__(self) -> None:
        self._buffer = ""
        self._first_sentence_done = False

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        return self._extract_ready_sentences()

    def flush(self) -> str | None:
        self._buffer = _THINK_RE.sub("", self._buffer)
        open_idx = self._buffer.find("<think>")
        text = self._buffer[:open_idx] if open_idx != -1 else self._buffer
        text = self._clean(text)
        self._buffer = ""
        return text or None

    def _extract_ready_sentences(self) -> list[str]:
        sentences: list[str] = []
        while True:
            sentence = self._pop_one_sentence()
            if sentence is None:
                break
            if sentence:  # a boundary was found but cleaned to nothing (pure emoji/markdown)
                sentences.append(sentence)
        return sentences

    def _pop_one_sentence(self) -> str | None:
        self._buffer = _THINK_RE.sub("", self._buffer)
        open_idx = self._buffer.find("<think>")
        scan_limit = open_idx if open_idx != -1 else len(self._buffer)

        end = self._find_boundary(self._buffer[:scan_limit])
        if end is None:
            return None

        raw = self._buffer[:end]
        self._buffer = self._buffer[end:]
        self._first_sentence_done = True
        return self._clean(raw)

    def _find_boundary(self, text: str) -> int | None:
        for match in _SENTENCE_END_RE.finditer(text):
            end = match.end()
            if text[:end].endswith(_ABBREVIATION_SUFFIXES):
                continue
            return end + 1 if end < len(text) and text[end] == " " else end

        if not self._first_sentence_done:
            return self._early_exit_index(text)
        return None

    @staticmethod
    def _early_exit_index(text: str) -> int | None:
        if len(text) < _EARLY_EXIT_MIN_CHARS:
            return None
        comma_idx = text.find(",")
        if comma_idx == -1:
            return None
        end = comma_idx + 1
        return end + 1 if end < len(text) and text[end] == " " else end

    @staticmethod
    def _clean(text: str) -> str:
        text = _THINK_RE.sub("", text)
        text = _MARKDOWN_RE.sub(lambda m: next((g for g in m.groups() if g is not None), ""), text)
        text = _EMOJI_RE.sub("", text).strip()
        # a "sentence" left with no letters/digits (bare punctuation, once its
        # only real content was an emoji) isn't worth speaking -- drop it.
        return text if any(ch.isalnum() for ch in text) else ""
