"""TTS text preprocessing pipeline (docs/tz/05_tts.md §3.3).

``clean_text`` runs steps 1-5 in order (wake-word removal, markdown/emoji/URL
cleanup, numbers -> words, abbreviation/pronunciation lookup, optional
Latin -> Cyrillic transliteration). ``split_sentences`` is the separate
step 6, used by ``scripts/tts_say.py`` and ``tts_compare.py`` to turn a
block of text into the per-sentence chunks the streaming interface expects
(docs/tz/00_umumiy_arxitektura.md's LLM sentence splitter does the same job
for live LLM output, so ``MMSEngine`` itself only ever sees one sentence at
a time in that path).
"""

from __future__ import annotations

import re

from ultron.text.numbers_uz import convert_numbers_in_text
from ultron.text.translit import latin_to_cyrillic

# Step 1: self-trigger guard. Ultron refers to itself as "men" (never says
# its own name), so any wake-word variant that slips into LLM output must be
# removed before synthesis (docs/tz/00_umumiy_arxitektura.md §7 point 6).
# Uzbek is agglutinative (a personal suffix attaches directly with no space,
# e.g. "Ultronman" = "I am Ultron"), so the match extends through any
# trailing letters of the same word rather than stopping at a word boundary.
_WAKE_WORD_RE = re.compile(
    r"\s*[,]?\s*\b(?:ultron|altron|oltron|ultiron|altiron)[a-z']*\s*[,]?\s*", re.IGNORECASE
)

# Step 2: markdown / emoji / URL cleanup.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_MARKDOWN_CHARS_RE = re.compile(r"[*#`_]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001fadf"  # symbols & pictographs, extended-A
    "\U0001f1e6-\U0001f1ff"  # flags
    "☀-➿"  # misc symbols & dingbats
    "←-⇿"  # arrows
    "️"  # variation selector
    "]+"
)

# Step 4: abbreviation and app-name pronunciation dictionary
# (docs/tz/05_tts.md §3.3 point 4 and §6 risk table). Kept intentionally
# small; grow it as new mismatches turn up in practice.
_ABBREVIATIONS: dict[str, str] = {
    "GB": "gigabayt",
    "MB": "megabayt",
    "KB": "kilobayt",
    "TB": "terabayt",
    "km": "kilometr",
    "kg": "kilogramm",
    "CPU": "protsessor",
    "GPU": "videokarta",
    "RAM": "operativ xotira",
    "YouTube": "Yutub",
    "Chrome": "Xrom",
    "Google": "Gugl",
    "WhatsApp": "Vatsap",
}
_ABBREVIATION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")\b")

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_DUPLICATE_PUNCT_RE = re.compile(r"[,;]\s*(?=[.!?])")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+")
_MAX_SENTENCE_CHARS = 200
_MIN_FRAGMENT_WORDS = 3


def _strip_wake_words(text: str) -> str:
    return _WAKE_WORD_RE.sub(" ", text)


def _clean_markup(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = _URL_RE.sub("havola", text)
    text = _MARKDOWN_CHARS_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text


def _expand_abbreviations(text: str) -> str:
    return _ABBREVIATION_RE.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)


def _tidy_whitespace(text: str) -> str:
    text = _DUPLICATE_PUNCT_RE.sub("", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(lambda m: m.group(1), text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_text(text: str, *, model_script: str = "cyrillic") -> str:
    """Run steps 1-5 of docs/tz/05_tts.md §3.3 on one chunk of text."""
    text = _strip_wake_words(text)
    text = _clean_markup(text)
    text = convert_numbers_in_text(text)
    text = _expand_abbreviations(text)
    text = _tidy_whitespace(text)
    if model_script == "cyrillic":
        text = latin_to_cyrillic(text)
    return text


def _split_long_sentence(sentence: str) -> list[str]:
    parts = [p.strip() for p in sentence.split(",")]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return [sentence]
    return [p + "," if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _merge_short_fragments(sentences: list[str]) -> list[str]:
    if len(sentences) <= 1:
        return sentences
    result: list[str] = []
    pending = ""
    for sentence in sentences:
        combined = f"{pending} {sentence}".strip() if pending else sentence
        if len(combined.split()) < _MIN_FRAGMENT_WORDS:
            pending = combined
            continue
        result.append(combined)
        pending = ""
    if pending:
        if result:
            result[-1] = f"{result[-1]} {pending}"
        else:
            result.append(pending)
    return result


def split_sentences(text: str) -> list[str]:
    """Step 6: split into per-sentence chunks, ready for one synthesize() call each."""
    text = text.strip()
    if not text:
        return []

    raw_sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    sentences: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) <= _MAX_SENTENCE_CHARS:
            sentences.append(sentence)
        else:
            sentences.extend(_split_long_sentence(sentence))
    return _merge_short_fragments(sentences)
