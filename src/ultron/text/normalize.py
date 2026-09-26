"""STT output cleanup (docs/tz/03_stt.md §3.4).

``normalize_transcript`` runs the four required steps in order: script
consolidation (mixed Cyrillic -> Latin), apostrophe canonicalization,
whitespace/punctuation trimming, and wake-word variant canonicalization.
Case is preserved, as the TZ requires it kept for the LLM; callers that need
the lowercase fast-path variant call ``.lower()`` on the result themselves
(plain ASCII-range lowering is enough once the text is Latin).
"""

from __future__ import annotations

import re

from ultron.text.translit import cyrillic_ratio, cyrillic_to_latin, normalize_apostrophes

_CYRILLIC_RATIO_THRESHOLD = 0.30

_STRIP_CHARS = ' \t\n\r.,!?;:()[]{}"“”«»-–—…'

_WAKE_WORD_RE = re.compile(
    r"\b(?:ultron|altron|oltron|ultiron|altiron|al[\s-]+tron)\b", re.IGNORECASE
)
_CANONICAL_WAKE_WORD = "Altron"


def normalize_transcript(raw: str) -> str:
    """Clean up one raw STT hypothesis into normalized, Latin-script text."""
    if not raw:
        return ""

    text = raw
    if cyrillic_ratio(text) > _CYRILLIC_RATIO_THRESHOLD:
        text = cyrillic_to_latin(text)

    text = normalize_apostrophes(text)
    text = re.sub(r"\s+", " ", text).strip(_STRIP_CHARS)
    text = _WAKE_WORD_RE.sub(_CANONICAL_WAKE_WORD, text)

    return text
