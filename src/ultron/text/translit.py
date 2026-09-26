"""Uzbek Cyrillic <-> Latin transliteration (docs/tz/03_stt.md §3.3).

Rule-based, word-by-word. A small exception dictionary covers loanwords
where the mechanical letter rules would otherwise disagree with the
commonly accepted spelling; it starts small on purpose and is meant to grow
as new mismatches are found in practice.
"""

from __future__ import annotations

import re

_CANONICAL_APOSTROPHE = "'"
_APOSTROPHE_VARIANTS = "ʻʼ‘’`´"
_APOSTROPHE_TRANSLATION = str.maketrans({ch: _CANONICAL_APOSTROPHE for ch in _APOSTROPHE_VARIANTS})

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЎўҚқҒғҲҳЁёЙй'ʻʼ‘’`´]+")

_CYR_VOWELS = set("аоуўиеэёюя")

# Single-letter Cyrillic -> Latin, lowercase keys. ц and е are handled
# separately below because their mapping depends on word position.
_CYR_TO_LAT_MAP: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ш": "sh",
    "ч": "ch",
    "ў": "o'",
    "ғ": "g'",
    "қ": "q",
    "ҳ": "h",
    "э": "e",
    "ъ": "'",
    "ь": "",
    "ё": "yo",
    "ю": "yu",
    "я": "ya",
    # Not part of the official Uzbek Cyrillic alphabet, but Russian loanwords
    # occasionally keep their original spelling in transcripts.
    "ы": "i",
    "щ": "sh",
}

_LAT_TO_CYR_MAP: dict[str, str] = {
    "a": "а",
    "b": "б",
    "d": "д",
    "f": "ф",
    "g": "г",
    "h": "ҳ",
    "i": "и",
    "j": "ж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "қ",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "x": "х",
    "y": "й",
    "z": "з",
    "c": "ц",
}

_LAT_TWO_CHAR_MAP: dict[str, str] = {
    "sh": "ш",
    "ch": "ч",
    "ng": "нг",
    "yo": "ё",
    "yu": "ю",
    "ya": "я",
    "ye": "е",
}

# Small, deliberately short starting dictionary for loanwords where the
# mechanical rules disagree with the commonly accepted spelling. Extend as
# new mismatches turn up in real transcripts.
_CYR_TO_LAT_EXCEPTIONS: dict[str, str] = {
    "объект": "obyekt",
    "субъект": "subyekt",
    "объектив": "obyektiv",
    "компьютер": "kompyuter",
    "сентябрь": "sentyabr",
    "октябрь": "oktyabr",
    "ноябрь": "noyabr",
    "декабрь": "dekabr",
    "январь": "yanvar",
    "февраль": "fevral",
    "апрель": "aprel",
}
_LAT_TO_CYR_EXCEPTIONS: dict[str, str] = {v: k for k, v in _CYR_TO_LAT_EXCEPTIONS.items()}


def normalize_apostrophes(text: str) -> str:
    """Collapse every apostrophe-like character to the canonical ``'``."""
    return text.translate(_APOSTROPHE_TRANSLATION)


def _match_case(original: str, converted: str) -> str:
    if not converted:
        return converted
    if original.isupper() and len(original) > 1:
        return converted.upper()
    if original[:1].isupper():
        return converted[0].upper() + converted[1:]
    return converted


def _convert_word_cyr_to_lat(word: str) -> str:
    key = word.lower()
    if key in _CYR_TO_LAT_EXCEPTIONS:
        return _match_case(word, _CYR_TO_LAT_EXCEPTIONS[key])

    out: list[str] = []
    for i, ch in enumerate(word):
        if ch == _CANONICAL_APOSTROPHE:
            out.append(ch)
            continue
        lower = ch.lower()

        if lower == "ц":
            mapped = "s" if i == 0 else "ts"
        elif lower == "е":
            prev = word[i - 1].lower() if i > 0 else None
            mapped = "ye" if (i == 0 or prev in _CYR_VOWELS) else "e"
        else:
            mapped = _CYR_TO_LAT_MAP.get(lower)
            if mapped is None:
                out.append(ch)
                continue

        out.append(_match_case(ch, mapped))
    return "".join(out)


def _convert_word_lat_to_cyr(word: str) -> str:
    key = word.lower()
    if key in _LAT_TO_CYR_EXCEPTIONS:
        return _match_case(word, _LAT_TO_CYR_EXCEPTIONS[key])

    out: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        two = word[i : i + 2].lower()

        if two in ("o'", "g'"):
            mapped = "ў" if two == "o'" else "ғ"
            out.append(_match_case(ch, mapped))
            i += 2
            continue

        if two in _LAT_TWO_CHAR_MAP:
            out.append(_match_case(ch, _LAT_TWO_CHAR_MAP[two]))
            i += 2
            continue

        if ch == _CANONICAL_APOSTROPHE:
            out.append("ъ")
            i += 1
            continue

        low = ch.lower()
        if low == "e":
            mapped = "э" if i == 0 else "е"
        else:
            mapped = _LAT_TO_CYR_MAP.get(low, ch)

        out.append(_match_case(ch, mapped))
        i += 1
    return "".join(out)


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Uzbek Cyrillic text to Latin script, preserving punctuation/case."""
    text = normalize_apostrophes(text)
    return _WORD_RE.sub(lambda m: _convert_word_cyr_to_lat(m.group(0)), text)


def latin_to_cyrillic(text: str) -> str:
    """Transliterate Uzbek Latin text to Cyrillic script, preserving punctuation/case."""
    text = normalize_apostrophes(text)
    return _WORD_RE.sub(lambda m: _convert_word_lat_to_cyr(m.group(0)), text)


def cyrillic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Cyrillic (0.0 if there are none)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum(1 for ch in letters if "а" <= ch.lower() <= "я" or ch.lower() in "ўқғҳё")
    return cyrillic / len(letters)
