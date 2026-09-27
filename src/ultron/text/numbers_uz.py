"""Uzbek number -> words conversion for TTS (docs/tz/05_tts.md §3.3 point 3).

Covers plain integers (0..999,999,999,999), decimals, percent, clock time,
calendar dates, ordinals, and temperatures. ``convert_numbers_in_text`` finds
every one of these patterns in a chunk of text and replaces it with its
spoken-word form; date/time/percent/temperature/ordinal patterns are matched
before the generic decimal/integer patterns so a run like ``27.09.2026``
becomes one date reading rather than three separate numbers.

Numeral-reading convention (matches the examples in docs/tz/05_tts.md
exactly, e.g. 2026 -> "ikki ming yigirma olti", 27.09.2026 -> "yigirma
yettinchi sentyabr, ikki ming yigirma oltinchi yil"):
  - "bir" is dropped before "yuz" (100) and "ming" (1000) when that group's
    value is exactly 1 -- Uzbek says "yuz so'm", "ming so'm", not "bir yuz".
  - "bir" is kept before "million"/"milliard" ("bir million"), since
    dropping it there reads as naming the unit rather than counting it.
  - Ordinal suffix selection ("-inchi" vs "-nchi") depends only on whether
    the last word of the cardinal reading ends in a vowel or a consonant --
    this is the actual Uzbek grammar rule, and it is what produces
    "yettinchi" (yetti + nchi) and "beshinchi" (besh + inchi) correctly
    without any irregular-word list.
"""

from __future__ import annotations

import re

_ONES = (
    "nol",
    "bir",
    "ikki",
    "uch",
    "to'rt",
    "besh",
    "olti",
    "yetti",
    "sakkiz",
    "to'qqiz",
)
_TENS = (
    "",
    "o'n",
    "yigirma",
    "o'ttiz",
    "qirq",
    "ellik",
    "oltmish",
    "yetmish",
    "sakson",
    "to'qson",
)
# (scale name, omit "bir" when that group's value is exactly 1)
_SCALES = (("", False), ("ming", True), ("million", False), ("milliard", False))

_MAX_NUMBER = 999_999_999_999

_MONTH_NAMES = {
    1: "yanvar",
    2: "fevral",
    3: "mart",
    4: "aprel",
    5: "may",
    6: "iyun",
    7: "iyul",
    8: "avgust",
    9: "sentyabr",
    10: "oktyabr",
    11: "noyabr",
    12: "dekabr",
}

_DECIMAL_DENOMINATOR_WORDS = {1: "o'n", 2: "yuz", 3: "ming"}

_VOWELS = set("aeiou")

_WORD_TO_ONE = {word: value for value, word in enumerate(_ONES)}
_WORD_TO_TEN = {word: value * 10 for value, word in enumerate(_TENS) if word}
_WORD_TO_SCALE = {"yuz": 100, "ming": 1000, "million": 1_000_000, "milliard": 1_000_000_000}


def _two_digit(n: int) -> str:
    """0..99, no trailing/leading "nol" (0 -> "")."""
    if n < 10:
        return _ONES[n] if n else ""
    tens, ones = divmod(n, 10)
    parts = [_TENS[tens]]
    if ones:
        parts.append(_ONES[ones])
    return " ".join(parts)


def _three_digit(n: int) -> str:
    """0..999, no trailing/leading "nol" (0 -> "")."""
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append("yuz" if hundreds == 1 else f"{_ONES[hundreds]} yuz")
    if rest:
        parts.append(_two_digit(rest))
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Convert an integer in [-999_999_999_999, 999_999_999_999] to Uzbek words."""
    if n < 0:
        return f"minus {number_to_words(-n)}"
    if n > _MAX_NUMBER:
        raise ValueError(f"number_to_words faqat {_MAX_NUMBER} gacha qo'llab-quvvatlaydi: {n}")
    if n == 0:
        return "nol"

    chunks = []
    remaining = n
    for _ in range(4):
        remaining, chunk = divmod(remaining, 1000)
        chunks.append(chunk)
    # chunks[0] = units group, chunks[1] = thousands, chunks[2] = millions, chunks[3] = milliards

    parts: list[str] = []
    for idx in (3, 2, 1, 0):
        value = chunks[idx]
        if value == 0:
            continue
        scale_name, omit_bir = _SCALES[idx]
        if not scale_name:
            parts.append(_three_digit(value))
        elif value == 1 and omit_bir:
            parts.append(scale_name)
        else:
            parts.append(f"{_three_digit(value)} {scale_name}")
    return " ".join(parts)


def ordinal_to_words(n: int) -> str:
    """Convert an integer to its Uzbek ordinal reading (5 -> "beshinchi")."""
    words = number_to_words(n)
    head, _, last_word = words.rpartition(" ")
    suffix = "nchi" if last_word[-1] in _VOWELS else "inchi"
    ordinal_last = last_word + suffix
    return f"{head} {ordinal_last}" if head else ordinal_last


def decimal_to_words(integer_part: int, fraction_digits: str) -> str:
    """3, "5" -> "uch butun o'ndan besh"; 3, "05" -> "uch butun yuzdan besh"."""
    stripped = fraction_digits.rstrip("0")
    if not stripped:
        return f"{number_to_words(integer_part)} butun"

    fraction_value = int(stripped)
    denominator_digits = len(stripped)
    denominator_word = _DECIMAL_DENOMINATOR_WORDS.get(denominator_digits)
    if denominator_word is None:
        denominator_word = number_to_words(10**denominator_digits)
    return (
        f"{number_to_words(integer_part)} butun "
        f"{denominator_word}dan {number_to_words(fraction_value)}"
    )


def percent_to_words(n: int) -> str:
    """50 -> "ellik foiz"."""
    return f"{number_to_words(n)} foiz"


def temperature_to_words(value: int) -> str:
    """-3 -> "minus uch daraja"; 25 -> "yigirma besh daraja"."""
    if value < 0:
        return f"minus {number_to_words(-value)} daraja"
    return f"{number_to_words(value)} daraja"


def time_to_words(hour: int, minute: int) -> str:
    """9, 30 -> "to'qqiz o'ttiz"; 14, 5 -> "o'n to'rt nol besh"."""
    hour_words = number_to_words(hour)
    if minute == 0:
        minute_words = "nol"
    elif minute < 10:
        minute_words = f"nol {_ONES[minute]}"
    else:
        minute_words = number_to_words(minute)
    return f"{hour_words} {minute_words}"


def date_to_words(day: int, month: int, year: int) -> str:
    """27, 9, 2026 -> "yigirma yettinchi sentyabr, ikki ming yigirma oltinchi yil"."""
    month_name = _MONTH_NAMES.get(month)
    if month_name is None:
        raise ValueError(f"Noto'g'ri oy raqami: {month}")
    return f"{ordinal_to_words(day)} {month_name}, {ordinal_to_words(year)} yil"


def words_to_number(text: str) -> int | None:
    """Parse an Uzbek number phrase ("ellik" -> 50, "ikki ming yigirma olti" ->
    2026) back into an integer, for fast-path argument parsing
    (docs/tz/06_llm.md §3.5). ``None`` if ``text`` isn't a recognized number.
    """
    tokens = text.lower().replace("-", " ").split()
    if not tokens:
        return None

    total = 0
    current = 0
    matched_any = False
    for token in tokens:
        if token in _WORD_TO_SCALE:
            scale = _WORD_TO_SCALE[token]
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            matched_any = True
        elif token in _WORD_TO_TEN:
            current += _WORD_TO_TEN[token]
            matched_any = True
        elif token in _WORD_TO_ONE:
            current += _WORD_TO_ONE[token]
            matched_any = True
        elif matched_any:
            break  # trailing unrelated words ("...ellik foiz qil") stop the number here
        else:
            return None

    return total + current if matched_any else None


# -- text-level regex pipeline -----------------------------------------------

_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
_TEMPERATURE_RE = re.compile(r"(-?\d+)\s?°\s?C?")
_ORDINAL_RE = re.compile(r"\b(\d+)-(?=\S)")
_DECIMAL_RE = re.compile(r"(-?\d+)\.(\d+)\b")
_INTEGER_RE = re.compile(r"-?\d+")


def convert_numbers_in_text(text: str) -> str:
    """Replace every recognized numeric pattern in ``text`` with Uzbek words.

    Order matters: date/time/percent/temperature/ordinal are matched (and
    their digits consumed) before the generic decimal/integer patterns, so
    e.g. "27.09.2026" is read as one date rather than three plain numbers.
    """
    text = _DATE_RE.sub(
        lambda m: date_to_words(int(m.group(1)), int(m.group(2)), int(m.group(3))), text
    )
    text = _TIME_RE.sub(lambda m: time_to_words(int(m.group(1)), int(m.group(2))), text)

    def _percent(m: re.Match[str]) -> str:
        int_part, _, frac = m.group(1).partition(".")
        if frac:
            return f"{decimal_to_words(int(int_part), frac)} foiz"
        return percent_to_words(int(int_part))

    text = _PERCENT_RE.sub(_percent, text)
    text = _TEMPERATURE_RE.sub(lambda m: temperature_to_words(int(m.group(1))), text)
    # The hyphen is consumed with no replacement, so the ordinal word needs an
    # explicit trailing space to stay separated from the word that follows it
    # directly (e.g. "5-sinf" -> "beshinchi sinf", not "beshinchisinf").
    text = _ORDINAL_RE.sub(lambda m: ordinal_to_words(int(m.group(1))) + " ", text)
    text = _DECIMAL_RE.sub(lambda m: decimal_to_words(int(m.group(1)), m.group(2)), text)
    text = _INTEGER_RE.sub(lambda m: number_to_words(int(m.group(0))), text)
    return text
