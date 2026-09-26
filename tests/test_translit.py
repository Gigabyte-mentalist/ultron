"""Tests for ultron.text.translit (docs/tz/03_stt.md §4)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ultron.text.translit import (
    cyrillic_ratio,
    cyrillic_to_latin,
    latin_to_cyrillic,
    normalize_apostrophes,
)

_PAIRS_PATH = Path(__file__).parent / "data" / "translit_pairs.tsv"


def _load_pairs() -> list[tuple[str, str]]:
    with _PAIRS_PATH.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    header, body = rows[0], rows[1:]
    assert header == ["cyrillic", "latin"]
    return [(cyr, lat) for cyr, lat in body]


_PAIRS = _load_pairs()


def test_at_least_forty_pairs_loaded() -> None:
    assert len(_PAIRS) >= 40


@pytest.mark.parametrize("cyr,lat", _PAIRS)
def test_cyrillic_to_latin(cyr: str, lat: str) -> None:
    assert cyrillic_to_latin(cyr) == lat


@pytest.mark.parametrize("cyr,lat", _PAIRS)
def test_latin_to_cyrillic(cyr: str, lat: str) -> None:
    assert latin_to_cyrillic(lat) == cyr


def test_apostrophe_variants_normalize_to_canonical() -> None:
    variants = ["toʻgʻri", "toʼgʼri", "to‘g‘ri", "to’g’ri", "to`g`ri"]
    for variant in variants:
        assert normalize_apostrophes(variant) == "to'g'ri"
        assert latin_to_cyrillic(variant) == "тўғри"


def test_ts_sound_is_lossy_only_in_the_reverse_direction() -> None:
    """ц has no dedicated Latin digraph reserved for it (docs/tz/03_stt.md §3.3), so the
    forward direction is well-defined but the reverse cannot recover the original ц."""
    assert cyrillic_to_latin("цирк") == "sirk"
    assert cyrillic_to_latin("милиция") == "militsiya"
    assert latin_to_cyrillic("sirk") == "сирк"  # not "цирк" -- documented limitation


def test_mixed_script_word_passes_latin_letters_through() -> None:
    assert cyrillic_to_latin("Altron ва дўст") == "Altron va do'st"


def test_cyrillic_ratio() -> None:
    assert cyrillic_ratio("salom dunyo") == 0.0
    assert cyrillic_ratio("салом дунё") == 1.0
    assert cyrillic_ratio("") == 0.0
    assert 0.0 < cyrillic_ratio("salom дунё") < 1.0
