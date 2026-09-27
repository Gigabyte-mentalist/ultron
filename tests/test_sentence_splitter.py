"""Tests for ultron.llm.sentence_splitter.SentenceSplitter (docs/tz/06_llm.md §3.6, §4)."""

from __future__ import annotations

from ultron.llm.sentence_splitter import SentenceSplitter


def _feed_all(splitter: SentenceSplitter, deltas: list[str]) -> list[str]:
    sentences: list[str] = []
    for delta in deltas:
        sentences.extend(splitter.feed(delta))
    return sentences


def test_simple_sentence_emitted_on_period_and_space() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["Salom. ", "Qandaysiz?"])

    assert sentences == ["Salom."]
    assert splitter.flush() == "Qandaysiz?"


def test_does_not_split_on_decimal_number() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["Natija 5.5 ", "edi. ", "Boshqa gap."])

    assert sentences == ["Natija 5.5 edi."]
    assert splitter.flush() == "Boshqa gap."


def test_does_not_split_on_known_abbreviation() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["Kitoblar, daftarlar va h.k. ", "kerak bo'ladi. ", "Xo'p."])

    assert sentences == ["Kitoblar, daftarlar va h.k. kerak bo'ladi."]
    assert splitter.flush() == "Xo'p."


def test_decimal_split_across_feed_calls_not_mistaken_for_sentence_end() -> None:
    splitter = SentenceSplitter()

    # "5" then ".5 " lands in a separate delta from a real streaming split --
    # must not be treated as a sentence boundary just because the buffer
    # happened to end right after the "." at some point.
    sentences = _feed_all(splitter, ["Natija 5", ".5 ", "edi."])

    assert sentences == []
    assert splitter.flush() == "Natija 5.5 edi."


def test_early_exit_on_comma_after_sixty_chars() -> None:
    splitter = SentenceSplitter()
    long_clause = "a" * 55 + " juda uzoq gap boshlandi, va u davom etmoqda"

    sentences = _feed_all(splitter, [long_clause])

    assert len(sentences) == 1
    assert sentences[0].endswith(",")
    assert len(sentences[0]) >= 60


def test_no_early_exit_below_sixty_chars() -> None:
    splitter = SentenceSplitter()

    # a comma is present, but the buffer is well under 60 chars and has no
    # terminal punctuation yet -- must wait for more input, not early-exit.
    sentences = splitter.feed("Qisqa, gap hali tugamagan")

    assert sentences == []
    assert splitter.flush() == "Qisqa, gap hali tugamagan"


def test_early_exit_only_applies_to_first_sentence() -> None:
    splitter = SentenceSplitter()
    # fed in two deltas, like real streaming: the comma-bearing clause arrives
    # with no terminal punctuation yet, *then* the period streams in later.
    before_period = "a" * 65 + ", davomi bor"
    second_long_clause = "b" * 65 + ", lekin bu ikkinchi gap uchun early exit yo'q"

    first_sentences = _feed_all(splitter, [before_period])
    trailing_sentences = _feed_all(splitter, [". "])
    second_sentences = _feed_all(splitter, [second_long_clause])

    assert first_sentences == ["a" * 65 + ","]  # early exit fired at the comma
    assert trailing_sentences == ["davomi bor."]  # the rest follows normally
    assert second_sentences == []  # long + has a comma, but the early-exit budget is already used
    assert splitter.flush() == second_long_clause


def test_think_block_is_stripped_even_split_across_feeds() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(
        splitter, ["<thi", "nk>ichida o'ylayapman</think>", "Tashqi javob. ", "Yana."]
    )

    assert sentences == ["Tashqi javob."]
    assert splitter.flush() == "Yana."
    assert "<think>" not in "".join(sentences)


def test_unclosed_think_block_holds_back_text_and_is_dropped_on_flush() -> None:
    splitter = SentenceSplitter()

    sentences = splitter.feed("Salom. <think>hali tugamadi")

    assert sentences == ["Salom."]
    # the stream ended mid-thought -- flush() drops the unclosed <think> content
    # rather than leaking raw internal reasoning to the user.
    assert splitter.flush() is None


def test_markdown_characters_are_stripped() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["**Bajarildi**. ", "`kod` emas."])

    assert sentences == ["Bajarildi."]
    assert splitter.flush() == "kod emas."


def test_emoji_are_stripped() -> None:
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["Bajarildi ✅. ", "Yana bir gap."])

    assert sentences == ["Bajarildi ."]
    assert splitter.flush() == "Yana bir gap."


def test_pure_emoji_sentence_is_dropped_not_emitted_empty() -> None:
    splitter = SentenceSplitter()

    # a whole "sentence" that's only an emoji must not surface as an empty
    # SentenceEvent -- it's simply consumed, and the real text after it
    # still comes through normally.
    sentences = _feed_all(splitter, ["\U0001f324️. ", "Haqiqiy gap."])

    assert sentences == []
    assert splitter.flush() == "Haqiqiy gap."


def test_lone_asterisk_used_as_multiplication_is_preserved() -> None:
    # a single, unpaired "*" is left alone -- Qwen sometimes writes it as a
    # multiplication sign, and blanket-stripping it garbles the number.
    splitter = SentenceSplitter()

    sentences = _feed_all(splitter, ["17 * 23 = 391. "])

    assert sentences == ["17 * 23 = 391."]


def test_flush_returns_none_for_empty_buffer() -> None:
    splitter = SentenceSplitter()

    assert splitter.flush() is None
