"""Tests for ultron.tts.preprocess (docs/tz/05_tts.md §4)."""

from __future__ import annotations

from ultron.tts.preprocess import clean_text, split_sentences

# -- step 1: wake-word removal ------------------------------------------------


def test_wake_word_removed_standalone() -> None:
    assert clean_text("Ultron, brauzerni och.", model_script="latin") == "brauzerni och."


def test_wake_word_removed_midsentence_with_surrounding_commas() -> None:
    assert (
        clean_text("Salom, men Altron emasman, oddiy yordamchiman.", model_script="latin")
        == "Salom, men emasman, oddiy yordamchiman."
    )


def test_wake_word_removed_case_insensitive_and_variants() -> None:
    for variant in ("ultron", "ULTRON", "Oltron", "ultiron", "Altiron"):
        result = clean_text(f"Bu {variant} ovozi.", model_script="latin")
        assert "ltron" not in result.lower()
        assert "ltiron" not in result.lower()


def test_wake_word_removed_even_with_uzbek_suffix_attached() -> None:
    """Uzbek suffixes attach with no space ("Ultronman" = "I am Ultron")."""
    result = clean_text("Men Ultronman.", model_script="latin")
    assert "ltron" not in result.lower()


def test_output_never_contains_wake_word() -> None:
    for text in (
        "Assalomu alaykum, men Ultron.",
        "ULTRON hozir band.",
        "Bu Altronning ovozi edi.",
    ):
        result = clean_text(text, model_script="latin")
        assert "ultron" not in result.lower()
        assert "altron" not in result.lower()


# -- step 2: markdown / emoji / url cleanup -----------------------------------


def test_markdown_emphasis_and_heading_chars_stripped() -> None:
    assert clean_text("**Muhim**: # sarlavha", model_script="latin") == "Muhim: sarlavha"


def test_markdown_link_keeps_label_drops_url() -> None:
    assert (
        clean_text("Batafsil: [shu yerda](https://example.com/x)", model_script="latin")
        == "Batafsil: shu yerda"
    )


def test_bare_url_replaced_with_havola() -> None:
    result = clean_text("Ko'ring: https://example.com/uzun-yol sahifasini.", model_script="latin")
    assert "https" not in result
    assert "havola" in result


def test_emoji_stripped() -> None:
    assert clean_text("Ajoyib ish! \U0001f389", model_script="latin") == "Ajoyib ish!"


# -- step 3: numbers ----------------------------------------------------------


def test_numbers_converted_to_words() -> None:
    assert (
        clean_text("Soat 09:30, harorat -3°C.", model_script="latin")
        == "Soat to'qqiz o'ttiz, harorat minus uch daraja."
    )


# -- step 4: abbreviations / pronunciation dictionary -------------------------


def test_abbreviations_expanded() -> None:
    result = clean_text("Fayl hajmi 5 GB, CPU yuklamasi 50%.", model_script="latin")
    assert "gigabayt" in result
    assert "protsessor" in result.lower()
    assert "GB" not in result
    assert "CPU" not in result


def test_app_names_pronounced() -> None:
    result = clean_text("YouTube va Chrome ochilmoqda.", model_script="latin")
    assert "Yutub" in result
    assert "Xrom" in result


# -- step 5: latin -> cyrillic transliteration --------------------------------


def test_cyrillic_conversion_applied_when_model_script_is_cyrillic() -> None:
    assert clean_text("Salom dunyo", model_script="cyrillic") == "Салом дунё"


def test_no_cyrillic_conversion_when_model_script_is_latin() -> None:
    assert clean_text("Salom dunyo", model_script="latin") == "Salom dunyo"


def test_numbers_and_translit_compose() -> None:
    assert clean_text("123", model_script="cyrillic") == "юз йигирма уч"


# -- misc ----------------------------------------------------------------


def test_empty_input_returns_empty_string() -> None:
    assert clean_text("", model_script="latin") == ""
    assert clean_text("Ultron", model_script="latin") == ""


# -- step 6: sentence splitting ------------------------------------------------


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_basic() -> None:
    assert split_sentences(
        "Assalomu alaykum, qandaysiz? Zo'r ishlayapman, rahmat. Sizga qanday yordam bera olaman?"
    ) == [
        "Assalomu alaykum, qandaysiz?",
        "Zo'r ishlayapman, rahmat.",
        "Sizga qanday yordam bera olaman?",
    ]


def test_split_sentences_single_sentence_untouched() -> None:
    assert split_sentences("Bugun ob-havo juda yaxshi.") == ["Bugun ob-havo juda yaxshi."]


def test_short_fragment_merges_forward() -> None:
    assert split_sentences("Ha. Bajarildi, hammasi tayyor.") == ["Ha. Bajarildi, hammasi tayyor."]


def test_short_fragment_merges_backward_when_last() -> None:
    assert split_sentences("Bugun ob-havo juda yaxshi va issiq. Ha.") == [
        "Bugun ob-havo juda yaxshi va issiq. Ha."
    ]


def test_long_sentence_split_on_commas() -> None:
    long_sentence = "Bu juda uzun gap, " * 15 + "oxiri."
    result = split_sentences(long_sentence)
    assert len(result) > 1
    assert all(len(part) <= 200 for part in result)
    assert "".join(result).replace(" ", "") == long_sentence.replace(" ", "")
