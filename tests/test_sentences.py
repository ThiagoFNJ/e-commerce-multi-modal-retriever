from emmr.retriever.sentences import dedup_product, review_to_rows, split_sentences


def test_split_basic():
    rows = split_sentences("Great boots for hiking! The sole wore out in a month. Meh.")
    assert rows == ["Great boots for hiking", "The sole wore out in a month"]  # "Meh." < floor


def test_split_whitespace_and_quotes():
    rows = split_sentences('She said "runs small." Order a size up, really.')
    assert "Order a size up, really" in rows


def test_floor_drops_fragments():
    assert split_sentences("Love it!!! Five stars.") == []


def test_empty_and_none():
    assert split_sentences("") == []


def test_lang_filter_drops_non_english():
    rows = review_to_rows("Muy comodo y el material se siente de buena calidad.")
    assert rows == []


def test_lang_filter_keeps_english():
    rows = review_to_rows("The arch support is excellent for long walks.")
    assert len(rows) == 1


def test_dedup_within_product():
    assert dedup_product(["Great sound quality", "great sound quality", "Bass is weak"]) == [
        "Great sound quality", "Bass is weak"]
