import pytest

from recsys.data.align import clean_title, extract_year, is_fuzzy_match, title_keys


@pytest.mark.parametrize("raw, keys", [
    ("Matrix, The (1999)", ["matrix"]),
    ("Postino, Il (The Postman) (1994)", ["postino", "postman"]),
    ("Hate (Haine, La) (1995)", ["hate", "haine"]),
    ("Seven (Se7en) (1995)", ["seven", "se7en"]),
    ("Karate Kid III, The (1989)", ["karate kid 3"]),
    ("Strike! (a.k.a. All I Wanna Do, The Hairy Bird) (1998)", ["strike", "all i wanna do the hairy bird"]),
    ("Star Maker, The (Uomo delle stelle, L') (1995)", ["star maker", "uomo delle stelle"]),
])
def test_title_keys(raw, keys):
    assert title_keys(raw) == keys


def test_clean_title_matches_cmu_forms():
    assert clean_title("The Matrix") == title_keys("Matrix, The (1999)")[0]
    assert clean_title("Les Misérables") == clean_title("Misérables, Les")
    assert clean_title("Mission: Impossible II") == clean_title("Mission: Impossible 2")
    assert clean_title("8½ Women") == clean_title("8 1/2 Women")


def test_extract_year_uses_last_parenthesised_year():
    assert extract_year("Seven (Se7en) (1995)") == 1995
    assert extract_year("1984 (1984)") == 1984
    assert extract_year("No year") is None


@pytest.mark.parametrize("a, b, ok", [
    ("two moon juction", "two moon junction", True),
    ("texas chainsaw massacre", "texas chain saw massacre", True),
    ("sleepover", "sleepers", False),
    ("godzilla 2000", "godzilla", False),
    ("mafia", "mafia raaj", False),
])
def test_fuzzy_guard(a, b, ok):
    assert is_fuzzy_match(a, b) is ok
