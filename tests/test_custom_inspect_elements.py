import pytest
from ethics_eval.custom_inspect_elements import match_str


@pytest.mark.parametrize(
    "value,target,location,ignore_case,ignore_punctuation,numeric,edit_distance,expected",
    [
        # Basic matching
        ("Hello, World!", "World", "end", True, True, False, 0, ("Hello, World!", True)),
        ("Hello, World!", "hello", "begin", True, True, False, 0, ("Hello, World!", True)),
        ("Hello, World!", "lo, Wo", "any", True, True, False, 0, ("Hello, World!", True)),
        ("Hello, World!", "Hello, World!", "exact", True, True, False, 0, ("Hello, World!", True)),
        
        # Case sensitivity
        ("Hello, World!", "world", "end", False, True, False, 0, ("Hello, World!", False)),
        ("Hello, World!", "world", "end", True, True, False, 0, ("Hello, World!", True)),
        
        # Punctuation
        ("Hello, World!", "World!", "end", True, False, False, 0, ("Hello, World!", True)),
        ("Hello, World!", "World", "end", True, True, False, 0, ("Hello, World!", True)),
        
        # Numeric matching
        ("The answer is 42.", "42", "end", True, True, True, 0, ("42", True)),
        ("The price is $99.99", "99.99", "end", True, True, True, 0, ("99.99", True)),
        ("Temperature: 23.5°C", "23.5", "any", True, True, True, 0, ("23.5", True)),
        
        # Edit distance
        ("Color", "Colour", "exact", True, True, False, 1, ("Color", True)),
        ("Approximately 100", "101", "end", True, True, True, 1, ("100", True)),
        
        # Edge cases
        ("", "", "exact", True, True, False, 0, ("", True)),
        ("No match here", "xyz", "any", True, True, False, 0, ("No match here", False)),

        ("Color", "Colour", "exact", True, True, False, 1, ("Color", True)),
        ("Behavior", "Behaviour", "exact", True, True, False, 1, ("Behavior", True)),
        ("Approximately 100", "101", "end", True, True, True, 1, ("100", True)),
        ("The answer is 42", "43", "end", True, True, True, 1, ("42", True)),
        ("Hello, World!", "Hello, Word!", "exact", True, True, False, 1, ("Hello, World!", True)),
        ("OpenAI", "OpenAl", "exact", True, True, False, 1, ("OpenAI", True)),
        ("Python 3.9", "Python 3.10", "begin", True, True, False, 2, ("Python 3.9", True)),

        # Cases where expected[1] is False
        ("United States", "STates", "end", True, True, False, 1, ("United States", True)),
        ("United States", "Statis", "end", True, True, False, 1, ("United States", True)),
        ("United States", "Stater", "end", True, True, False, 1, ("United States", True)),
        ("United States", "State", "end", False, True, False, 1, ("United States", False)),
        ("United States", "STates", "end", True, True, False, 0, ("United States", True)),
        ("United States", "STates", "end", False, True, False, 1, ("United States", True)),
        ("Python", "Java", "exact", True, True, False, 0, ("Python", False)),
        ("OpenAI", "Google", "begin", True, True, False, 0, ("OpenAI", False)),
        ("The answer is 42", "24", "end", True, True, True, 0, ("42", False)),
        ("Machine Learning", "Deep Learning", "exact", True, True, False, 2, ("Machine Learning", False)),
        ("Artificial Intelligence", "Natural Language Processing", "any", True, True, False, 3, ("Artificial Intelligence", False)),
        ("The price is $99.99", "100", "end", True, True, True, 1, ("99.99", False)),
        ("Python 3.9", "Python 4.0", "exact", True, True, False, 1, ("Python 3.9", False)),
        ("Hello, World!", "Hello, Universe!", "exact", True, True, False, 2, ("Hello, World!", False)),
        ("OpenAI GPT", "Google BERT", "begin", True, True, False, 3, ("OpenAI GPT", False)),

        # Edge cases
        ("", "Something", "any", True, True, False, 0, ("", False)),
        # ("Something", "", "any", True, True, False, 0, ("Something", False)),
        ("A", "B", "exact", True, True, False, 0, ("A", False)),
        ("A", "B", "exact", True, True, False, 1, ("A", True)),
    ]
)
def test_match_str(
    value: str,
    target: str,
    location: str,
    ignore_case: bool,
    ignore_punctuation: bool,
    numeric: bool,
    edit_distance: int,
    expected: tuple[str, bool],
):
    result = match_str(
        value,
        target,
        location,
        ignore_case,
        ignore_punctuation,
        numeric,
        edit_distance,
    )
    assert result == expected


def test_match_str_numeric_normalization():
    # Test numeric normalization
    assert match_str("The answer is 42.000", "42", "end", True, True, True, 0) == ("42", True)
    assert match_str("The answer is 42.100", "42.1", "end", True, True, True, 0) == ("42.1", True)
    assert match_str("The answer is 0.0001", "1e-4", "end", True, True, True, 0) == ("0.0001", True)


def test_match_str_currency():
    # Test currency handling
    assert match_str("The price is $99.99", "99.99", "end", True, True, True, 0) == ("99.99", True)
    assert match_str("The price is €99,99", "99.99", "end", True, True, True, 0) == ("99.99", True)
    assert match_str("The price is £1,234.56", "1234.56", "end", True, True, True, 0) == ("1234.56", True)


def test_match_str_location_specific():
    # Test location-specific behavior
    assert match_str("Start 42 Middle 43 End", "42", "begin", True, True, True, 0) == ("42", True)
    assert match_str("Start 42 Middle 43 End", "43", "end", True, True, True, 0) == ("43", True)
    assert match_str("Start 42 Middle 43 End", "42", "end", True, True, True, 0) == ("43", False)


def test_match_str_edit_distance():
    # Test edit distance
    assert match_str("The answer is 42", "41", "end", True, True, True, 1) == ("42", True)
    assert match_str("The answer is 42", "40", "end", True, True, True, 1) == ("42", False)
    assert match_str("Color", "Colour", "exact", True, True, False, 1) == ("Color", True)
    assert match_str("Color", "Coulour", "exact", True, True, False, 1) == ("Color", False)


def test_match_str_ignore_case():
    # Test case sensitivity
    assert match_str("The Answer Is 42", "answer", "any", True, True, False, 0) == ("The Answer Is 42", True)
    assert match_str("The Answer Is 42", "answer", "any", False, True, False, 0) == ("The Answer Is 42", False)


def test_match_str_ignore_punctuation():
    # Test punctuation handling
    assert match_str("Hello, World!", "Hello World", "exact", True, True, False, 0) == ("Hello, World!", True)
    assert match_str("Hello, World!", "Hello World", "exact", True, False, False, 0) == ("Hello, World!", False)
