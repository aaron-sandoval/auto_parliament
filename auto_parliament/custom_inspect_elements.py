import re
import string
from typing import Literal, Callable

import Levenshtein
from inspect_ai.scorer import (
    scorer,
    Scorer,
    Score,
    Target,
    accuracy,
    stderr,
)
from inspect_ai.solver import TaskState


def first_number_normalized(words: list[str]) -> str:
    number = next(
        (word for word in words if word.replace(".", "").isnumeric()), words[0]
    )
    return normalize_number(number)


def normalize_number(number: str, precision: int = 5) -> str:
    if number.replace(".", "").isnumeric():
        num = float(number)
        return format(num, f".{precision}g")
    else:
        return number
    
def strip_punctuation(s: str) -> str:
    return s.strip(string.whitespace + string.punctuation)


def strip_numeric_punctuation(s: str) -> str:
    # strip $, €, £, and ,
    stripped = re.sub(r"[$,£,€]", "", s)
    # strip . if it's followed by a space, the end of the string,
    # or a non-digit character
    stripped = re.sub(r"\.(?=\s|$|\D)", "", stripped)
    return stripped

def match_str(
    value: str,
    target: str,
    location: Literal["begin", "end", "any", "exact"] = "end",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    numeric: bool = False,
    edit_distance: int = 0,
) -> tuple[str, bool]:
    """
    Match a value to a target string, allowing for edit distance.
    """
    # strip ws
    v = value.strip()
    t = target.strip()

    # baseline answer (will only change for numeric)
    answer = v

    # further cleanup
    if ignore_case:
        v = v.lower()
        t = t.lower()
    if numeric:
        # remove punctuation
        v = strip_numeric_punctuation(v)
        t = strip_numeric_punctuation(t)
        # normalize as required
        t = normalize_number(t)
        if location == "begin":
            words = v.split(" ")
            v = first_number_normalized(words)
        elif location == "end":
            words = v.split(" ")
            words.reverse()
            v = first_number_normalized(words)
        elif location == "exact":
            v = normalize_number(v)
        answer = v
    elif ignore_punctuation:
        v = strip_punctuation(v)
        t = strip_punctuation(t)

    # comparisons
    if location == "begin":
        exact_match = v.startswith(t)
        if not exact_match and edit_distance > 0:
            distance = Levenshtein.distance(v[:len(t)], t)
            return answer, distance <= edit_distance
    elif location == "end":
        exact_match = v.endswith(t)
        if not exact_match and edit_distance > 0:
            distance = Levenshtein.distance(v[-len(t):], t)
            return answer, distance <= edit_distance
    elif location == "exact":
        exact_match = v == t
        if not exact_match and edit_distance > 0:
            distance = Levenshtein.distance(v, t)
            return answer, distance <= edit_distance
    else:  # "any"
        exact_match = t in v
        if not exact_match and edit_distance > 0:
            # Find the substring with the smallest edit distance
            min_distance = float('inf')
            for i in range(len(v) - len(t) + 1):
                substring = v[i:i+len(t)]
                distance = Levenshtein.distance(substring, t)
                if distance < min_distance:
                    min_distance = distance
            return answer, min_distance <= edit_distance

    return answer, exact_match


def str_match_scorer(match: Callable[[str, str], tuple[str, bool]]) -> Scorer:
    """Scorer that uses a matching function.

    The matching function returns tuple[str,bool], where str is the answer
    extracted from the model output and bool is whether it matched the target
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer: str | None = None
        for value in target:
            answer, matched = match(state.output.completion, value)
            if matched:
                return Score(
                    value="C", answer=answer, explanation=state.output.completion
                )

        return Score(
            value="I", answer=answer, explanation=state.output.completion
        )

    return score

@scorer(metrics=[accuracy(), stderr()])
def match_with_edit_distance(
    location: Literal["begin", "end", "any", "exact"] = "end",
    *,
    ignore_case: bool = True,
    numeric: bool = False,
    edit_distance: int = 0,
) -> Scorer:
    """Scorer which matches text or a number, allowing for edit distance.

    Args:
       location (Literal["begin", "end", "any", "exact"]):
          Location to match at. "any" matches anywhere in the
          output; "exact" requires the output be exactly
          equal to the target (module whitespace, etc.)
       ignore_case (bool): Do case insensitive comparison.
       numeric (bool): Is this a numeric match? (in this
          case different punctuation removal rules are
          used and numbers are normalized before comparison).
       edit_distance (int): Maximum allowed edit distance between
          the target and the matched text.
    """

    def check(value: str, target: str) -> tuple[str, bool]:
        matched_text, is_match = match_str(
            value=value,
            target=target,
            location=location,
            ignore_case=ignore_case,
            numeric=numeric,
            edit_distance=edit_distance
        )
        return matched_text, is_match

    return str_match_scorer(check)

# ... rest of the file ...
