from typing import Literal
import Levenshtein

# ... existing imports ...

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
        matched_text, exact_match = match_str(
            value=value,
            target=target,
            location=location,
            ignore_case=ignore_case,
            numeric=numeric,
        )
        
        if exact_match:
            return matched_text, True
        
        if edit_distance > 0:
            distance = Levenshtein.distance(matched_text, target)
            return matched_text, distance <= edit_distance
        
        return matched_text, False

    return str_match_scorer(check)

# ... rest of the file ...
