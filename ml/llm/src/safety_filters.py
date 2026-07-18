"""Shared safety-exclusion logic for both dialogue source datasets, per
docs/PRODUCT_SPEC.md sections 6.2/6.3: keyword-level defense-in-depth against
diagnosis, medication, self-harm/suicide, and crisis language slipping through
category/strategy-level filtering.
"""

import re

# Deliberately broad and conservative: false positives (dropping a borderline-
# safe example) are the acceptable failure mode here, not false negatives.
EXCLUDED_KEYWORD_PATTERNS = [
    r"\bsuicid\w*",
    r"\bself.?harm\w*",
    r"\bkill myself\b",
    r"\bkill.?ing myself\b",
    r"\bcutting myself\b",
    r"\boverdos\w*",
    r"\bmedicat\w*",
    r"\bprescri\w*",
    r"\bantidepressant\w*",
    r"\btherap\w*",  # excludes "therapist"/"therapy" recommendations, which the LLM must not make
    r"\bdiagnos\w*",
    r"\bPHQ\b",
    r"\bpsychiatr\w*",
    r"\bhospitaliz\w*",
    r"\babuse\w*",
    r"\balcoholic\w*",
    r"\bdrunk\b",
    r"\bdrinking problem\b",
    # Added after inspecting real filtered output: a kept EmpatheticDialogues
    # example's acknowledgement text referenced a friend's cancer scare —
    # not medical advice or diagnosis, but tonally mismatched with a mild,
    # routine-focused wellbeing app voice. Serious-illness/death references
    # are excluded even when not clinical advice.
    r"\bcancer\b",
    r"\bterminal(ly)? ill\w*",
    r"\bdying\b",
    r"\bdeath\b",
    r"\bdied\b",
    r"\bemergency room\b",
    r"\bICU\b",
    r"\bassault\w*",
    r"\brape\w*",
    r"\bdomestic violence\b",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_KEYWORD_PATTERNS]


def contains_excluded_content(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _COMPILED_PATTERNS)
