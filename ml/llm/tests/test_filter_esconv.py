"""Unit tests for ESConv filtering, using small fixtures shaped like the real
ESConv.json structure (verified keys: problem_type, emotion_type, situation,
dialog[].speaker/annotation.strategy/content)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filter_esconv import filter_conversations  # noqa: E402


def test_allowed_strategy_turn_is_kept():
    conversations = [
        {
            "problem_type": "job crisis",
            "emotion_type": "anxiety",
            "situation": "I'm stressed about work.",
            "dialog": [
                {"speaker": "seeker", "annotation": {}, "content": "I'm stressed about work."},
                {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "What is stressing you?"},
            ],
        }
    ]
    kept = filter_conversations(conversations)
    assert len(kept) == 1
    assert len(kept[0]["turns"]) == 2


def test_disallowed_strategy_turn_is_dropped_but_conversation_kept():
    conversations = [
        {
            "problem_type": "job crisis",
            "emotion_type": "anxiety",
            "situation": "Work stress.",
            "dialog": [
                {"speaker": "seeker", "annotation": {}, "content": "Work stress."},
                {"speaker": "supporter", "annotation": {"strategy": "Self-disclosure"}, "content": "I went through this too."},
                {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "What happened?"},
            ],
        }
    ]
    kept = filter_conversations(conversations)
    assert len(kept) == 1
    strategies = [t["strategy"] for t in kept[0]["turns"] if t["speaker"] == "supporter"]
    assert strategies == ["Question"]


def test_alcohol_abuse_problem_type_excluded_entirely():
    conversations = [
        {
            "problem_type": "Alcohol Abuse",
            "emotion_type": "shame",
            "situation": "Struggling with drinking.",
            "dialog": [{"speaker": "seeker", "annotation": {}, "content": "Struggling with drinking."}],
        }
    ]
    kept = filter_conversations(conversations)
    assert len(kept) == 0


def test_unsafe_keyword_in_any_turn_drops_whole_conversation():
    conversations = [
        {
            "problem_type": "job crisis",
            "emotion_type": "anxiety",
            "situation": "Work stress.",
            "dialog": [
                {"speaker": "seeker", "annotation": {}, "content": "Work stress."},
                {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "What happened?"},
                {"speaker": "seeker", "annotation": {}, "content": "I was diagnosed with depression."},
            ],
        }
    ]
    kept = filter_conversations(conversations)
    assert len(kept) == 0


def test_conversation_with_only_disallowed_turns_is_dropped():
    conversations = [
        {
            "problem_type": "job crisis",
            "emotion_type": "anxiety",
            "situation": "Work stress.",
            "dialog": [
                {"speaker": "supporter", "annotation": {"strategy": "Self-disclosure"}, "content": "Me too."},
                {"speaker": "supporter", "annotation": {"strategy": "Others"}, "content": "Hmm."},
            ],
        }
    ]
    kept = filter_conversations(conversations)
    assert len(kept) == 0
