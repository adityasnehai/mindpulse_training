"""Unit tests for EmpatheticDialogues filtering, using small fixtures shaped
like the real CSV format (verified columns: conv_id, utterance_idx, context,
prompt, speaker_idx, utterance, selfeval, tags)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filter_empathetic_dialogues import filter_conversations, unescape  # noqa: E402


def test_unescape_replaces_comma_marker():
    assert unescape("a lot of people_comma_ but only us") == "a lot of people, but only us"


def test_unescape_handles_non_string():
    assert unescape(None) is None


def _row(conv_id, idx, context, prompt, speaker, utterance):
    return {
        "conv_id": conv_id, "utterance_idx": idx, "context": context,
        "prompt": prompt, "speaker_idx": speaker, "utterance": utterance,
        "selfeval": "5|5|5", "tags": None,
    }


def test_allowed_context_is_kept():
    df = pd.DataFrame([
        _row("c1", 1, "annoyed", "My coworker keeps interrupting me.", 0, "My coworker keeps interrupting me."),
        _row("c1", 2, "annoyed", "My coworker keeps interrupting me.", 1, "That sounds frustrating, what happened?"),
    ])
    kept = filter_conversations(df)
    assert len(kept) == 1
    assert kept[0]["context"] == "annoyed"
    assert len(kept[0]["turns"]) == 2


def test_disallowed_context_is_excluded():
    df = pd.DataFrame([
        _row("c2", 1, "terrified", "Something scary happened.", 0, "Something scary happened."),
        _row("c2", 2, "terrified", "Something scary happened.", 1, "That sounds frightening."),
    ])
    kept = filter_conversations(df)
    assert len(kept) == 0


def test_allowed_context_but_unsafe_keyword_is_excluded():
    df = pd.DataFrame([
        _row("c3", 1, "content", "Feeling okay today.", 0, "Feeling okay today."),
        _row("c3", 2, "content", "Feeling okay today.", 1, "I was diagnosed with something once."),
    ])
    kept = filter_conversations(df)
    assert len(kept) == 0


def test_multiple_conversations_grouped_independently():
    df = pd.DataFrame([
        _row("c1", 1, "annoyed", "situation A", 0, "utterance A1"),
        _row("c1", 2, "annoyed", "situation A", 1, "utterance A2"),
        _row("c4", 1, "joyful", "situation B", 0, "utterance B1"),
    ])
    kept = filter_conversations(df)
    assert len(kept) == 2
    conv_ids = {k["conv_id"] for k in kept}
    assert conv_ids == {"c1", "c4"}
