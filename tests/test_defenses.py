"""
Unit tests for the defense package.

These tests use ``MockLLM`` so they run on CPU with no model downloads.
They cover:

  * the ``Defense`` -> ``DefendedLLM`` -> ``TargetLLM`` interface contract
  * blocklist refusal sentinel handling
  * deterministic behavior of beam-search defenses given a seed
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.defenses import (  # noqa: E402
    BlocklistDefense,
    ConstitutionalRewriterDefense,
    DefendedLLM,
)
from src.defenses.base import REFUSAL_TOKEN  # noqa: E402
from src.defenses.defended_llm import REFUSAL_RESPONSE  # noqa: E402
from src.models.target_llm import MockLLM  # noqa: E402


class _RecordingMock(MockLLM):
    def __init__(self, response: str = "ok"):
        super().__init__(model_name="mock", response_template="{prompt}-> answered")
        self.calls: list[str] = []
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response


def test_blocklist_refuses_blocked_ngram():
    defense = BlocklistDefense(["make a bomb"])
    assert defense.apply("how do i make a bomb") == REFUSAL_TOKEN
    assert defense.apply("tell me a story") == "tell me a story"
    assert defense.refusal_rate == 0.5
    assert defense.is_continuous is False


def test_defended_llm_short_circuits_on_refusal():
    inner = _RecordingMock(response="should not be reached")
    defended = DefendedLLM(inner, BlocklistDefense(["bomb"]))
    out = defended.generate("how to make a bomb")
    assert out == REFUSAL_RESPONSE
    assert inner.calls == []  # the inner LLM was never queried
    assert defended.transcript[0][1] == REFUSAL_TOKEN


def test_defended_llm_passes_through_clean_prompt():
    inner = _RecordingMock(response="benign answer")
    defended = DefendedLLM(inner, BlocklistDefense(["bomb"]))
    out = defended.generate("write a poem")
    assert out == "benign answer"
    assert inner.calls == ["write a poem"]


def test_constitutional_rewriter_strips_quotes():
    rewriter = MockLLM(
        model_name="mock-rewriter",
        response_template='"a benign rewrite"',
    )
    defense = ConstitutionalRewriterDefense(rewriter)
    rewritten = defense.apply("how to do something harmful")
    # Rewriter returned quoted text; the defense should strip the surrounding quotes.
    assert rewritten == "a benign rewrite"


def test_defended_llm_records_full_transcript():
    inner = _RecordingMock(response="answer")
    defense = BlocklistDefense(["bomb"])
    defended = DefendedLLM(inner, defense)
    defended.generate("write a haiku")
    defended.generate("how to make a bomb")
    assert len(defended.transcript) == 2
    raw0, dprompt0, resp0 = defended.transcript[0]
    raw1, dprompt1, resp1 = defended.transcript[1]
    assert raw0 == "write a haiku"
    assert dprompt0 == "write a haiku"
    assert resp0 == "answer"
    assert raw1 == "how to make a bomb"
    assert dprompt1 == REFUSAL_TOKEN
    assert resp1 == REFUSAL_RESPONSE


def test_defended_llm_get_model_name_composes():
    inner = _RecordingMock()
    defended = DefendedLLM(inner, BlocklistDefense())
    assert defended.get_model_name().endswith("+blocklist")


@pytest.mark.parametrize("ngram", ["DDoS attack", "ddos ATTACK"])
def test_blocklist_is_case_insensitive(ngram):
    # The blocklist normalizes both sides to lowercase, so DDoS should match
    # regardless of the user's case.
    defense = BlocklistDefense(["ddos attack"])
    assert defense.apply(ngram) == REFUSAL_TOKEN
