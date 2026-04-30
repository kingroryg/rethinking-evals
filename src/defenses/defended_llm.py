"""
Compose a ``Defense`` with a ``TargetLLM``.

The defense is applied to the prompt before generation. If the defense
returns the refusal sentinel, ``DefendedLLM`` short-circuits the underlying
model and returns a fixed refusal string so the judge committee receives a
well-formed response.
"""

from __future__ import annotations

from typing import Optional

from src.defenses.base import Defense, REFUSAL_TOKEN
from src.models.target_llm import TargetLLM


REFUSAL_RESPONSE = "I cannot help with that request."


class DefendedLLM(TargetLLM):
    """Wraps a ``TargetLLM`` so every ``generate`` call goes through ``defense.apply`` first."""

    def __init__(
        self,
        inner: TargetLLM,
        defense: Defense,
        record_transcript: bool = True,
    ):
        """
        Args:
            inner: The underlying target model (e.g., ``LocalLLM`` for
                Llama-3-8B).
            defense: A ``Defense`` instance that will transform every
                incoming prompt.
            record_transcript: If True, every (raw_prompt, defended_prompt,
                response) triple is buffered in ``self.transcript``. Used by
                the Lipschitz estimator and per-cell persistence analysis.
        """
        self.inner = inner
        self.defense = defense
        self.record_transcript = record_transcript
        self.transcript: list[tuple[str, str, str]] = []

    def generate(self, prompt: str, **kwargs) -> str:
        defended_prompt = self.defense.apply(prompt)
        if defended_prompt == REFUSAL_TOKEN:
            response = REFUSAL_RESPONSE
        else:
            response = self.inner.generate(defended_prompt, **kwargs)
        if self.record_transcript:
            self.transcript.append((prompt, defended_prompt, response))
        return response

    def get_model_name(self) -> str:
        return f"{self.inner.get_model_name()}+{self.defense.get_name()}"

    def seed(self, seed: Optional[int]) -> None:
        """Forward seeding to the defense (target LLM seeding is handled by torch globally)."""
        self.defense.seed(seed)

    def reset_transcript(self) -> None:
        self.transcript.clear()
