"""
Abstract interface for input defenses.

A Defense is a (possibly stochastic) self-map on the prompt space. Concretely
it consumes a prompt string and returns a (possibly identical) prompt string
that is then forwarded to the target LLM. The wider system treats the
target_llm + defense composition as a single ``DefendedLLM`` (see
``defended_llm.py``), so that the rest of the MAP-Elites pipeline is
agnostic to whether a defense is installed.

The persistence theorem in Section sec:reliability is stated for continuous
defenses. ``is_continuous`` lets downstream code (Lipschitz estimation, the
trilemma plot) distinguish the n-gram blocklist (discontinuous) from the
three paraphrase-style wrappers (continuous).

Defenses are required to be deterministic given a seed so that multi-seed
MAP-Elites runs are reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


REFUSAL_TOKEN = "[REFUSED_BY_DEFENSE]"
"""Sentinel returned by discontinuous defenses when a prompt is rejected.

Downstream the ``DefendedLLM`` translates this into a fixed refusal response
('I cannot help with that.') so that the judge committee sees a coherent,
non-empty string rather than an exception.
"""


class Defense(ABC):
    """Abstract base class for input defenses."""

    @abstractmethod
    def apply(self, prompt: str) -> str:
        """Transform ``prompt`` before it reaches the target LLM.

        Args:
            prompt: Raw user prompt.

        Returns:
            The transformed prompt that should be sent to the target LLM, or
            ``REFUSAL_TOKEN`` if a discontinuous defense rejects the prompt.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Stable, human-readable identifier (e.g., ``"perplexity_filter"``)."""

    @property
    def is_continuous(self) -> bool:
        """Whether this defense is a continuous self-map.

        Continuous defenses satisfy the assumptions of the persistence
        theorem (Equation 1 in the paper). Discontinuous defenses (e.g.,
        n-gram blocklists) do not and are evaluated separately as the
        trilemma baseline.
        """
        return True

    def seed(self, seed: Optional[int]) -> None:
        """Set the per-call RNG seed (no-op for deterministic defenses).

        Subclasses that use sampling (paraphrasing, rewriting) override
        this to make their outputs reproducible across seeds.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.get_name()!r})"
