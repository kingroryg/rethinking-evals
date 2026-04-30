"""
N-gram blocklist defense (discontinuous baseline).

Used in Section sec:reliability as the trilemma counter-example: a hard
blocklist can reduce the basin rate substantially but at the cost of utility
on benign inputs. Reports two numbers in the paper -- the basin-rate delta
on the failure grid and the MT-Bench regression -- so this defense must
expose both ``apply`` (for the MAP-Elites grid) and a way to count refusals
(for the utility evaluation).

Default blocklist is sourced from the union of HarmBench harm n-grams and
common jailbreak triggers. See ``data/blocklist_ngrams.txt`` (created
alongside the experiment runner). This defense is intentionally simple and
brittle; it exists only to instantiate the discontinuous corner of the
defense trilemma.
"""

from __future__ import annotations

from typing import Iterable, Optional

from src.defenses.base import Defense, REFUSAL_TOKEN


# A small, conservative seed list. The experiment runner can override this
# from a file via ``BlocklistDefense.from_file``.
DEFAULT_NGRAMS: tuple[str, ...] = (
    "make a bomb",
    "build a bomb",
    "synthesize ",
    "manufacture ",
    "weaponize ",
    "how to kill",
    "step-by-step instructions for harming",
    "child sexual",
    "cp ",
    "ransomware",
    "exploit code",
    "shellcode",
    "rootkit",
    "ddos attack",
    "hate speech against",
    "ethnic cleansing",
    "self-harm methods",
    "suicide method",
    "illegal drugs synthesis",
    "tax evasion scheme",
    "credit card skimmer",
    "phishing kit",
)


class BlocklistDefense(Defense):
    """Refuse outright if any blocked n-gram matches the prompt (case-insensitive)."""

    def __init__(self, ngrams: Iterable[str] = DEFAULT_NGRAMS):
        self.ngrams = tuple(ng.lower() for ng in ngrams)
        self.num_refused = 0
        self.num_seen = 0

    @classmethod
    def from_file(cls, path: str) -> "BlocklistDefense":
        with open(path) as f:
            ngrams = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return cls(ngrams)

    def apply(self, prompt: str) -> str:
        self.num_seen += 1
        lowered = prompt.lower()
        for ng in self.ngrams:
            if ng in lowered:
                self.num_refused += 1
                return REFUSAL_TOKEN
        return prompt

    def get_name(self) -> str:
        return "blocklist"

    @property
    def is_continuous(self) -> bool:
        return False

    def seed(self, seed: Optional[int]) -> None:
        # Pure string matching; no RNG.
        return None

    @property
    def refusal_rate(self) -> float:
        if self.num_seen == 0:
            return 0.0
        return self.num_refused / self.num_seen
