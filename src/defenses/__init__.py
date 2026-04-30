"""
Continuous and discontinuous input defenses for LLM safety evaluation.

This package implements the four defenses evaluated in Section "Basin Persistence
Under Continuous Defenses" of the paper:

  * PerplexityFilterDefense  -- continuous: high-perplexity prompts are
    rephrased by a small rephraser before reaching the target.
  * ParaphraseWrapperDefense -- continuous: every prompt is paraphrased.
  * ConstitutionalRewriterDefense -- continuous: an instruction-tuned model
    rewrites each prompt to remove stated harmful intent while preserving
    semantics.
  * BlocklistDefense -- discontinuous: prompts containing flagged n-grams are
    refused outright. Used only as the trilemma baseline.

All defenses implement the ``Defense`` interface and are composed with a
``TargetLLM`` via ``DefendedLLM``, so the rest of the MAP-Elites pipeline is
unchanged.
"""

from .base import Defense
from .defended_llm import DefendedLLM
from .perplexity_filter import PerplexityFilterDefense
from .paraphrase_wrapper import ParaphraseWrapperDefense
from .constitutional_rewriter import ConstitutionalRewriterDefense
from .blocklist import BlocklistDefense
from .factory import create_defense

__all__ = [
    "Defense",
    "DefendedLLM",
    "PerplexityFilterDefense",
    "ParaphraseWrapperDefense",
    "ConstitutionalRewriterDefense",
    "BlocklistDefense",
    "create_defense",
]
