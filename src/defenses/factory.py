"""
Factory for constructing defenses from a YAML config block.

The defended experiment runner reads ``config/defenses.yaml`` and dispatches
to one of the four defense classes here. Keeping the dispatch in one place
means the runner can be agnostic to which defense is loaded.
"""

from __future__ import annotations

from typing import Any, Dict

from src.defenses.base import Defense
from src.defenses.blocklist import BlocklistDefense
from src.defenses.constitutional_rewriter import ConstitutionalRewriterDefense
from src.defenses.paraphrase_wrapper import ParaphraseWrapperDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.models.target_llm import create_target_llm


def create_defense(name: str, defenses_config: Dict[str, Any]) -> Defense:
    """Instantiate a defense by name from the parsed YAML config.

    Args:
        name: One of {perplexity_filter, paraphrase, constitutional, blocklist}.
        defenses_config: The parsed contents of ``config/defenses.yaml``.

    Returns:
        A ready-to-use ``Defense`` instance.
    """
    if name not in defenses_config:
        raise KeyError(f"Defense {name!r} not in defenses.yaml. Have: {list(defenses_config)}")
    cfg = defenses_config[name]
    kind = cfg.get("kind", name)

    if kind == "perplexity_filter":
        rephraser = create_target_llm(cfg["rephraser"])
        return PerplexityFilterDefense(
            rephraser=rephraser,
            threshold=float(cfg.get("threshold", 100.0)),
            ppl_model_name=cfg.get("ppl_model_name", "gpt2"),
            device=cfg.get("device", "auto"),
        )

    if kind == "paraphrase":
        return ParaphraseWrapperDefense(
            model_name=cfg.get("model_name", "Vamsi/T5_Paraphrase_Paws"),
            max_length=int(cfg.get("max_length", 256)),
            num_beams=int(cfg.get("num_beams", 5)),
            device=cfg.get("device", "auto"),
        )

    if kind == "constitutional":
        rewriter = create_target_llm(cfg["rewriter"])
        return ConstitutionalRewriterDefense(rewriter=rewriter)

    if kind == "blocklist":
        path = cfg.get("ngrams_file")
        if path:
            return BlocklistDefense.from_file(path)
        return BlocklistDefense()

    raise ValueError(f"Unknown defense kind: {kind!r}")
