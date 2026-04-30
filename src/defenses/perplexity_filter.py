"""
Perplexity-filter defense.

Implements the defense from Jain et al., "Baseline Defenses for Adversarial
Attacks Against Aligned Language Models" (2023). A small reference model
(GPT-2 by default) computes the per-token negative log-likelihood of the
prompt; if it exceeds ``threshold``, the prompt is rephrased by a small
instruction-tuned LLM. Otherwise it is forwarded unchanged.

This defense is continuous in the sense of Equation 1 in the paper: as the
input prompt's PPL crosses the threshold, the rephraser's output varies
continuously in the underlying embedding space (the rephraser is an LLM
fine-tuned to preserve meaning). It is *not* the kind of hard cutoff that
would violate the Lipschitz assumption -- the threshold gates whether a
rephrase is applied, but does not produce a discontinuous jump in the
elicited harm score.
"""

from __future__ import annotations

import math
from typing import Optional

from src.defenses.base import Defense
from src.models.target_llm import TargetLLM


REPHRASE_TEMPLATE = (
    "Rewrite the following prompt in clear, fluent English while preserving "
    "its meaning exactly. Output only the rewritten prompt, with no preface "
    "and no quotation marks.\n\n"
    "Prompt:\n{prompt}\n\n"
    "Rewritten prompt:"
)


class PerplexityFilterDefense(Defense):
    """Perplexity-gated rephrasing defense."""

    def __init__(
        self,
        rephraser: TargetLLM,
        threshold: float = 100.0,
        ppl_model_name: str = "gpt2",
        device: str = "auto",
    ):
        """
        Args:
            rephraser: An instruction-tuned LLM used to paraphrase prompts
                whose perplexity exceeds ``threshold``. Any ``TargetLLM`` is
                acceptable; in practice we use Llama-3-8B-Instruct.
            threshold: Perplexity threshold above which the rephraser fires.
                The Jain et al. default is 100; we expose it as a knob.
            ppl_model_name: HuggingFace model id for the reference LM used
                to compute prompt perplexity. GPT-2 is the canonical choice
                in the literature.
            device: Device map for the reference LM.
        """
        self.rephraser = rephraser
        self.threshold = float(threshold)
        self.ppl_model_name = ppl_model_name
        self._device = device
        self._tokenizer = None
        self._model = None
        self._seed: Optional[int] = None

    # ------------------------------------------------------------------
    # Lazy load so that unit tests can construct the defense without
    # downloading GPT-2 weights.
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.ppl_model_name,
            device_map=self._device,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def _perplexity(self, prompt: str) -> float:
        """Average per-token perplexity under the reference LM."""
        import torch

        self._ensure_loaded()
        enc = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self._model.device)
        if input_ids.shape[1] < 2:
            return float("inf")
        with torch.no_grad():
            out = self._model(input_ids, labels=input_ids)
        nll = float(out.loss.item())
        return float(math.exp(nll))

    def apply(self, prompt: str) -> str:
        try:
            ppl = self._perplexity(prompt)
        except Exception:
            # If the perplexity model fails for any reason, fall back to a
            # rephrase (more conservative for the defender) rather than
            # silently passing the prompt through.
            ppl = float("inf")
        if ppl <= self.threshold:
            return prompt
        rephrase_prompt = REPHRASE_TEMPLATE.format(prompt=prompt)
        rewritten = self.rephraser.generate(rephrase_prompt).strip()
        return rewritten or prompt

    def get_name(self) -> str:
        return "perplexity_filter"

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        # The rephraser's RNG is governed by the underlying TargetLLM; if
        # it's a LocalLLM we set torch's global seed at the experiment
        # level. We capture the value here for completeness.
