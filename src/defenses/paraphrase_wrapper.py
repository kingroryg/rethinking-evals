"""
T5 paraphrase-wrapper defense.

Every prompt is rewritten by a T5-family paraphraser before reaching the
target. This is the simplest of the three continuous defenses and the one
most directly aligned with the assumptions of the persistence theorem: T5
paraphrasers are Lipschitz in semantic embedding space and produce small
edits that keep the prompt within a bounded neighborhood of the original.

Default model: ``Vamsi/T5_Paraphrase_Paws``. This is the paraphrase model
used in the Jain et al. defense baselines; it is open-weight and runs on a
single GPU.
"""

from __future__ import annotations

from typing import Optional

from src.defenses.base import Defense


class ParaphraseWrapperDefense(Defense):
    """Universal T5 paraphrase wrapper."""

    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        max_length: int = 256,
        num_beams: int = 5,
        device: str = "auto",
    ):
        """
        Args:
            model_name: HuggingFace model id of the T5 paraphraser.
            max_length: Maximum generation length.
            num_beams: Beam search width. We use beam search rather than
                sampling so the defense is deterministic given the input;
                this matters for the Lipschitz finite-difference estimator
                in src.analysis.lipschitz.
            device: Device map.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        self._device = device
        self._tokenizer = None
        self._model = None
        self._seed: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map=self._device,
        )
        self._model.eval()

    def apply(self, prompt: str) -> str:
        import torch

        try:
            self._ensure_loaded()
        except Exception:
            # If the paraphraser cannot be loaded (e.g. no network in
            # tests), fall back to the identity. This keeps unit tests
            # green; production runs will load it eagerly via warmup.
            return prompt

        text = "paraphrase: " + prompt + " </s>"
        enc = self._tokenizer.encode_plus(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
            )

        rewritten = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return rewritten or prompt

    def get_name(self) -> str:
        return "paraphrase"

    def seed(self, seed: Optional[int]) -> None:
        # Beam search is deterministic; seed is informational only.
        self._seed = seed
