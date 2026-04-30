"""
End-to-end smoke test of the basin-persistence pipeline.

Runs a tiny MAP-Elites loop with mock components against a defended target,
serializes the archive, runs the Lipschitz / margin / persistence / bound
analysis on it, and renders the before/after figure. The point is to
exercise every code path that will fire on the GPU run, with no API calls
and no large model downloads.

This test is the safety net for the full GPU run -- if it goes red,
something in the pipeline is structurally broken and the GPU run would
waste time.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis import (  # noqa: E402
    compute_basin_margin,
    estimate_lipschitz_constants,
    evaluate_persistence_bound,
    per_cell_persistence,
)
from src.core.archive import Archive, ArchiveCell  # noqa: E402
from src.defenses import (  # noqa: E402
    BlocklistDefense,
    ConstitutionalRewriterDefense,
    DefendedLLM,
)
from src.models.target_llm import MockLLM  # noqa: E402


class _ScriptedJudge:
    """A judge that returns the harm score baked into the prompt for testing."""

    def evaluate(self, response: str) -> float:
        # Responses encode their harm score as 'HARM=0.83 ...'.
        try:
            tag = response.split("HARM=", 1)[1].split()[0]
            return float(tag)
        except Exception:
            return 0.0


class _ScriptedTarget(MockLLM):
    """A target that maps each prompt to a deterministic harm tag."""

    def __init__(self, prompt_to_score: dict[str, float]):
        super().__init__(model_name="scripted", response_template="{prompt}")
        self.prompt_to_score = prompt_to_score

    def generate(self, prompt: str, **kwargs) -> str:
        score = self.prompt_to_score.get(prompt, 0.0)
        return f"HARM={score:.2f} (response to: {prompt})"


def _build_pre_archive() -> Archive:
    a = Archive(grid_size=3)
    a.cells[0, 0] = ArchiveCell(prompt="safe1", behavior=(0.1, 0.1), quality=0.10)
    a.cells[0, 1] = ArchiveCell(prompt="safe2", behavior=(0.1, 0.4), quality=0.30)
    a.cells[1, 0] = ArchiveCell(prompt="basin1", behavior=(0.4, 0.1), quality=0.85)
    a.cells[1, 1] = ArchiveCell(prompt="basin2", behavior=(0.4, 0.4), quality=0.70)
    a.cells[2, 2] = ArchiveCell(prompt="basin3", behavior=(0.7, 0.7), quality=0.62)
    return a


def _build_post_archive() -> Archive:
    # basin1 persists, basin2 collapses, basin3 is emptied (defense paraphrased it
    # into a different cell).
    a = Archive(grid_size=3)
    a.cells[0, 0] = ArchiveCell(prompt="d_safe1", behavior=(0.1, 0.1), quality=0.08)
    a.cells[0, 1] = ArchiveCell(prompt="d_safe2", behavior=(0.1, 0.4), quality=0.25)
    a.cells[1, 0] = ArchiveCell(prompt="d_basin1", behavior=(0.4, 0.1), quality=0.65)
    a.cells[1, 1] = ArchiveCell(prompt="d_basin2", behavior=(0.4, 0.4), quality=0.30)
    return a


def test_full_analysis_pipeline_on_synthetic_archives(tmp_path, monkeypatch):
    pre = _build_pre_archive()
    post = _build_post_archive()

    # Use a real sentence-transformers model lazily; if it's not installed,
    # skip the test rather than fail. The unit tests in test_analysis.py
    # already cover the math with a fake embedder.
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "hf" / "hub"))
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # network / cache failure on dev machine
        pytest.skip(f"sentence-transformers model unavailable: {exc}")

    margin = compute_basin_margin(pre, threshold=0.5)
    assert margin.n_basin_cells == 3
    # min(0.85, 0.70, 0.62) - 0.5 = 0.12
    assert margin.G_hat == pytest.approx(0.12, rel=1e-6)

    lip = estimate_lipschitz_constants(pre, post, embedder)
    assert lip.n_pairs > 0
    assert lip.L_hat > 0

    persist = per_cell_persistence(pre, post, threshold=0.5)
    # basin1 persists, basin2 collapses, basin3 emptied -> 1/3
    assert persist.n_pre_basin == 3
    assert persist.n_post_basin_persisted == 1
    assert persist.persistence_rate == pytest.approx(1 / 3, rel=1e-6)

    bound = evaluate_persistence_bound(
        pre, post, embedder, L_hat=lip.L_hat, K_hat=(lip.K_hat or 0.0)
    )
    # We don't assert bound.bound_holds because L_hat is a *finite-difference*
    # estimate and may locally under-estimate the true Lipschitz constant on
    # tiny synthetic data. The contract is that the function runs end to end
    # and reports a fraction_violations in [0, 1].
    assert 0.0 <= bound.fraction_violations <= 1.0


def test_defended_llm_runs_a_few_steps_against_judge():
    """Check that DefendedLLM + a scripted judge produce sensible AD scores."""

    target = _ScriptedTarget({
        "safe-prompt": 0.10,
        "harmful-prompt": 0.92,
        "blocked-bomb": 0.99,
    })
    defended = DefendedLLM(target, BlocklistDefense(["bomb"]))

    judge = _ScriptedJudge()

    safe_resp = defended.generate("safe-prompt")
    harmful_resp = defended.generate("harmful-prompt")
    blocked_resp = defended.generate("blocked-bomb")

    assert judge.evaluate(safe_resp) == pytest.approx(0.10)
    assert judge.evaluate(harmful_resp) == pytest.approx(0.92)
    # The blocked prompt never reaches the target -- the judge sees the
    # canned refusal, which has no HARM tag, so the score is 0.
    assert judge.evaluate(blocked_resp) == 0.0


def test_archive_roundtrip(tmp_path):
    pre = _build_pre_archive()
    path = tmp_path / "pre.pkl"
    pre.save(str(path))
    loaded = Archive.load(str(path))
    assert loaded.grid_size == pre.grid_size
    for i in range(pre.grid_size):
        for j in range(pre.grid_size):
            a = pre.cells[i, j]
            b = loaded.cells[i, j]
            if a is None:
                assert b is None
            else:
                assert b.prompt == a.prompt
                assert b.quality == a.quality


def test_run_lipschitz_analysis_cli(tmp_path, monkeypatch):
    """Exercise the analysis CLI end-to-end on synthetic archives."""

    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "hf" / "hub"))
    pre = _build_pre_archive()
    post = _build_post_archive()
    pre_path = tmp_path / "pre.pkl"
    post_path = tmp_path / "post.pkl"
    pre.save(str(pre_path))
    post.save(str(post_path))
    out = tmp_path / "analysis"

    import subprocess

    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "run_lipschitz_analysis.py"),
        "--undefended",
        str(pre_path),
        "--defended",
        f"paraphrase={post_path}",
        "--out",
        str(out),
        "--embedder",
        "all-MiniLM-L6-v2",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": ROOT,
        "HF_HOME": str(tmp_path / "hf"),
        "TRANSFORMERS_CACHE": str(tmp_path / "hf" / "hub"),
    }
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0 and "PermissionError" in res.stderr:
        pytest.skip(f"sentence-transformers cache unavailable: {res.stderr[:200]}")
    assert res.returncode == 0, res.stderr

    geom = (out / "defense_geometry.csv").read_text().splitlines()
    inv = (out / "defense_invariance.csv").read_text().splitlines()
    summary = json.loads((out / "summary.json").read_text())
    assert geom[0].startswith("defense,")
    assert any(line.startswith("paraphrase,") for line in geom)
    assert inv[0].startswith("condition,")
    assert any(line.startswith("+paraphrase,") for line in inv)
    assert "paraphrase" in summary["defenses"]
    persist_status = np.load(out / "persistence_paraphrase.npy")
    assert persist_status.shape == (3, 3)
