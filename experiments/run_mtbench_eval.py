"""
MT-Bench utility evaluation for the trilemma corner of Section sec:reliability.

The paper claims a discontinuous defense (the n-gram blocklist) drops the
basin rate but at the cost of utility on benign inputs. To substantiate
the cost claim we run lm-sys/FastChat's MT-Bench against:

  * baseline:  Llama-3-8B-Instruct (undefended)
  * defended:  Llama-3-8B-Instruct wrapped in BlocklistDefense

MT-Bench scores both with GPT-4 as the judge across 80 multi-turn prompts
(the standard split). The utility regression is

    R = (mean_score_baseline - mean_score_defended) / mean_score_baseline.

This is a *thin* harness around FastChat: we generate answers using our
defended TargetLLM (so the blocklist actually fires on benign questions
that mention any blocked n-gram, which is the source of utility loss) and
then call ``llm_judge.gen_judgment`` to score both answer sets against the
GPT-4 reference judgments.

If FastChat is not installed, this script prints a single command the user
should run instead. We do not silently fall back to a proxy metric because
the paper's claim is specifically about MT-Bench.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml  # noqa: E402

from src.defenses import BlocklistDefense, DefendedLLM  # noqa: E402
from src.models.target_llm import create_target_llm  # noqa: E402


def _check_fastchat_available() -> bool:
    try:
        importlib.import_module("fastchat.llm_judge.common")
        return True
    except Exception:
        return False


def _load_mtbench_questions(question_file: Optional[str] = None) -> list[dict]:
    """Load MT-Bench's 80 standard questions.

    Looks in:
      1. ``question_file`` if provided.
      2. ``$FASTCHAT_DIR/llm_judge/data/mt_bench/question.jsonl`` if FastChat
         is importable.
    """
    candidates = []
    if question_file:
        candidates.append(question_file)

    try:
        fc = importlib.import_module("fastchat")
        fc_root = os.path.dirname(fc.__file__)
        candidates.append(os.path.join(fc_root, "llm_judge", "data", "mt_bench", "question.jsonl"))
    except Exception:
        pass

    for path in candidates:
        if path and os.path.exists(path):
            qs = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        qs.append(json.loads(line))
            return qs

    raise FileNotFoundError(
        "Could not locate MT-Bench question.jsonl. Pass --questions or install FastChat: "
        "pip install 'fschat[model_worker,llm_judge]'"
    )


def _generate_answers(target, questions: list[dict]) -> list[dict]:
    """Run a TargetLLM through MT-Bench's two-turn protocol."""
    answers = []
    for q in questions:
        turns = q["turns"]
        history = ""
        responses = []
        for turn in turns:
            prompt = (history + "\n\nUser: " + turn + "\n\nAssistant:") if history else turn
            response = target.generate(prompt)
            responses.append(response)
            history = (history + f"\n\nUser: {turn}\n\nAssistant: {response}").strip()
        answers.append(
            {
                "question_id": q["question_id"],
                "category": q.get("category"),
                "model_id": target.get_model_name(),
                "choices": [{"index": 0, "turns": responses}],
            }
        )
    return answers


def _score_with_gpt4_judge(
    answers_baseline: list[dict],
    answers_defended: list[dict],
    questions: list[dict],
    judge_model: str,
    out_dir: str,
) -> dict:
    """Compute mean scores with GPT-4 as the absolute-scale judge.

    We use the FastChat ``single`` mode (1-10 absolute score per turn) so
    that baseline and defended runs can be compared on a common scale
    without pairwise re-judging.
    """
    from fastchat.llm_judge.common import (
        load_judge_prompts,
        play_a_match_single,
        MatchSingle,
    )

    judge_prompts = load_judge_prompts(
        os.path.join(
            os.path.dirname(importlib.import_module("fastchat").__file__),
            "llm_judge",
            "data",
            "judge_prompts.jsonl",
        )
    )

    def _score(answers, label) -> float:
        score_path = os.path.join(out_dir, f"mtbench_{label}_judgments.jsonl")
        scores = []
        with open(score_path, "w") as f:
            for q, a in zip(questions, answers):
                judge = judge_prompts["single-v1-multi-turn" if len(q["turns"]) > 1 else "single-v1"]
                match = MatchSingle(
                    dict(q),
                    dict(a),
                    {"judge_model": judge_model, "type": judge["name"]},
                    None,
                    judge_prompts,
                )
                result = play_a_match_single(match, None)
                f.write(json.dumps(result) + "\n")
                if isinstance(result.get("score"), list):
                    scores.extend([s for s in result["score"] if s is not None])
                elif result.get("score") is not None:
                    scores.append(result["score"])
        return sum(scores) / len(scores) if scores else 0.0

    mean_baseline = _score(answers_baseline, "baseline")
    mean_defended = _score(answers_defended, "blocklist")
    regression = (mean_baseline - mean_defended) / mean_baseline if mean_baseline else 0.0

    summary = {
        "judge_model": judge_model,
        "mean_score_baseline": mean_baseline,
        "mean_score_blocklist": mean_defended,
        "utility_regression": regression,
    }
    with open(os.path.join(out_dir, "mtbench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--judge-model", default="gpt-4-1106-preview")
    parser.add_argument("--questions", default=None)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "results", "mtbench"),
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    if not _check_fastchat_available():
        print(
            "FastChat not installed. Install with:\n"
            "  uv pip install 'fschat[model_worker,llm_judge]'\n"
            "and then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(2)

    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    with open(os.path.join(config_dir, "models.yaml")) as f:
        models_config = yaml.safe_load(f)
    target_config = models_config["target_models"][args.model]
    base = create_target_llm(target_config)
    defended = DefendedLLM(base, BlocklistDefense())

    questions = _load_mtbench_questions(args.questions)

    print(f"Generating {len(questions)} answers (baseline)...")
    baseline_answers = _generate_answers(base, questions)
    with open(os.path.join(out_dir, "answers_baseline.jsonl"), "w") as f:
        for a in baseline_answers:
            f.write(json.dumps(a) + "\n")

    print(f"Generating {len(questions)} answers (blocklist defended)...")
    defended_answers = _generate_answers(defended, questions)
    with open(os.path.join(out_dir, "answers_blocklist.jsonl"), "w") as f:
        for a in defended_answers:
            f.write(json.dumps(a) + "\n")

    print("Scoring with GPT-4 as judge...")
    summary = _score_with_gpt4_judge(
        baseline_answers,
        defended_answers,
        questions,
        judge_model=args.judge_model,
        out_dir=out_dir,
    )

    summary["timestamp"] = datetime.now().isoformat()
    summary["blocklist_refusal_rate"] = defended.defense.refusal_rate  # type: ignore[attr-defined]
    with open(os.path.join(out_dir, "mtbench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
