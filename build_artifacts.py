#!/usr/bin/env python3
"""Stage a clean artifacts/ tree from the completed sweep.

Strips cost fields and provider-specific model IDs.
Skips: pickles, checkpoints, cost ledger, archive prompt files, seed prompts.
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

SRC = "/home/ubuntu/rethinking-evals/data/results/sweep_full_20260424_152642"
DST = "/home/ubuntu/rethinking-evals/artifacts"

COST_KEYS = {
    "cost_delta_usd", "cost_for_target_only_usd", "cumulative_usd", "cap_frac",
    "cost", "cap_usd", "total_usd", "self_usd", "external_usd", "call_count",
    "external_call_count", "per_model_usd", "per_model_tokens",
    "cost_target_usd", "budget_cap_usd",
    "ledger_path", "pricing_path", "cost_snapshot",
}


def strip_cost(obj):
    if isinstance(obj, dict):
        return {k: strip_cost(v) for k, v in obj.items() if k not in COST_KEYS}
    if isinstance(obj, list):
        return [strip_cost(x) for x in obj]
    return obj


def main() -> None:
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    with open(os.path.join(SRC, "sweep_summary.json")) as f:
        s = json.load(f)
    s = strip_cost(s)
    for _m, rec in s.get("per_model", {}).items():
        for r in rec.get("runs", []):
            r.pop("target_model_id", None)
            r.pop("resumed", None)
    with open(os.path.join(DST, "sweep_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print("wrote sweep_summary.json")

    cm = os.path.join(DST, "cross_model")
    os.makedirs(cm, exist_ok=True)
    shutil.copy(os.path.join(SRC, "cross_model/cross_model_heatmap_grid.png"), cm)
    shutil.copy(os.path.join(SRC, "cross_model/cross_model_heatmap_grid.pdf"), cm)
    drop = {"cost_target_usd", "cost_delta_usd", "elapsed_sec"}
    with open(os.path.join(SRC, "cross_model/cross_model_table.csv")) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        cols = [c for c in rdr.fieldnames if c not in drop]
    with open(os.path.join(cm, "cross_model_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote cross_model_table.csv ({len(cols)} cols)")

    main_dst = os.path.join(DST, "main")
    os.makedirs(main_dst, exist_ok=True)
    for model_dir in sorted(glob.glob(os.path.join(SRC, "main/worker_*/*"))):
        model = os.path.basename(model_dir)
        if model in ("per_category_rerun", "cross_model"):
            continue
        if not os.path.isdir(model_dir):
            continue
        for run_dir in sorted(glob.glob(os.path.join(model_dir, "run_*"))):
            run = os.path.basename(run_dir)
            out = os.path.join(main_dst, model, run)
            os.makedirs(out, exist_ok=True)
            viz_src = os.path.join(run_dir, "visualizations")
            if os.path.isdir(viz_src):
                viz_dst = os.path.join(out, "visualizations")
                os.makedirs(viz_dst, exist_ok=True)
                for png in glob.glob(os.path.join(viz_src, "*.png")):
                    shutil.copy(png, viz_dst)
            for fname in ("final_statistics.json", "statistics_history.json", "run_summary.json"):
                sp = os.path.join(run_dir, fname)
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    obj = json.load(f)
                obj = strip_cost(obj)
                if isinstance(obj, dict):
                    obj.pop("target_model_id", None)
                    obj.pop("resumed", None)
                with open(os.path.join(out, fname), "w") as f:
                    json.dump(obj, f, indent=2)
    print("wrote per-model run dirs")

    pc = os.path.join(DST, "per_category_rerun")
    os.makedirs(pc, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(SRC, "main/worker_*/per_category_rerun"))):
        worker = os.path.basename(os.path.dirname(d))
        out = os.path.join(pc, worker)
        os.makedirs(out, exist_ok=True)
        # Aggregates only — per_category_scores.jsonl contains prompts; omit for public release.
        sp = os.path.join(d, "calibration_summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                cal = json.load(f)
            cal = strip_cost(cal)
            with open(os.path.join(out, "calibration_summary.json"), "w") as f:
                json.dump(cal, f, indent=2)
    print("wrote per_category_rerun (calibration_summary only, cost stripped)")

    abl = os.path.join(DST, "ablation")
    os.makedirs(abl, exist_ok=True)
    shutil.copy(os.path.join(SRC, "ablation/summary.json"), abl)
    for run in sorted(glob.glob(os.path.join(SRC, "ablation/run_*"))):
        rname = os.path.basename(run)
        out = os.path.join(abl, rname)
        os.makedirs(out, exist_ok=True)
        for fname in ("full_results.json", "no_ad_results.json", "no_me_results.json"):
            sp = os.path.join(run, fname)
            if os.path.exists(sp):
                shutil.copy(sp, out)
    print("wrote ablation")

    bl = os.path.join(DST, "baselines")
    os.makedirs(bl, exist_ok=True)
    for model_dir in sorted(glob.glob(os.path.join(SRC, "baselines/*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        out = os.path.join(bl, model)
        os.makedirs(out, exist_ok=True)
        sp = os.path.join(model_dir, "summary.json")
        if os.path.exists(sp):
            shutil.copy(sp, out)
        for run in sorted(glob.glob(os.path.join(model_dir, "run_*"))):
            rname = os.path.basename(run)
            ro = os.path.join(out, rname)
            os.makedirs(ro, exist_ok=True)
            for fname in ("random_results.json", "gcg_results.json", "pair_results.json", "tap_results.json"):
                sp = os.path.join(run, fname)
                if os.path.exists(sp):
                    shutil.copy(sp, ro)
    print("wrote baselines")

    panel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_panel.py")
    subprocess.check_call([sys.executable, panel])


if __name__ == "__main__":
    main()
