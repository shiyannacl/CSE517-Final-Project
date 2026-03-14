#!/usr/bin/env python3
"""Plot training/evaluation figures from a PCCoT output directory.

Example:
python plot_results.py \
  --run-dir outputs/pccot-gpt2-lora-3-24 \
  --out-dir outputs/pccot-gpt2-lora-3-24/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def extract_series(log_history: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for row in log_history:
        if key in row and "step" in row:
            try:
                xs.append(int(row["step"]))
                ys.append(float(row[key]))
            except Exception:
                continue
    return xs, ys


def moving_average(vals: List[float], window: int) -> List[float]:
    if window <= 1 or len(vals) < window:
        return vals[:]
    out = []
    cumsum = 0.0
    q: List[float] = []
    for v in vals:
        q.append(v)
        cumsum += v
        if len(q) > window:
            cumsum -= q.pop(0)
        out.append(cumsum / len(q))
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(log_history: List[Dict[str, Any]], out_dir: Path, smooth_window: int) -> None:
    train_x, train_y = extract_series(log_history, "loss")
    eval_x, eval_y = extract_series(log_history, "eval_loss")
    if not train_x and not eval_x:
        return

    plt.figure(figsize=(8, 5))
    if train_x:
        plt.plot(train_x, moving_average(train_y, smooth_window), label="train_loss", linewidth=1.4)
    if eval_x:
        plt.plot(eval_x, eval_y, label="eval_loss", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def plot_em_curves(log_history: List[Dict[str, Any]], out_dir: Path) -> None:
    ccot_x, ccot_y = extract_series(log_history, "eval_ccot_exact_match")
    cot_x, cot_y = extract_series(log_history, "eval_cot_exact_match")
    if not ccot_x and not cot_x:
        return

    plt.figure(figsize=(8, 5))
    if ccot_x:
        plt.plot(ccot_x, ccot_y, label="eval_ccot_exact_match", linewidth=2.0)
    if cot_x:
        plt.plot(cot_x, cot_y, label="eval_cot_exact_match", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Exact Match")
    plt.title("Validation Exact Match Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eval_em_curves.png", dpi=200)
    plt.close()


def plot_gap_curve(log_history: List[Dict[str, Any]], out_dir: Path) -> None:
    eval_rows = [
        row for row in log_history
        if "step" in row and "eval_ccot_exact_match" in row and "eval_cot_exact_match" in row
    ]
    if not eval_rows:
        return

    xs = [int(r["step"]) for r in eval_rows]
    gaps = [float(r["eval_cot_exact_match"]) - float(r["eval_ccot_exact_match"]) for r in eval_rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, gaps, linewidth=2.0)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Step")
    plt.ylabel("CoT - CCoT (EM gap)")
    plt.title("Validation EM Gap Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_gap_curve.png", dpi=200)
    plt.close()


def plot_final_em_bars(eval_results: Dict[str, Any], test_results: Dict[str, Any], out_dir: Path) -> None:
    labels = ["Eval CCoT", "Eval CoT", "Test CCoT", "Test CoT"]
    values = [
        eval_results.get("eval_ccot_exact_match"),
        eval_results.get("eval_cot_exact_match"),
        test_results.get("test_ccot_exact_match"),
        test_results.get("test_cot_exact_match"),
    ]
    if all(v is None for v in values):
        return

    cleaned = [float(v) if v is not None else 0.0 for v in values]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    plt.figure(figsize=(7.8, 4.8))
    bars = plt.bar(labels, cleaned, color=colors)
    plt.ylabel("Exact Match")
    plt.title("Final Exact Match Comparison")
    plt.ylim(0, max(cleaned + [0.05]) * 1.15)
    plt.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, cleaned):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "final_em_bars.png", dpi=200)
    plt.close()


def plot_efficiency_bars(train_results: Dict[str, Any], eval_results: Dict[str, Any], test_results: Dict[str, Any], out_dir: Path) -> None:
    labels = ["Train", "Eval", "Test"]
    sps = [
        train_results.get("train_samples_per_second"),
        eval_results.get("eval_samples_per_second"),
        test_results.get("test_samples_per_second"),
    ]
    rt = [
        train_results.get("train_runtime"),
        eval_results.get("eval_runtime"),
        test_results.get("test_runtime"),
    ]
    if any(v is not None for v in sps):
        sps_vals = [float(v) if v is not None else 0.0 for v in sps]
        plt.figure(figsize=(7.2, 4.5))
        bars = plt.bar(labels, sps_vals, color=["#4C78A8", "#F58518", "#54A24B"])
        plt.ylabel("Samples / second")
        plt.title("Throughput Comparison")
        plt.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, sps_vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "throughput_bars.png", dpi=200)
        plt.close()

    if any(v is not None for v in rt):
        rt_vals = [float(v) if v is not None else 0.0 for v in rt]
        plt.figure(figsize=(7.2, 4.5))
        bars = plt.bar(labels, rt_vals, color=["#4C78A8", "#F58518", "#54A24B"])
        plt.ylabel("Runtime (seconds)")
        plt.title("Runtime Comparison")
        plt.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, rt_vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "runtime_bars.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCCoT figures from output artifacts.")
    parser.add_argument("--run-dir", default="outputs/pccot-gpt2-lora-3-24", help="Path to run output dir.")
    parser.add_argument("--out-dir", default=None, help="Figure output dir. Default: <run-dir>/figures")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for train loss curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "figures"
    ensure_dir(out_dir)

    trainer_state = load_json(run_dir / "trainer_state.json")
    eval_results = load_json(run_dir / "eval_results.json")
    test_results = load_json(run_dir / "test_results.json")
    train_results = load_json(run_dir / "train_results.json")
    log_history = trainer_state.get("log_history", [])
    if not isinstance(log_history, list):
        log_history = []

    plot_loss_curves(log_history, out_dir, args.smooth_window)
    plot_em_curves(log_history, out_dir)
    plot_gap_curve(log_history, out_dir)
    plot_final_em_bars(eval_results, test_results, out_dir)
    plot_efficiency_bars(train_results, eval_results, test_results, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
