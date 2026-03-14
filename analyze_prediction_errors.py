#!/usr/bin/env python3
"""Analyze prediction error types and plot diagnostic figures.

Input JSONL format (from test_ccot.py dump):
{
  "question": "...",
  "prediction": "...",
  "reference": "...",
  "exact_match": true/false
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction error types.")
    parser.add_argument("--pred-file", required=True, help="Path to *_predictions.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and summary.")
    parser.add_argument("--bins", type=int, default=30, help="Bins for abs error histogram.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_number(text: str) -> Optional[float]:
    s = "" if text is None else str(text)
    m = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if m is None:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def is_decimal_number(text: str) -> bool:
    s = "" if text is None else str(text)
    m = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if m is None:
        return False
    return "." in m.group(0)


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def main() -> None:
    args = parse_args()
    pred_file = Path(args.pred_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(pred_file)
    if not rows:
        raise ValueError(f"No rows found in {pred_file}")

    total = len(rows)
    exact_count = sum(1 for r in rows if bool(r.get("exact_match", False)))
    mismatch_count = total - exact_count

    numeric_abs_errors: List[float] = []
    int_total = int_hit = 0
    dec_total = dec_hit = 0

    for r in rows:
        pred = str(r.get("prediction", ""))
        gold = str(r.get("reference", ""))
        hit = bool(r.get("exact_match", False))

        pred_num = extract_number(pred)
        gold_num = extract_number(gold)
        if pred_num is not None and gold_num is not None:
            numeric_abs_errors.append(abs(pred_num - gold_num))

        if is_decimal_number(gold):
            dec_total += 1
            if hit:
                dec_hit += 1
        else:
            int_total += 1
            if hit:
                int_hit += 1

    summary = {
        "num_samples": total,
        "exact_match_count": exact_count,
        "mismatch_count": mismatch_count,
        "exact_match_ratio": safe_ratio(exact_count, total),
        "numeric_error_covered_count": len(numeric_abs_errors),
        "integer_accuracy": safe_ratio(int_hit, int_total),
        "decimal_accuracy": safe_ratio(dec_hit, dec_total),
        "integer_count": int_total,
        "decimal_count": dec_total,
    }

    with (out_dir / "error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, only wrote error_summary.json")
        print(f"import error: {e}")
        return

    # 1) Exact vs mismatch pie
    plt.figure(figsize=(5.6, 5.6))
    plt.pie(
        [exact_count, mismatch_count],
        labels=["Exact match", "Mismatch"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4C78A8", "#E45756"],
    )
    plt.title("Exact Match vs Mismatch")
    plt.tight_layout()
    plt.savefig(out_dir / "exact_vs_mismatch_pie.png", dpi=220)
    plt.close()

    # 2) Numeric absolute error histogram
    if numeric_abs_errors:
        plt.figure(figsize=(7.2, 4.6))
        plt.hist(numeric_abs_errors, bins=args.bins, color="#54A24B", alpha=0.9)
        plt.xlabel("|pred - gold|")
        plt.ylabel("Count")
        plt.title("Numeric Absolute Error Distribution")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "numeric_abs_error_hist.png", dpi=220)
        plt.close()

    # 3) Integer vs decimal accuracy
    labels = ["Integer Gold", "Decimal Gold"]
    vals = [safe_ratio(int_hit, int_total), safe_ratio(dec_hit, dec_total)]
    plt.figure(figsize=(6.8, 4.6))
    bars = plt.bar(labels, vals, color=["#4C78A8", "#F58518"])
    plt.ylim(0, max(vals + [0.05]) * 1.2)
    plt.ylabel("Exact Match")
    plt.title("Accuracy by Answer Type")
    plt.grid(axis="y", alpha=0.2)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "int_decimal_accuracy.png", dpi=220)
    plt.close()

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
