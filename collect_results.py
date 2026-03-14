#!/usr/bin/env python3
"""Collect PCCoT training results and export CSV/LaTeX table.

Usage example:
python collect_results.py \
  --run-dirs outputs/pccot-gpt2-lora-3-24_gsm8k outputs/pccot-llama1b-lora-3-24 \
  --out-csv results_summary.csv \
  --out-tex results_table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _fmt_num(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
        return f"{v:.{ndigits}f}"
    except Exception:
        return "N/A"


def _fmt_pct(x: Any, ndigits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x) * 100.0
        return f"{v:.{ndigits}f}"
    except Exception:
        return "N/A"


def _get_first(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    eval_metrics = _load_json(run_dir / "eval_results.json") or {}
    test_metrics = _load_json(run_dir / "test_results.json") or {}
    train_metrics = _load_json(run_dir / "train_results.json") or {}
    trainer_state = _load_json(run_dir / "trainer_state.json") or {}

    model_name = run_dir.name
    if "gpt2" in model_name.lower():
        model = "GPT-2"
    elif "llama" in model_name.lower():
        model = "Llama-3.2-1B"
    else:
        model = "Unknown"

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": model,
        "eval_ccot_exact_match": _get_first(eval_metrics, ["eval_ccot_exact_match"]),
        "eval_cot_exact_match": _get_first(eval_metrics, ["eval_cot_exact_match"]),
        "test_ccot_exact_match": _get_first(test_metrics, ["test_ccot_exact_match"]),
        "test_cot_exact_match": _get_first(test_metrics, ["test_cot_exact_match"]),
        "eval_loss": _get_first(eval_metrics, ["eval_loss"]),
        "test_loss": _get_first(test_metrics, ["test_loss"]),
        "eval_runtime_sec": _get_first(eval_metrics, ["eval_runtime"]),
        "test_runtime_sec": _get_first(test_metrics, ["test_runtime"]),
        "train_runtime_sec": _get_first(train_metrics, ["train_runtime"]),
        "train_samples_per_sec": _get_first(train_metrics, ["train_samples_per_second"]),
        "best_metric": _get_first(trainer_state, ["best_metric"]),
        "best_model_checkpoint": _get_first(trainer_state, ["best_model_checkpoint"]),
        "global_step": _get_first(trainer_state, ["global_step"]),
    }
    return row


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    fields = [
        "run_dir",
        "model",
        "eval_ccot_exact_match",
        "eval_cot_exact_match",
        "test_ccot_exact_match",
        "test_cot_exact_match",
        "eval_loss",
        "test_loss",
        "eval_runtime_sec",
        "test_runtime_sec",
        "train_runtime_sec",
        "train_samples_per_sec",
        "best_metric",
        "best_model_checkpoint",
        "global_step",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, Any]], out_tex: Path) -> None:
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Reproduction results for PCCoT training runs.}")
    lines.append("\\label{tab:repro_results}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Model & Eval CCoT EM (\\%) & Test CCoT EM (\\%) & Eval CoT EM (\\%) & Test CoT EM (\\%) \\\\")
    lines.append("\\hline")
    for r in rows:
        line = (
            f"{r['model']} & "
            f"{_fmt_pct(r.get('eval_ccot_exact_match'))} & "
            f"{_fmt_pct(r.get('test_ccot_exact_match'))} & "
            f"{_fmt_pct(r.get('eval_cot_exact_match'))} & "
            f"{_fmt_pct(r.get('test_cot_exact_match'))} \\\\"
        )
        lines.append(line)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_console_summary(rows: List[Dict[str, Any]]) -> None:
    print("=== PCCoT Result Summary ===")
    for i, r in enumerate(rows, start=1):
        print(f"[Run {i}] {r['run_dir']}")
        print(f"  model                 : {r['model']}")
        print(f"  eval_ccot_exact_match : {_fmt_pct(r.get('eval_ccot_exact_match'))}%")
        print(f"  test_ccot_exact_match : {_fmt_pct(r.get('test_ccot_exact_match'))}%")
        print(f"  eval_cot_exact_match  : {_fmt_pct(r.get('eval_cot_exact_match'))}%")
        print(f"  test_cot_exact_match  : {_fmt_pct(r.get('test_cot_exact_match'))}%")
        print(f"  best_metric           : {_fmt_num(r.get('best_metric'))}")
        print(f"  best_checkpoint       : {r.get('best_model_checkpoint') or 'N/A'}")
        print(f"  global_step           : {r.get('global_step') if r.get('global_step') is not None else 'N/A'}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PCCoT run metrics and export CSV/LaTeX.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more output dirs, e.g. outputs/pccot-gpt2-lora-3-24_gsm8k",
    )
    parser.add_argument("--out-csv", default="results_summary.csv", help="Output CSV path.")
    parser.add_argument("--out-tex", default="results_table.tex", help="Output LaTeX table path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    rows = [collect_run_metrics(d) for d in run_dirs]

    out_csv = Path(args.out_csv).resolve()
    out_tex = Path(args.out_tex).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    write_latex(rows, out_tex)
    print_console_summary(rows)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved LaTeX: {out_tex}")


if __name__ == "__main__":
    main()
