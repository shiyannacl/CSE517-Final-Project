#!/usr/bin/env python3
"""Convert openai/gsm8k to whynlp/gsm8k-aug-like schema.

Output schema per sample:
{
  "question": "...",
  "steps": ["<<a+b=c>>", ...],   # expression-only by default
  "answer": "123"
}

This script does NOT perform data augmentation. It only converts format.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


def extract_final_answer(answer_text: str) -> str:
    text = "" if answer_text is None else str(answer_text)
    final = text.rsplit("####", 1)[-1].strip() if "####" in text else text.strip()
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", final)
    if match is None:
        return final
    return match.group(0).replace(",", "")


def split_rationale_and_final(answer_text: str) -> Tuple[str, str]:
    text = "" if answer_text is None else str(answer_text)
    if "####" not in text:
        return text, extract_final_answer(text)
    rationale, _ = text.rsplit("####", 1)
    return rationale, extract_final_answer(text)


def extract_steps_math_only(rationale: str) -> List[str]:
    # Match GSM8K calculator spans, e.g. <<48+24=72>>
    return re.findall(r"<<[^<>]+>>", rationale)


def extract_steps_nl_lines(rationale: str) -> List[str]:
    lines = [ln.strip() for ln in rationale.splitlines()]
    return [ln for ln in lines if ln]


def convert_records(records: Iterable[Dict], step_style: str) -> List[Dict]:
    converted: List[Dict] = []
    for ex in records:
        question = ex.get("question", "")
        rationale, final_answer = split_rationale_and_final(ex.get("answer", ""))

        if step_style == "math":
            steps = extract_steps_math_only(rationale)
            # Fallback: if no <<...>> markers, keep non-empty rationale lines.
            if not steps:
                steps = extract_steps_nl_lines(rationale)
        else:
            steps = extract_steps_nl_lines(rationale)

        converted.append(
            {
                "question": question,
                "steps": steps,
                "answer": final_answer,
            }
        )
    return converted


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM8K in gsm8k-aug-like format.")
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--out_dir", default="data/gsm8k_aug_like")
    parser.add_argument(
        "--step_style",
        choices=["math", "nl"],
        default="math",
        help="'math': extract only <<...>> expressions (closest to gsm8k-aug); "
        "'nl': keep rationale lines as steps.",
    )
    parser.add_argument(
        "--test_from_validation",
        action="store_true",
        help="Also write test.jsonl by copying validation split (useful for --do_predict).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()

    ds = load_dataset(args.dataset_name, args.dataset_config)
    if "train" not in ds:
        raise ValueError("Expected split: train")

    # openai/gsm8k has train+test, while some augmented datasets expose train+validation(+test).
    if "validation" in ds:
        valid_split = "validation"
    elif "test" in ds:
        valid_split = "test"
    else:
        raise ValueError(f"Cannot find validation-like split. Available splits: {list(ds.keys())}")

    train_rows = convert_records(ds["train"], args.step_style)
    valid_rows = convert_records(ds[valid_split], args.step_style)

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "validation.jsonl", valid_rows)

    # Write test.jsonl:
    # 1) If explicitly requested, always copy validation rows.
    # 2) Else if source dataset has test split, export converted test split.
    if args.test_from_validation:
        write_jsonl(out_dir / "test.jsonl", valid_rows)
        test_note = f"test rows: {len(valid_rows)} (copied from {valid_split})"
    elif "test" in ds:
        test_rows = convert_records(ds["test"], args.step_style)
        write_jsonl(out_dir / "test.jsonl", test_rows)
        test_note = f"test rows: {len(test_rows)} (from source test split)"
    else:
        test_note = "test rows: not written (no source test split)"

    print("Done.")
    print(f"splits found: {list(ds.keys())}")
    print(f"validation.jsonl source split: {valid_split}")
    print(f"train rows: {len(train_rows)}")
    print(f"validation rows: {len(valid_rows)}")
    print(test_note)
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
