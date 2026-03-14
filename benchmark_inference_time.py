#!/usr/bin/env python
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import models


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_questions(num_questions: int) -> List[str]:
    questions = []
    for i in range(num_questions):
        a = 17 + i
        b = 29 + i * 2
        questions.append(
            f"If Alice has {a} apples and buys {b} more, how many apples does she have now?"
        )
    return questions


def resolve_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    sync_if_needed()
    return time.perf_counter()


def benchmark_cot(
    model,
    tokenizer,
    questions: List[str],
    batch_size: int,
    max_new_tokens: int,
    warmup_batches: int,
) -> Dict[str, float]:
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    with torch.inference_mode():
        for i in range(min(warmup_batches, len(batches))):
            inputs = tokenizer(
                batches[i],
                padding=True,
                padding_side="left",
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.device)
            if "gpt2" in str(model.config.model_type):
                position_ids = inputs["attention_mask"].cumsum(dim=-1) - 1
                inputs["position_ids"] = position_ids.masked_fill(
                    inputs["attention_mask"] == 0, 0
                )
            _ = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        total_new_tokens = 0
        total_samples = 0
        t0 = now()
        for batch in batches:
            inputs = tokenizer(
                batch,
                padding=True,
                padding_side="left",
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            if "gpt2" in str(model.config.model_type):
                position_ids = inputs["attention_mask"].cumsum(dim=-1) - 1
                inputs["position_ids"] = position_ids.masked_fill(
                    inputs["attention_mask"] == 0, 0
                )
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            total_new_tokens += int((outputs.shape[1] - prompt_len) * outputs.shape[0])
            total_samples += outputs.shape[0]
        t1 = now()

    elapsed_s = t1 - t0
    return {
        "method": "cot",
        "elapsed_s": elapsed_s,
        "samples": total_samples,
        "new_tokens": total_new_tokens,
        "samples_per_s": total_samples / elapsed_s,
        "tokens_per_s": total_new_tokens / elapsed_s,
    }


def benchmark_pccot(
    model,
    tokenizer,
    pccot_args: models.PCCoTArguments,
    questions: List[str],
    batch_size: int,
    max_new_tokens: int,
    warmup_batches: int,
) -> Dict[str, float]:
    processor = models.COTDataProcessor(tokenizer=tokenizer, pccot_args=pccot_args)
    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    with torch.inference_mode():
        for i in range(min(warmup_batches, len(batches))):
            collated = processor.process(batches[i], device=model.device)
            _ = model.generate(
                collated=collated,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        total_new_tokens = 0
        total_samples = 0
        t0 = now()
        for batch in batches:
            collated = processor.process(batch, device=model.device)
            prompt_len = collated["input_ids"].shape[1]
            outputs = model.generate(
                collated=collated,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            total_new_tokens += int((outputs.shape[1] - prompt_len) * outputs.shape[0])
            total_samples += outputs.shape[0]
        t1 = now()

    elapsed_s = t1 - t0
    return {
        "method": "pccot",
        "elapsed_s": elapsed_s,
        "samples": total_samples,
        "new_tokens": total_new_tokens,
        "samples_per_s": total_samples / elapsed_s,
        "tokens_per_s": total_new_tokens / elapsed_s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed of PCCoT vs standard CoT."
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument(
        "--pccot_config_name", type=str, default="configs/pccot_gpt2_small.json"
    )
    parser.add_argument("--iterations", type=str, default="1,3,5,7")
    parser.add_argument("--latent_tokens", type=str, default="6,12,18,24")
    parser.add_argument("--num_questions", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "flash_attention_2", "sdpa"],
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/inference_benchmarks/pccot_vs_cot.csv",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/inference_benchmarks/pccot_vs_cot.json",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype)
    iterations = parse_int_list(args.iterations)
    latent_tokens = parse_int_list(args.latent_tokens)
    questions = build_questions(args.num_questions)

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})

    print("Loading CoT baseline model...")
    cot_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    cot_model.eval()

    cot_result = benchmark_cot(
        model=cot_model,
        tokenizer=tokenizer,
        questions=questions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        warmup_batches=args.warmup_batches,
    )
    print(
        f"[cot] tokens/s={cot_result['tokens_per_s']:.2f}, samples/s={cot_result['samples_per_s']:.2f}"
    )
    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading PCCoT model...")
    pccot_config = AutoConfig.from_pretrained(args.pccot_config_name)
    pccot_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=pccot_config,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    pccot_model.eval()

    pccot_args = models.PCCoTArguments(use_peft=False)

    def get_special_token(tok, token_text):
        if token_text not in tok.additional_special_tokens:
            tok.add_special_tokens(
                {"additional_special_tokens": (token_text,)},
                replace_additional_special_tokens=False,
            )
        return tok.convert_tokens_to_ids(token_text)

    pccot_args.bot_token_id = get_special_token(tokenizer, "<pccot.bot>")
    pccot_args.eot_token_id = get_special_token(tokenizer, "<pccot.eot>")
    pccot_args.latent_token_id = get_special_token(tokenizer, "<pccot.latent>")

    if len(tokenizer) > pccot_model.get_input_embeddings().weight.shape[0]:
        pccot_model.resize_token_embeddings(len(tokenizer))

    rows = []
    for n_iter in iterations:
        pccot_model.config.num_iterations = n_iter
        for n_latent in latent_tokens:
            pccot_args.num_latent_tokens = n_latent
            result = benchmark_pccot(
                model=pccot_model,
                tokenizer=tokenizer,
                pccot_args=pccot_args,
                questions=questions,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                warmup_batches=args.warmup_batches,
            )
            row = {
                "method": "pccot",
                "num_iterations": n_iter,
                "num_latent_tokens": n_latent,
                "elapsed_s": result["elapsed_s"],
                "samples": result["samples"],
                "new_tokens": result["new_tokens"],
                "samples_per_s": result["samples_per_s"],
                "tokens_per_s": result["tokens_per_s"],
                "cot_tokens_per_s": cot_result["tokens_per_s"],
                "speedup_vs_cot": result["tokens_per_s"] / cot_result["tokens_per_s"],
            }
            rows.append(row)
            print(
                f"[pccot] iter={n_iter}, latent={n_latent}, "
                f"tokens/s={row['tokens_per_s']:.2f}, speedup_vs_cot={row['speedup_vs_cot']:.3f}"
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "setup": {
            "model_name_or_path": args.model_name_or_path,
            "pccot_config_name": args.pccot_config_name,
            "iterations": iterations,
            "latent_tokens": latent_tokens,
            "num_questions": args.num_questions,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "device": device,
        },
        "cot_baseline": cot_result,
        "pccot_results": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved CSV to {out_csv}")
    print(f"Saved JSON to {out_json}")


if __name__ == "__main__":
    main()
