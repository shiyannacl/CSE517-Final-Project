# PCCoT

<div align="center">
<img width="200" src="imgs/cover.jpg" />
<p>
  CSE517 reproduction and benchmarking project for PCCoT
  <br>
  with added local training, inference benchmarking,
  <br>
  analysis, and data preprocessing utilities.
</p>
</div>

This repository is our **CSE517 reproduction project** for the paper
"[Parallel Continuous Chain-of-Thought with Jacobi Iteration](https://arxiv.org/abs/2506.18582)".
The original PCCoT project studies parallel continuous chain-of-thought reasoning with Jacobi iteration.(https://github.com/whyNLP/PCCoT)

This codebase contains both the original PCCoT implementation and our reproduction-oriented additions, including:

- local GSM8K-format data preprocessing
- reproducible training shell scripts
- benchmark inference speed testing
- multi-seed evaluation utilities
- result collection and error analysis scripts

## Project Scope

This repo should be read as a **reproduction + course project workspace**, not just a clean mirror of the original paper code.

In particular, our CSE517 work adds support for:

- training from locally converted GSM8K-style JSONL data
- benchmarking PCCoT vs standard CoT inference throughput
- collecting and summarizing multi-run evaluation results
- analyzing prediction errors for reproduced runs

## Installation

Install `pytorch`, `flash-attn`, and the packages in `requirements.txt`.
The original project was developed with Python 3.12.4.

```bash
conda install pytorch=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install mkl=2022.1.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -r requirements.txt
```

## Repository Additions For CSE517

The most relevant scripts for our reproduction workflow are:

- [run_ccot_train0.sh](/data1/yan/PCCoT/run_ccot_train0.sh): local PCCoT training entry point
- [run_ccot_test_3seeds.sh](/data1/yan/PCCoT/run_ccot_test_3seeds.sh): multi-seed evaluation wrapper
- [prepare_gsm8k_aug_like.py](/data1/yan/PCCoT/prepare_gsm8k_aug_like.py): convert `openai/gsm8k` to PCCoT-compatible JSONL
- [benchmark_inference_time.py](/data1/yan/PCCoT/benchmark_inference_time.py): benchmark PCCoT vs CoT inference speed
- [collect_results.py](/data1/yan/PCCoT/collect_results.py): collect metrics from experiment folders
- [plot_results.py](/data1/yan/PCCoT/plot_results.py): plot summarized metrics
- [analyze_prediction_errors.py](/data1/yan/PCCoT/analyze_prediction_errors.py): inspect prediction failures

## Data Preprocessing

For our reproduction, we often use locally converted GSM8K data instead of relying only on the hosted augmented dataset.

Convert `openai/gsm8k` into a `gsm8k-aug-like` schema:

```bash
python prepare_gsm8k_aug_like.py \
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --out_dir data/gsm8k_aug_like \
  --step_style math
```

Optional:

- use `--step_style nl` to keep natural-language rationale lines
- use `--test_from_validation` if you want `test.jsonl` copied from the validation split

The generated format is compatible with the processor in [data_processor.py](/data1/yan/PCCoT/models/data_processor.py).

## Training

For the CSE517 reproduction, the main training entry point is:

```bash
bash run_ccot_train0.sh
```

This script:

- checks for local JSONL files under `data/gsm8k_aug_like` or `data/gsm8k_aug_like_check`
- launches PCCoT training with `run_ccot.py`
- currently trains a GPT-2 based PCCoT configuration

The main trainer implementation remains:

- [run_ccot.py](/data1/yan/PCCoT/run_ccot.py)
- [run_cot.py](/data1/yan/PCCoT/run_cot.py)

If you want the original template-based workflow, the repo still includes:

- [run_ccot.sh.template](/data1/yan/PCCoT/run_ccot.sh.template)
- [run_cot.sh.template](/data1/yan/PCCoT/run_cot.sh.template)

## Benchmark Inference Testing

We added a dedicated benchmark script to measure inference throughput for PCCoT and compare it against standard CoT:

```bash
python benchmark_inference_time.py \
  --model_name_or_path gpt2 \
  --pccot_config_name configs/pccot_gpt2_small.json \
  --iterations 1,3,5,7 \
  --latent_tokens 6,12,18,24 \
  --num_questions 128 \
  --batch_size 16
```

This script:

- benchmarks a standard CoT baseline first
- benchmarks PCCoT under different iteration counts and latent token counts
- writes CSV and JSON summaries under `outputs/inference_benchmarks/`

## Evaluation And Multi-Seed Testing

For repeated evaluation in our reproduction experiments:

```bash
bash run_ccot_test_3seeds.sh
```

By default, this runs seeds `42 52 62`.

Useful environment variables:

- `CKPT_PATH=/abs/path/to/checkpoint` to evaluate the same checkpoint under multiple seeds
- `EVAL_MODE=numeric` to use trainer metrics
- `EVAL_MODE=generation` to use free generation with [test_ccot.py](/data1/yan/PCCoT/test_ccot.py)

This workflow also writes a summary JSON using `summarize_seed_tests.py`.

## Analysis

We added several scripts for analyzing reproduced runs:

```bash
python collect_results.py
python plot_results.py
python analyze_prediction_errors.py
```

These scripts are intended for post-hoc reporting and debugging:

- aggregate metrics across output folders
- visualize performance trends
- inspect wrong predictions and failure patterns

## Quick Example

The original example entry point is still available:

```bash
python example.py
```

## Configuration

We provide sample model configs in `configs/`.
The core PCCoT configuration classes are defined in:

- [configuration_gpt2.py](/data1/yan/PCCoT/models/configuration_gpt2.py)
- [configuration_llama.py](/data1/yan/PCCoT/models/configuration_llama.py)
- [pccot_arguments.py](/data1/yan/PCCoT/models/pccot_arguments.py)

You can either modify the JSON config files directly or override config values from the command line.

Example:

```bash
python -m accelerate.commands.launch run_ccot.py \
  --config_name configs/pccot_gpt2_small.json \
  --config_overrides num_iterations=3
```


