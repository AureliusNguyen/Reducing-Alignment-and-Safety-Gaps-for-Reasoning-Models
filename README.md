# Reducing Alignment and Safety Gaps for Reasoning Models

Reinforcement-learning fine-tuning of small instruction-tuned language models with
GRPO (Group Relative Policy Optimization), studying how reasoning-oriented training
affects instruction following and safety. We train three 1-2B instruct models under
two reward settings and two group sizes, then evaluate reasoning, instruction
following, and safety at pass@1 and pass@4.

## Summary

Small instruct models are often tuned to reason better on math, but it is unclear
whether that training erodes their instruction following and safety behavior. We
fine-tune with GRPO under two reward configurations and measure the trade-off across
five benchmarks.

- **Reward settings**
  - `reasoning`: math-correctness plus answer-format reward, trained on DAPO-Math-17k.
  - `reasoning+instruction`: adds an instruction/constraint-following reward on a
    math-with-constraints dataset, so the model is rewarded for both solving the
    problem and obeying the stated constraints.
- **Group sizes**: G = 4 and G = 8 generations per prompt.
- **Models**: Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, Gemma-3-1B-Instruct.

## Benchmarks

Each model is evaluated at pass@1 and pass@4:

- **IFEval** - instruction following (prompt-level and instruction-level, strict and loose).
- **AIME** - competition math reasoning.
- **MathIF** - instruction following inside math problems (strict/loose accuracy,
  correctness with and without the constraint).
- **SafetyBench** - multiple-choice safety.
- **SORRY-Bench** - refusal / safe-response rate, scored by a judge model.

## Repository structure

```
training/            GRPO training code
  main.py              entry point (accelerate + TRL GRPOTrainer + vLLM generation)
  custom_grpo_trainer.py
  select_g_methods.py, pd_g_select*.py   group-size selection methods
  custom_constraints_verifier/           instruction-constraint checker and rewards
  deepspeed_config/                      ZeRO-2 configs
  utils.py, varrs.py, run_vllm.py
evaluation/
  pass1/               pass@1 scoring and result parsing
  pass4/               pass@4 generation, scoring, and result parsing
results/
  pass1_all.csv        all models, pass@1
  pass4_all.csv        all models, pass@4
  results.html         both tables rendered for viewing
  view_csv.py          render a CSV as an aligned table in the terminal
docs/                  evaluation-pipeline notes
```

The benchmark implementations themselves (MathIF, SafetyBench, SORRY-Bench / FastChat,
IFEval, AIME) come from a separate harness and are not vendored here. Cluster launch
scripts (`*.slurm`, `*.sh`), model checkpoints, datasets, and cached generations are
intentionally excluded; see `.gitignore`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# vLLM is installed separately; a recent release matching your torch/CUDA build works.
pip install vllm
```

Gated models (the instruct checkpoints and the SORRY-Bench judge) require a Hugging
Face token with access. Provide it through the environment, never in code:

```bash
export HF_TOKEN=your_hf_token
```

## Training

One run trains a single model under one reward setting and group size. Example,
Gemma-3-1B with the reasoning+instruction reward at G = 4:

```bash
cd training
accelerate launch --num_processes=1 --mixed_precision=bf16 main.py \
  --model-name google/gemma-3-1b-it \
  --dataset math_constraints \
  --math-constraints-path ../data/train/math_with_constraints.json \
  --out-dir-suffix reasoning_instruction_G4 \
  --num-generations 4 \
  --per-dev-batch-size 2 --grad-accum-steps 32 --gen-batchsize 32 \
  --vllm-mode colocate --vllm-gpu-memory 0.3 \
  --deepspeed-config deepspeed_config/deepspeed_zero2_no_offload.json \
  --optim adamw_torch_fused \
  --no-hf-upload
```

For the plain reasoning reward, use `--dataset dapo` and `--out-dir-suffix reasoning_G4`.
Set `--num-generations 8` for the G = 8 variants. The trained model is saved to
`../models/<model>-GRPO-<out-dir-suffix>`.

Key flags: `--dataset {dapo,math_constraints}`, `--num-generations`, `--kl-coeff`,
`--dynamic-g` (adaptive group size), `--no-constraint` (drop constraint text from the
prompt). Run `python main.py --help` for the full list.

## Evaluation

The evaluation scripts load a trained checkpoint (or a base model id), generate on each
benchmark, score, and write a per-model result file. `evaluation/pass1` and
`evaluation/pass4` then parse those result files into CSVs:

```bash
python evaluation/pass1/parse_results.py <results_dir> > pass1_all.csv
python evaluation/pass4/parse_pass4.py   <results_dir> > pass4_all.csv
```

SORRY-Bench scoring needs the judge model cached; once it is, run offline
(`HF_HUB_OFFLINE=1`) to avoid gated-endpoint lookups.

## Results

Full numbers are in `results/pass1_all.csv` and `results/pass4_all.csv`, rendered in
`results/results.html`. IFEval cells are `strict, loose`; MathIF reports strict
accuracy, loose accuracy, and correctness with and without the constraint. All values
are percentages.

A consistent effect across model families: the reasoning+instruction reward improves
MathIF constraint-following over the base model, with the largest gains at G = 8.

## Notes

Training was run on NCSA Delta (one H200 GPU per run, bf16, DeepSpeed ZeRO-2, vLLM in
colocate mode). Results depend on sampling (pass@4 uses temperature 0.7, seed 0).

## Citation

Paper in preparation. Please check back for a citation, or open an issue to reference
this work.
