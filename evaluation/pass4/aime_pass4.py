"""
AIME pass@k evaluation (default k=4).

Derived from all-benchmark-code/aime/aime.py. Only change: generate k samples
per problem (SamplingParams n=k) and score a problem correct if ANY of the k
normalized predictions matches gold (pass@k). Also reports pass@1 (first sample)
for reference.

Run from inside a copy of the eval repo's aime/ dir (needs normalize_answer's deps).
"""
import argparse
import os
import re
from typing import List, Tuple
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^Answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(". ")
    boxed_match = re.search(r"\\\(\s*\\boxed\{([^}]*)\}\s*\\\)", cleaned)
    if boxed_match:
        cleaned = boxed_match.group(1).strip()
    else:
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", cleaned)
        if boxed_match:
            cleaned = boxed_match.group(1).strip()
    cleaned = re.sub(r"\\[()$]", "", cleaned)
    cleaned = cleaned.strip()
    digits = re.findall(r"-?\d+", cleaned)
    if digits and len(cleaned) <= 10:
        return digits[-1].lstrip("+")
    return cleaned.lower()


def generate_samples(llm, tokenizer, problem: str, max_new_tokens: int,
                     n: int, temperature: float) -> List[str]:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the following problem."},
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_new_tokens, n=n,
    )
    outputs = llm.generate([text], sampling_params)
    # outputs[0].outputs is a list of n completions
    return [o.text for o in outputs[0].outputs]


def evaluate(csv_path, model_name, max_model_len, max_new_tokens, n, temperature):
    df = pd.read_csv(csv_path)
    if not {"problem", "answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'problem' and 'answer' columns")

    llm = LLM(model=model_name, dtype="auto", gpu_memory_utilization=0.9,
              max_model_len=max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    total = 0
    correct_pass_k = 0   # any of n correct
    correct_pass_1 = 0   # first sample correct

    for idx, row in df.iterrows():
        problem = str(row["problem"]) if pd.notna(row["problem"]) else ""
        gold = str(row["answer"]) if pd.notna(row["answer"]) else ""
        gold_norm = normalize_answer(gold)

        samples = generate_samples(llm, tokenizer, problem, max_new_tokens, n, temperature)
        norms = [normalize_answer(s) for s in samples]

        if norms and norms[0] == gold_norm:
            correct_pass_1 += 1
        if any(nm == gold_norm for nm in norms):
            correct_pass_k += 1
        total += 1

        if total % 10 == 0:
            print(f"Processed {total}... pass@{n}: {correct_pass_k/total:.2%} | pass@1: {correct_pass_1/total:.2%}")

    return total, correct_pass_1, correct_pass_k


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-model-len", type=int, default=32000)
    p.add_argument("--max-new-tokens", type=int, default=25000)
    p.add_argument("--csv-file", type=str, default="./aime_dataset.csv")
    p.add_argument("--n", type=int, default=4, help="samples per problem (pass@n)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature; needs to be >0 for diverse samples")
    p.add_argument("--accuracy-output", type=str, default="result_aime_pass4.txt")
    args = p.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, os.path.basename(args.accuracy_output))

    total, c1, ck = evaluate(args.csv_file, args.model, args.max_model_len,
                             args.max_new_tokens, args.n, args.temperature)
    acc_k = ck / total if total else 0.0
    acc_1 = c1 / total if total else 0.0
    print(f"\nAIME pass@{args.n} complete. total={total} "
          f"pass@{args.n}={acc_k:.4f} pass@1={acc_1:.4f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("aime pass@k evaluation result\n")
        f.write(f"model: {args.model}\n")
        f.write(f"n_samples: {args.n}\n")
        f.write(f"temperature: {args.temperature}\n")
        f.write(f"max_model_len: {args.max_model_len}\n")
        f.write(f"max_new_tokens: {args.max_new_tokens}\n")
        f.write(f"total: {total}\n")
        f.write(f"pass@{args.n}: {acc_k:.6f}\n")
        f.write(f"pass@1: {acc_1:.6f}\n")
