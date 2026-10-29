"""
IFEval pass@k generation: generate k samples per prompt and write k response
files (ifeval_responses_0.jsonl ... ifeval_responses_{k-1}.jsonl), each in the
exact single-response format the google-research scorer expects. Scoring/OR is
done by score_ifeval_pass4.py.

Run from the eval-repo root (same cwd as ifeval.py uses).
"""
import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams

INPUT_DATA = "google-research/instruction_following_eval/data/input_data.jsonl"
OUT_TMPL = "google-research/instruction_following_eval/data/ifeval_responses_{}.jsonl"


def run(model_name, max_model_len, max_tokens, n, temperature):
    llm = LLM(model=model_name, trust_remote_code=True, max_model_len=max_model_len)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    dataset = load_dataset("json", data_files=INPUT_DATA)["train"]

    files = [open(OUT_TMPL.format(i), "w", encoding="utf-8") for i in range(n)]
    try:
        for example in dataset:
            prompt = example["prompt"]
            outputs = llm.generate([prompt], sp)
            comps = outputs[0].outputs  # n completions
            for i in range(n):
                text = comps[i].text.strip() if i < len(comps) else ""
                text = text.split("</think>", 1)[-1].strip()
                files[i].write(json.dumps({"prompt": prompt, "response": text},
                                          ensure_ascii=False) + "\n")
    finally:
        for f in files:
            f.close()
    print(f"IFEval: wrote {n} response files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-model-len", type=int, default=32000)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()
    run(a.model, a.max_model_len, a.max_tokens, a.n, a.temperature)
