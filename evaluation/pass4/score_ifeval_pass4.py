"""
IFEval pass@k scorer. ifeval_pass4.py wrote k response files
(ifeval_responses_0..{k-1}.jsonl). For each, run the existing google-research
scorer (which writes per-prompt eval_results_strict/loose.jsonl), then OR across
the k samples:
  - prompt-level (strict/loose) = mean over prompts of any(follow_all_instructions)
  - instruction-level (strict/loose) = mean over prompts of
    max_i(sum(follow_instruction_list_i)/len)

Prints 4 numbers in the same shape the pass@1 IFEval report uses, so parsing is
trivial:
  IFEVAL_PASS4 strict_prompt strict_instr loose_prompt loose_instr

Run from the eval-repo root. Requires google-research/ on the path (cwd trick same
as run_evaluations.run_ifeval_evaluation).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

GR = "google-research"
DATA = os.path.join(GR, "instruction_following_eval", "data")
RESP_LIVE = os.path.join(DATA, "ifeval_responses.jsonl")  # what the scorer reads


def read_results(strict_or_loose):
    path = os.path.join(DATA, f"eval_results_{strict_or_loose}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_one_sample(resp_file):
    """Run the scorer on one response file; return per-prompt dicts for strict+loose."""
    shutil.copy(resp_file, RESP_LIVE)
    subprocess.run(
        [sys.executable, "-m", "instruction_following_eval.evaluation_main",
         "--input_data=./instruction_following_eval/data/input_data.jsonl",
         "--input_response_data=./instruction_following_eval/data/ifeval_responses.jsonl",
         "--output_dir=./instruction_following_eval/data/"],
        cwd=GR, check=True, capture_output=True, text=True,
    )
    return read_results("strict"), read_results("loose")


def frac(follow_list):
    return sum(1 for x in follow_list if x) / len(follow_list) if follow_list else 0.0


def main():
    resp_files = sorted(glob.glob(os.path.join(DATA, "ifeval_responses_*.jsonl")))
    if not resp_files:
        print("IFEVAL_PASS4 ERROR no response files", flush=True)
        return

    # per-prompt aggregates keyed by prompt text
    strict_any = {}   # prompt -> bool any sample followed ALL
    loose_any = {}
    strict_best_frac = {}  # prompt -> best instruction-level fraction (strict file)
    loose_best_frac = {}

    for rf in resp_files:
        s_rows, l_rows = score_one_sample(rf)
        for r in s_rows:
            p = r["prompt"]
            strict_any[p] = strict_any.get(p, False) or bool(r["follow_all_instructions"])
            strict_best_frac[p] = max(strict_best_frac.get(p, 0.0), frac(r["follow_instruction_list"]))
        for r in l_rows:
            p = r["prompt"]
            loose_any[p] = loose_any.get(p, False) or bool(r["follow_all_instructions"])
            loose_best_frac[p] = max(loose_best_frac.get(p, 0.0), frac(r["follow_instruction_list"]))

    def mean(d):
        return sum(1 if v is True else v for v in d.values()) / len(d) if d else 0.0

    sp = mean(strict_any)              # strict prompt-level pass@k
    lp = mean(loose_any)               # loose prompt-level pass@k
    si = mean(strict_best_frac)        # strict instruction-level pass@k
    li = mean(loose_best_frac)         # loose instruction-level pass@k
    print(f"IFEVAL_PASS4 strict_prompt={sp:.6f} strict_instr={si:.6f} "
          f"loose_prompt={lp:.6f} loose_instr={li:.6f}", flush=True)


if __name__ == "__main__":
    main()
