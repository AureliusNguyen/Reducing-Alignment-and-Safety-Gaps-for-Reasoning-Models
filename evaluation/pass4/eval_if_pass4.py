"""
MathIF pass@k scorer (default k = whatever the generation produced).

Derived from all-benchmark-code/MathIF/code/eval_if.py. Difference: the generation
step writes {"output": [s1, s2, ... sN]} (N samples). This scorer marks a problem
'correct' if ANY of the N samples passes compute_score (pass@N math accuracy).
Strict/loose instruction-following are reported as any-of-N as well (did any sample
obey all constraints). Prints: strict, loose, correct  (same 3-line format as the
original, so the existing parser structure still works).

Drop this next to eval_if.py inside MathIF/code and call it the same way.
"""
import argparse
import json
from prime_math import compute_score
from constraint_registry import INSTRUCTION_DICT

parser = argparse.ArgumentParser()
parser.add_argument('--hypothesis_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument("--delimiter", type=str, default="</think>")
args = parser.parse_args()


def clean(hypothesis: str) -> str:
    if '<think>' in hypothesis:
        hypothesis = hypothesis.split('<think>')[1]
    if '</think>' in hypothesis:
        hypothesis = hypothesis.split('</think>')[1]
    if '<answer>' in hypothesis:
        hypothesis = hypothesis.split('<answer>')[1]
    if '</answer>' in hypothesis:
        hypothesis = hypothesis.split('</answer>')[0]
    return hypothesis


def follows_all(sample, data, prompt):
    """True if this single sample obeys ALL constraints (strict)."""
    is_following = []
    for index, instruction_id in enumerate(data["constraint_name"]):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        params = data["constraint_args"][index]
        kwargs = {n: p for n, p in params.items() if p} if params else {}
        instruction.build_description(**kwargs)
        cargs = instruction.get_constraint_args()
        if cargs and "prompt" in cargs:
            instruction.build_description(prompt=prompt)
        try:
            ok = bool(sample.strip()) and instruction.check_following(sample)
        except Exception:
            ok = False
        is_following.append(ok)
    return is_following


strict, loose, correct = [], [], []
noconstraint = "noconstraint" in args.hypothesis_path

for line1, line2 in zip(open(args.hypothesis_path).readlines(),
                        open(args.data_path).readlines()):
    rec = json.loads(line1)
    samples = rec.get("output", rec.get("response"))
    if not isinstance(samples, list):
        samples = [samples]
    samples = [clean(s) for s in samples]
    data = json.loads(line2)

    # math correctness: pass@N = any sample correct
    any_correct = 0
    for s in samples:
        try:
            if compute_score(s, data['answer'])[0]:
                any_correct = 1
                break
        except Exception:
            pass
    correct.append(any_correct)

    # instruction following: any-of-N
    if not noconstraint:
        best_strict = 0
        best_loose = 0.0
        for s in samples:
            fl = follows_all(s, data, data["question"])
            best_strict = max(best_strict, 1 if all(fl) else 0)
            best_loose = max(best_loose, sum(fl) / len(fl) if fl else 0.0)
        strict.append(best_strict)
        loose.append(best_loose)
    else:
        strict.append(1)
        loose.append(1)

print(sum(strict) / len(strict))
print(sum(loose) / len(loose))
print(sum(correct) / len(correct))
