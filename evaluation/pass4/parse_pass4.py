#!/usr/bin/env python3
"""
Parse pass@4 result files into a CSV. Columns:
  Model | AIME_pass4 | AIME_pass1 | IFEval_Prompt_pass4 | IFEval_Instruct_pass4 |
  MathIF_Strict4 | MathIF_Loose4 | MathIF_Correct_wc4 | MathIF_Correct_woc4 |
  SafetyBench_pass4 | Sorry_safe_pass4
Usage: python parse_pass4.py [results_dir]   (default ./results). CSV to stdout.
"""
import re, os, glob, csv, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def parse(path):
    L = [l.rstrip("\n") for l in open(path)]
    txt = "\n".join(L)
    r = {}

    m = re.search(r"pass@4:\s*([0-9.]+)", txt)
    r["AIME_pass4"] = round(float(m.group(1)) * 100, 2) if m else ""
    m = re.search(r"pass@1:\s*([0-9.]+)", txt)
    r["AIME_pass1"] = round(float(m.group(1)) * 100, 2) if m else ""

    # IFEval: "IFEVAL_PASS4 strict_prompt=.. strict_instr=.. loose_prompt=.. loose_instr=.."
    m = re.search(r"IFEVAL_PASS4 strict_prompt=([0-9.]+) strict_instr=([0-9.]+) "
                  r"loose_prompt=([0-9.]+) loose_instr=([0-9.]+)", txt)
    if m:
        sp, si, lp, li = (round(float(x) * 100, 2) for x in m.groups())
        r["IFEval_Prompt_pass4"] = f"{sp}, {lp}"
        r["IFEval_Instruct_pass4"] = f"{si}, {li}"
    else:
        r["IFEval_Prompt_pass4"] = ""
        r["IFEval_Instruct_pass4"] = ""

    # MathIF block (any-of-4). Skip non-numeric noise lines (e.g. langdetect errors).
    try:
        start = next(i for i, l in enumerate(L) if "MathIF pass@" in l)
    except StopIteration:
        start = len(L)
    i = start + 1
    con = {"s": [], "l": [], "c": []}
    noc = []
    while i < len(L):
        s = L[i].strip()
        if s.startswith("===") and i > start:
            break
        # a benchmark block header = a non-numeric line that is NOT stderr noise
        is_num = bool(re.match(r"^-?[0-9.]+$", s))
        is_noise = ("ERROR" in s) or ("WARNING" in s) or (not s)
        if s and not is_num and not is_noise:
            # collect the next 3 numeric lines, skipping noise between them
            nums = []
            j = i + 1
            while j < len(L) and len(nums) < 3:
                t = L[j].strip()
                if re.match(r"^-?[0-9.]+$", t):
                    nums.append(float(t))
                elif ("ERROR" in t) or ("WARNING" in t) or (not t):
                    pass  # skip stderr noise
                else:
                    break  # hit the next header unexpectedly
                j += 1
            if len(nums) == 3:
                a, b, c = nums
                if s.endswith("_noconstraint"):
                    noc.append(c)
                else:
                    con["s"].append(a); con["l"].append(b); con["c"].append(c)
                i = j
                continue
        i += 1
    mean = lambda x: round(100 * sum(x) / len(x), 2) if x else ""
    r["MathIF_Strict4"] = mean(con["s"])
    r["MathIF_Loose4"] = mean(con["l"])
    r["MathIF_Correct_wc4"] = mean(con["c"])
    r["MathIF_Correct_woc4"] = mean(noc)

    # SafetyBench (dict, handles np.float64(...))
    m = re.search(r"'Avg\.':\s*(?:np\.float64\()?([0-9.]+)", txt)
    r["SafetyBench_pass4"] = round(float(m.group(1)), 2) if m else ""

    # Sorry: written by the separate sorry run to <base>.sorry.txt
    r["Sorry_safe_pass4"] = ""
    sidecar = path[:-4] + ".sorry.txt"
    if os.path.exists(sidecar):
        st = open(sidecar).read()
        m = re.search(r"SORRY_SAFE_PASS4\s*([0-9.]+)", st)
        if m:
            r["Sorry_safe_pass4"] = round(float(m.group(1)), 2)
    return r


# Column order matches the pass@1 scoresheet exactly (IFEval, AIME, Sorry, MathIF..., SafetyBench)
# so the CSV pastes straight into the same-shape pass@4 sheet. AIME_pass1 kept last as reference.
COLS = ["Model", "IFEval_Prompt_pass4", "IFEval_Instruct_pass4", "AIME_pass4",
        "Sorry_safe_pass4", "MathIF_Strict4", "MathIF_Loose4", "MathIF_Correct_wc4",
        "MathIF_Correct_woc4", "SafetyBench_pass4", "AIME_pass1"]
w = csv.DictWriter(sys.stdout, fieldnames=COLS)
w.writeheader()
for p in sorted(glob.glob(os.path.join(RES, "*.txt"))):
    if p.endswith(".sorry.txt"):
        continue  # sidecar, merged into its base model's row
    row = parse(p)
    row["Model"] = os.path.basename(p)[:-4]
    w.writerow(row)
