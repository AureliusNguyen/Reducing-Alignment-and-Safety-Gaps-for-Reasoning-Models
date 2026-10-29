#!/usr/bin/env python3
"""
Turn the per-model result_<tag>.txt files into one CSV row each, matching the
scoresheet columns:
  Model | IFEval Prompt | IFEval Instruct | AIME | Sorry |
  MathIF Strict acc | MathIF Loose acc | MathIF Correct w const | MathIF Correct w/o const | SafetyBench

Usage:  python parse_results.py [results_dir]   (default: ./results)
Prints CSV to stdout.  Missing values show as "" (that benchmark failed for that model).
"""
import re, os, glob, csv, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def parse(path):
    L = [l.rstrip("\n") for l in open(path)]
    txt = "\n".join(L)
    r = {}

    # AIME (first "accuracy: X" line, written by aime.py)
    m = re.search(r"accuracy:\s*([0-9.]+)", txt)
    r["AIME"] = round(float(m.group(1)) * 100, 2) if m else ""

    # IFEval strict + loose, prompt-level + instruction-level
    def grab(marker):
        for i, l in enumerate(L):
            if marker in l:
                pl = next((x for x in L[i:i + 6] if "prompt-level" in x), None)
                il = next((x for x in L[i:i + 6] if "instruction-level" in x), None)
                p = round(float(pl.split(":")[1]) * 100, 2) if pl else ""
                inst = round(float(il.split(":")[1]) * 100, 2) if il else ""
                return p, inst
        return "", ""
    sp, si = grab("eval_results_strict")
    lp, li = grab("eval_results_loose")
    r["IFEval_Prompt"] = f"{sp}, {lp}" if sp != "" else ""
    r["IFEval_Instruct"] = f"{si}, {li}" if si != "" else ""

    # MathIF: blocks of (header, strict, loose, correct); noconstraint blocks have
    # strict=loose=1.0 and give the "correct w/o constraint" baseline.
    try:
        start = next(i for i, l in enumerate(L) if "MathIF Evaluation" in l)
    except StopIteration:
        start = len(L)
    i = start + 1
    con = {"s": [], "l": [], "c": []}
    noc = []
    while i < len(L):
        s = L[i].strip()
        if s.startswith("===") and i > start:
            break
        if s and not re.match(r"^-?[0-9.]+$", s):
            try:
                a, b, c = float(L[i + 1]), float(L[i + 2]), float(L[i + 3])
            except (ValueError, IndexError):
                i += 1
                continue
            if s.endswith("_noconstraint"):
                noc.append(c)
            else:
                con["s"].append(a); con["l"].append(b); con["c"].append(c)
            i += 4
        else:
            i += 1
    mean = lambda x: round(100 * sum(x) / len(x), 2) if x else ""
    r["MathIF_Strict"] = mean(con["s"])
    r["MathIF_Loose"] = mean(con["l"])
    r["MathIF_Correct_wc"] = mean(con["c"])
    r["MathIF_Correct_woc"] = mean(noc)

    # Sorry (non-fulfillment / safe rate as computed by score_output.py)
    m = re.search(r"Percentage of score 1:\s*([0-9.]+)", txt)
    r["Sorry"] = m.group(1) if m else ""

    # SafetyBench average
    m = re.search(r"'Avg\.':\s*(?:np\.float64\()?(?:np\.float64\()?([0-9.]+)", txt)
    r["SafetyBench"] = round(float(m.group(1)), 2) if m else ""
    return r


COLS = ["Model", "IFEval_Prompt", "IFEval_Instruct", "AIME", "Sorry",
        "MathIF_Strict", "MathIF_Loose", "MathIF_Correct_wc", "MathIF_Correct_woc",
        "SafetyBench"]

w = csv.DictWriter(sys.stdout, fieldnames=COLS)
w.writeheader()
for p in sorted(glob.glob(os.path.join(RES, "*.txt"))):
    row = parse(p)
    row["Model"] = os.path.basename(p)[:-4]
    w.writerow(row)
