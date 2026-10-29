#!/usr/bin/env python3
"""
Render CSV files as clean aligned tables for easy hand-entry into a report.
Empty cells (like Sorry, to be filled later) show as "____".

Usage:
  python view_csv.py                  # render all *.csv in this folder
  python view_csv.py gemma_pass1.csv  # render specific file(s)
  python view_csv.py -t gemma_pass1.csv   # transposed: one block per model
"""
import csv, sys, glob, os

BLANK = "____"

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

def grid(path):
    header, rows = load(path)
    cols = list(zip(*([header] + rows))) if rows else [header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            c = c if c != "" else BLANK
            out.append(str(c).ljust(widths[i]))
        return " | ".join(out)
    line = "-+-".join("-" * w for w in widths)
    print(f"\n### {os.path.basename(path)}\n")
    print(fmt(header))
    print(line)
    for r in rows:
        print(fmt(r))
    print()

def transpose(path):
    header, rows = load(path)
    mw = max(len(h) for h in header)
    print(f"\n### {os.path.basename(path)} (per model)\n")
    for r in rows:
        name = r[0]
        print("=" * len(name)); print(name); print("=" * len(name))
        for h, v in zip(header[1:], r[1:]):
            v = v if v != "" else BLANK
            print(f"  {h.ljust(mw)} : {v}")
        print()

def main():
    args = sys.argv[1:]
    t = "-t" in args
    args = [a for a in args if a != "-t"]
    files = args if args else sorted(glob.glob("*.csv"))
    if not files:
        print("no csv files found"); return
    for f in files:
        (transpose if t else grid)(f)

if __name__ == "__main__":
    main()
