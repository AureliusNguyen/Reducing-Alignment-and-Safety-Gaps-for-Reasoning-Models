"""
Score the 4 non-AIME benchmarks for the model configured in run_benchmarks.py.
(AIME is scored inline by run_benchmarks.py itself.)
Each scorer is wrapped so one failure does not block the others.
Run from inside an isolated eval-repo copy, with PYTHONPATH including sorry/FastChat
and the SafetyBench processed file already copied into place.
"""
import run_evaluations as R

STEPS = [
    ("IFEval", R.run_ifeval_evaluation),
    ("MathIF", R.run_mathif_evaluation),
    ("Sorry", R.run_sorry_evaluation),
    ("SafetyBench", R.run_safetybench_evaluation),
]

for name, fn in STEPS:
    try:
        out = fn()
        with open(R.RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== {name} Evaluation ===\n")
            f.write(out)
            if not out.endswith("\n"):
                f.write("\n")
        print(f"[OK] scored {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
