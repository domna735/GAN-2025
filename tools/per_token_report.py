"""Compute per-token accuracy and basic distributions from predictions CSV.

Usage:
    python tools/per_token_report.py runs/logreg_vn_predictions_acoustic.csv
"""
from pathlib import Path
import sys
import csv
from collections import defaultdict

def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python tools/per_token_report.py <predictions_csv>")
        return 1
    fp = Path(argv[0])
    if not fp.exists():
        print("File not found:", fp)
        return 1

    per = defaultdict(lambda: {"n":0, "correct":0, "true_1":0, "true_0":0})
    with fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            token = row.get("token","")
            true = int(row.get("true_label", "0"))
            pred = int(row.get("pred_label", "0"))
            per[token]["n"] += 1
            if true == pred:
                per[token]["correct"] += 1
            if true == 1:
                per[token]["true_1"] += 1
            else:
                per[token]["true_0"] += 1

    out_fp = fp.parent / (fp.stem + "_per_token.csv")
    with out_fp.open("w", encoding="utf-8") as f:
        f.write("token,n,correct,accuracy,true_0,true_1\n")
        for token, stats in sorted(per.items(), key=lambda x: -x[1]["n"]):
            n = stats["n"]
            acc = stats["correct"] / n if n>0 else 0.0
            f.write(f"{token},{n},{stats['correct']},{acc:.3f},{stats['true_0']},{stats['true_1']}\n")

    print("Wrote per-token report:", out_fp)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
