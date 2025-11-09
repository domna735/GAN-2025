"""Simple per-token plotting utility.

Reads a predictions CSV produced by `logistic_regression_vietnamese.py` and writes
- `<predictions>_per_token.csv` with counts and accuracy per token
- `<lang>_<mode>_per_token.png` bar plot saved to `runs/plots/`

Usage:
    python tools/plot_per_token.py runs/logreg_vn_predictions_acoustic.csv
"""
from pathlib import Path
import csv
from collections import defaultdict
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def run(pred_csv_path: Path):
    if not pred_csv_path.exists():
        print(f"Predictions file not found: {pred_csv_path}")
        return
    rows = []
    with pred_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    per = defaultdict(lambda: {"total":0, "correct":0, "short":0, "long":0})
    for r in rows:
        tok = r["token"]
        true = int(r["true_label"]) if r.get("true_label") is not None else int(r.get("true",0))
        pred = int(r["pred_label"]) if r.get("pred_label") is not None else int(r.get("pred",0))
        per[tok]["total"] += 1
        if true == pred:
            per[tok]["correct"] += 1
        if true == 0:
            per[tok]["short"] += 1
        else:
            per[tok]["long"] += 1

    out_csv = pred_csv_path.parent / (pred_csv_path.stem + "_per_token.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token","total","correct","accuracy","short_count","long_count"])
        for tok, d in sorted(per.items(), key=lambda x: -x[1]["total"]):
            acc = d["correct"] / d["total"] if d["total"]>0 else 0.0
            writer.writerow([tok, d["total"], d["correct"], f"{acc:.3f}", d["short"], d["long"]])
    print(f"Wrote per-token CSV -> {out_csv}")

    # plotting
    if HAS_MATPLOTLIB:
        labels = [tok for tok, _ in sorted(per.items(), key=lambda x: -x[1]["total"])]
        accs = [per[t]["correct"]/per[t]["total"] for t in labels]
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.25), 4))
        ax.bar(range(len(labels)), accs)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0,1)
        ax.set_title(pred_csv_path.stem + " per-token accuracy")
        plots_dir = pred_csv_path.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        out_png = plots_dir / (pred_csv_path.stem + "_per_token.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    else:
        print("matplotlib not available; skipping plot generation.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_per_token.py <predictions.csv>")
    else:
        run(Path(sys.argv[1]))
