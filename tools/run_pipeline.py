"""Convenience runner to chain manifest, VOT, generation (optional), and comparison.

Example (PowerShell):
  .\.venv_gpu\Scripts\python.exe tools\run_pipeline.py \
    --real-root Vietnamese \
    --spec-input processed_data \
    --gen-out runs/generated_samples \
    --manifest-out manifest/manifest.csv \
    --vot-real-out runs/vot_real.csv \
    --vot-gen-out runs/vot_gen.csv \
    --compare-out runs/compare/generated_vs_real.csv \
    --generate --limit 20 --fuzzy-match

Notes:
- If --generate is omitted, the script skips spectrogram inversion and only runs other steps.
- Each step gracefully skips on missing inputs and reports what ran.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], desc: str) -> None:
    print(f"\n[run] {desc}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[warn] step failed ({desc}): {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", required=True)
    ap.add_argument("--spec-input", default=None, help="Directory with .npy spectrograms to invert")
    ap.add_argument("--gen-out", default="runs/generated_samples")
    ap.add_argument("--manifest-out", default="manifest/manifest.csv")
    ap.add_argument("--vot-real-out", default="runs/vot_real.csv")
    ap.add_argument("--vot-gen-out", default="runs/vot_gen.csv")
    ap.add_argument("--compare-out", default="runs/compare/generated_vs_real.csv")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="Limit generation to N files (0=all)")
    ap.add_argument("--generate", action="store_true", help="Run spectrogram inversion step")
    ap.add_argument("--fuzzy-match", action="store_true", help="Enable fuzzy matching in comparison")
    ap.add_argument("--min-sim", type=float, default=0.6)
    args = ap.parse_args()

    py = sys.executable
    tools = Path(__file__).parent

    # 1) Manifest for real data
    run_step(
        [py, str(tools / "build_manifest.py"), "--root", args.real_root, "--out", args.manifest_out],
        "build manifest",
    )

    # 2) VOT for real data
    run_step(
        [py, str(tools / "compute_vot.py"), "--root", args.real_root, "--out", args.vot_real_out, "--ext", ".wav"],
        "compute VOT (real)",
    )

    # 3) Optional generation from spectrograms
    gen_out = Path(args.gen_out)
    if args.generate:
        if not args.spec_input:
            print("[warn] --generate specified but --spec-input is missing; skipping generation")
        else:
            run_step(
                [
                    py,
                    str(tools / "generate_from_spectrogram.py"),
                    "--input-dir",
                    args.spec_input,
                    "--out-dir",
                    str(gen_out),
                    "--sr",
                    str(args.sr),
                    "--n_iter",
                    str(args.n_iter),
                    "--limit",
                    str(args.limit),
                ],
                "generate from spectrograms",
            )

    # 4) VOT for generated (if any wav exists)
    if gen_out.exists() and any(gen_out.rglob("*.wav")):
        run_step(
            [py, str(tools / "compute_vot.py"), "--root", str(gen_out), "--out", args.vot_gen_out, "--ext", ".wav"],
            "compute VOT (generated)",
        )
        vot_csv = args.vot_real_out  # By default compare uses real VOT; pass gen VOT too if desired.
    else:
        vot_csv = args.vot_real_out

    # 5) Compare generated vs real
    if gen_out.exists() and any(gen_out.rglob("*.wav")):
        cmd = [
            py,
            str(tools / "compare_generated.py"),
            "--real-root",
            args.real_root,
            "--gen-root",
            str(gen_out),
            "--out",
            args.compare_out,
            "--vot-csv",
            vot_csv,
        ]
        if args.fuzzy_match:
            cmd += ["--fuzzy-match", "--min-sim", str(args.min_sim)]
        run_step(cmd, "compare generated vs real")
    else:
        print(f"[info] No generated wavs found under {gen_out}; skipping comparison")


if __name__ == "__main__":
    main()
