"""Compare generated audio (fiwGAN/ciwGAN outputs) to original syllables.

Metrics implemented:
- Spectral MSE between mel-spectrograms.
- Intensity envelope correlation.
- (Optional) VOT difference if precomputed CSV provided.

Matching strategy:
- Default: exact stem match (generated stem == prefix of real file stem).
- Optional fuzzy matching: normalize stems (strip parens, punctuation, diacritics) and
    use closest match with difflib when no exact candidate is found.

Usage:
    .\.venv_gpu\Scripts\python.exe tools\compare_generated.py \
            --real-root Vietnamese --gen-root runs/generated_samples \
            --out runs/compare/generated_vs_real.csv --vot-csv runs/vot.csv \
            --fuzzy-match --min-sim 0.6
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import unicodedata
import difflib
import numpy as np
import librosa

SR = 16000
N_FFT = 1024
HOP = 256
MELS = 128


def load_audio(path: Path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y

def mel_spec(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=MELS)
    return librosa.power_to_db(S + 1e-9, ref=np.max)


def intensity_envelope(y):
    frame = int(0.010 * SR)
    hop = int(0.005 * SR)
    env = []
    for i in range(0, len(y) - frame, hop):
        seg = y[i:i+frame]
        env.append(np.sqrt(np.mean(seg * seg) + 1e-12))
    env = np.asarray(env)
    return (env - env.mean()) / (env.std() + 1e-9)


def spectral_mse(a, b):
    m = min(a.shape[1], b.shape[1])
    return float(np.mean((a[:, :m] - b[:, :m]) ** 2))


def corr(a, b):
    m = min(len(a), len(b))
    if m < 3:
        return 0.0
    aa = a[:m]
    bb = b[:m]
    return float(np.corrcoef(aa, bb)[0, 1])


def read_vot_csv(path: Path):
    data = {}
    if not path.exists():
        return data
    import csv as _csv
    with path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            data[row["rel_path"]] = row
    return data


IPA_MAP = {
    # Common IPA-to-ASCII approximations seen in filenames
    "ɛ": "e",
    "ŋ": "ng",
    "ɔ": "o",
    "ɯ": "u",
    "ɤ": "o",
}


def strip_parens(s: str) -> str:
    return re.sub(r"\([^)]*\)", "", s)


def remove_diacritics(s: str) -> str:
    # Decompose unicode and remove combining marks
    nfkd = unicodedata.normalize("NFKD", s)
    out = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Map a few IPA symbols to rough ASCII
    out = "".join(IPA_MAP.get(ch, ch) for ch in out)
    return out


def normalize_stem(s: str) -> str:
    s = s.lower()
    s = strip_parens(s)
    s = remove_diacritics(s)
    # Remove non-alnum characters
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def build_real_index(real_files: list[Path]) -> tuple[dict[str, list[Path]], list[tuple[str, Path]]]:
    """Return (exact_index, normalized_list) for lookup.

    exact_index maps raw stems to list of paths.
    normalized_list is a list of (normalized_stem, path) pairs for fuzzy search.
    """
    exact: dict[str, list[Path]] = {}
    norm_list: list[tuple[str, Path]] = []
    for p in real_files:
        stem = p.stem
        exact.setdefault(stem, []).append(p)
        norm_list.append((normalize_stem(stem), p))
    return exact, norm_list


def find_match_for_gen(
    gf: Path,
    exact_index: dict[str, list[Path]],
    norm_list: list[tuple[str, Path]],
    real_root: Path,
    fuzzy: bool,
    min_sim: float,
) -> tuple[Path | None, str, float]:
    """Return (matched_path, mode, score). mode in {exact, prefix, fuzzy, none}"""
    stem = gf.stem
    # 1) exact stem match
    if stem in exact_index:
        return exact_index[stem][0], "exact", 1.0
    # 2) prefix match (either direction)
    prefix_candidates = [
        p
        for s, paths in exact_index.items()
        if s.startswith(stem) or stem.startswith(s)
        for p in paths
    ]
    if prefix_candidates:
        return prefix_candidates[0], "prefix", 1.0
    # 3) fuzzy match on normalized stems
    if fuzzy:
        ng = normalize_stem(stem)
        # Build a mapping norm->paths and candidate list
        norm_to_paths: dict[str, list[Path]] = {}
        for ns, p in norm_list:
            norm_to_paths.setdefault(ns, []).append(p)
        choices = list(norm_to_paths.keys())
        if choices:
            best = difflib.get_close_matches(ng, choices, n=1, cutoff=min_sim)
            if best:
                match_ns = best[0]
                return norm_to_paths[match_ns][0], "fuzzy", difflib.SequenceMatcher(None, ng, match_ns).ratio()
    return None, "none", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", required=True)
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vot-csv", default=None, help="Optional VOT CSV to augment metrics")
    ap.add_argument("--fuzzy-match", action="store_true", help="Enable fuzzy matching by normalized stems when exact match fails")
    ap.add_argument("--min-sim", type=float, default=0.6, help="Minimum similarity (0-1) for fuzzy matches")
    args = ap.parse_args()

    real_root = Path(args.real_root)
    gen_root = Path(args.gen_root)
    vot_map = read_vot_csv(Path(args.vot_csv)) if args.vot_csv else {}

    rows = []
    gen_files = list(gen_root.rglob("*.wav"))
    real_files = list(real_root.rglob("*.wav"))
    exact_index, norm_list = build_real_index(real_files)

    for gf in gen_files:
        # Attempt to find matching real file
        real, mode, score = find_match_for_gen(
            gf, exact_index, norm_list, real_root, args.fuzzy_match, args.min_sim
        )
        if real is None:
            continue
        y_real = load_audio(real)
        y_gen = load_audio(gf)
        ms_real = mel_spec(y_real)
        ms_gen = mel_spec(y_gen)
        mse_val = spectral_mse(ms_real, ms_gen)
        inten_corr = corr(intensity_envelope(y_real), intensity_envelope(y_gen))
        vot_diff = ""
        if vot_map:
            # Use rel paths if available
            rp = str(real.relative_to(real_root))
            gp = str(gf.relative_to(gen_root))
            if rp in vot_map and gp in vot_map:
                try:
                    diff = abs(float(vot_map[rp]["vot_ms"]) - float(vot_map[gp]["vot_ms"]))
                    vot_diff = f"{diff:.2f}"
                except Exception:
                    vot_diff = ""
        rows.append({
            "generated_rel_path": str(gf.relative_to(gen_root)),
            "real_rel_path": str(real.relative_to(real_root)),
            "spectral_mse": f"{mse_val:.6f}",
            "intensity_corr": f"{inten_corr:.4f}",
            "vot_abs_diff_ms": vot_diff,
            "match_mode": mode,
            "match_score": f"{score:.3f}",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_rel_path",
                "real_rel_path",
                "spectral_mse",
                "intensity_corr",
                "vot_abs_diff_ms",
                "match_mode",
                "match_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compared {len(rows)} generated files -> {out}")

if __name__ == "__main__":
    main()
