"""Logistic regression for Vietnamese subset.

This script builds a dataset from the Vietnamese annotation folders under
`vowel_length_gan_2025-08-24/Vietnamese/Vietnamese/*` and matches those token
names to the precomputed spectrogram `.npy` files in
`vowel_length_gan_2025-08-24/processed_data`.

It creates two categorical variables:
 - duration: short (0) vs long (1) (derived from the Vietnamese folder group)
 - sound_quality: the vowel token (categorical)

Features: per-file summary statistics of the spectrogram array (mean, std, min,
max, median, 10th and 90th percentiles, mean over time, mean over frequency).

Outputs (saved to `runs/`):
 - `logreg_vn_model.pkl` (joblib pipeline)
 - `logreg_vn_metrics.json` (classification metrics)
 - `logreg_vn_predictions.csv` (filename, true_label, pred_label, pred_prob)

Usage:
    python tools/logistic_regression_vietnamese.py
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import joblib
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix

import argparse

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False
    # librosa is optional; MFCC mode will fall back if not available


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "vowel_length_gan_2025-08-24" / "processed_data"
VIET_DIR = ROOT / "vowel_length_gan_2025-08-24" / "Vietnamese" / "Vietnamese"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def gather_vietnamese_tokens(viet_dir: Path) -> dict:
    """Walk the Vietnamese folders and return mapping token -> duration_group

    duration_group is 'long' or 'short' based on the top-level folder names.
    """
    tokens = {}
    if not viet_dir.exists():
        print(f"Vietnamese folder not found at {viet_dir}")
        return tokens

    for group in viet_dir.iterdir():
        if not group.is_dir():
            continue
        group_name = group.name.lower()
        # group directories include 'long vowels-#VVT' or 'short vowels-#VVT'
        if group_name.startswith("long"):
            duration = "long"
        elif group_name.startswith("short"):
            duration = "short"
        else:
            # inside these there may be per-vowel subfolders
            # if this folder appears to be a token (no dash), assume unknown
            duration = None

        # collect tokens from child directories (token subfolders) or filenames
        for child in group.iterdir():
            if child.is_dir():
                token = child.name
                if duration is not None:
                    tokens[token] = duration
            else:
                # If the group contains files (mp3) rather than token subfolders,
                # use the filename stem as the token key (keeps trailing numbers like '55').
                try:
                    token = child.stem
                    if duration is not None:
                        tokens[token] = duration
                except Exception:
                    pass

    return tokens


def find_matching_npy_files(data_dir: Path, tokens: dict) -> list[tuple[Path,str,str]]:
    """Return list of (npy_path, token, duration_group) by substring matching token in filename."""
    all_files = sorted(data_dir.glob("*.npy"))
    results = []
    token_items = list(tokens.items())
    for p in all_files:
        name = p.name
        matched = False
        for token, duration in token_items:
            if token in name:
                results.append((p, token, duration))
                matched = True
                break
        # Optionally skip unmatched files
    return results


def find_audio_for_token(viet_dir: Path, token: str) -> Path | None:
    """Search recursively under viet_dir for a raw audio file matching token in its stem.

    Prefer WAV over MP3 when both exist.
    """
    exts = [".wav", ".mp3", ".flac", ".m4a"]
    matches = []
    for ext in exts:
        for p in viet_dir.rglob(f"*{token}*{ext}"):
            matches.append(p)
    if not matches:
        # try looser match: token normalized
        norm = token.replace(" ", "").lower()
        for ext in exts:
            for p in viet_dir.rglob(f"*{ext}"):
                if norm in p.stem.replace(" ", "").lower():
                    matches.append(p)
    if not matches:
        return None
    # prefer WAV
    for p in matches:
        if p.suffix.lower() == ".wav":
            return p
    return matches[0]


def extract_features(npy_path: Path, use_mfcc: bool = False, mfcc_from_wave: bool = False, wave_path: Path | None = None) -> np.ndarray:
    arr = np.load(npy_path)
    # Ensure numeric array
    arr = np.asarray(arr, dtype=float)
    # Flatten independent stats
    vals = []
    vals.append(np.nanmean(arr))
    vals.append(np.nanstd(arr))
    vals.append(np.nanmin(arr))
    vals.append(np.nanmax(arr))
    vals.append(np.nanmedian(arr))
    vals.append(np.nanpercentile(arr, 10))
    vals.append(np.nanpercentile(arr, 90))
    # mean over time (axis 1) and mean over frequency (axis 0) if 2D
    if arr.ndim >= 2:
        vals.append(np.nanmean(np.mean(arr, axis=0)))
        vals.append(np.nanmean(np.mean(arr, axis=1)))
    else:
        # duplicate overall mean
        vals.append(np.nanmean(arr))
        vals.append(np.nanmean(arr))
    # Option A: coarse MFCCs from the stored spectrogram (legacy)
    if use_mfcc and HAS_LIBROSA and not mfcc_from_wave:
        try:
            mfccs = librosa.feature.mfcc(S=arr, n_mfcc=13)
            mfcc_mean = np.nanmean(mfccs, axis=1)
            mfcc_std = np.nanstd(mfccs, axis=1)
            vals.extend(mfcc_mean.tolist())
            vals.extend(mfcc_std.tolist())
        except Exception:
            vals.extend([0.0] * 26)

    # Option B: compute MFCCs (and deltas) from waveform if requested and available
    if mfcc_from_wave and HAS_LIBROSA:
        try:
            if wave_path is not None and wave_path.exists():
                y, sr = librosa.load(str(wave_path), sr=16000)
            else:
                # fallback: try to infer from npy-derived arr (not ideal)
                y = None
                sr = None
            if y is not None:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                # for each of mfcc, delta, delta2 compute mean and std per coeff
                for mat in (mfcc, delta, delta2):
                    mean = np.nanmean(mat, axis=1)
                    std = np.nanstd(mat, axis=1)
                    vals.extend(mean.tolist())
                    vals.extend(std.tolist())
        except Exception:
            # on error, append zeros for 13*2*3 = 78 features
            vals.extend([0.0] * 78)
    return np.array(vals, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Vietnamese logistic regression modes")
    parser.add_argument("--mode", choices=["token", "acoustic", "grouped", "mfcc", "loog"], default="token",
                        help="token/acoustic/grouped/mfcc/loog (loog = LeaveOneGroupOut by token)")
    parser.add_argument("--mfcc-from-wave", action="store_true", help="If set, compute MFCCs from raw audio files located under the language folder (requires librosa)")
    parser.add_argument("--viet-dir", default=None, help="Optional path to a specific language folder under vowel_length_gan_2025-08-24/Vietnamese")
    args = parser.parse_args()
    mode = args.mode
    if args.viet_dir:
        viet_dir = Path(args.viet_dir)
        if not viet_dir.exists():
            print(f"Provided --viet-dir does not exist: {viet_dir}")
            sys.exit(1)
        local_viet_dir = viet_dir
    else:
        local_viet_dir = VIET_DIR
    lang_label = local_viet_dir.name

    tokens = gather_vietnamese_tokens(local_viet_dir)
    if not tokens:
        print("No Vietnamese tokens found. Aborting.")
        sys.exit(1)

    matches = find_matching_npy_files(DATA_DIR, tokens)
    if not matches:
        print("No matching .npy files found for Vietnamese tokens. Aborting.")
        sys.exit(1)

    filenames = []
    tokens_list = []
    duration_list = []
    X_list = []
    for p, token, duration in matches:
        # try to find a raw waveform for the token when requested
        wave_p = None
        if args.mfcc_from_wave:
            wave_p = find_audio_for_token(local_viet_dir, token)
        feats = extract_features(p, use_mfcc=(mode == "mfcc"), mfcc_from_wave=args.mfcc_from_wave, wave_path=wave_p)
        X_list.append(feats)
        filenames.append(p.name)
        tokens_list.append(token)
        duration_list.append(duration)

    X_num = np.vstack(X_list)
    # token column for optional categorical inclusion
    token_col = np.array(tokens_list, dtype=object).reshape(-1, 1)

    # label: short=0, long=1
    y = np.array([0 if d == "short" else 1 for d in duration_list], dtype=int)

    n_num = X_num.shape[1]

    if mode == "token":
        X_all = np.hstack([X_num, token_col])
        preproc = ColumnTransformer([
            ("num", StandardScaler(), list(range(n_num))),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [n_num]),
        ], remainder="drop")
        clf = Pipeline([
            ("preproc", preproc),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        print(f"Mode=token: running 5-fold stratified CV on {len(filenames)} samples (tokens={len(set(tokens_list))})")
        y_pred = cross_val_predict(clf, X_all, y, cv=cv, method="predict")
        y_prob = cross_val_predict(clf, X_all, y, cv=cv, method="predict_proba")

    elif mode == "acoustic":
        X_all = X_num
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        print(f"Mode=acoustic: running 5-fold stratified CV on numeric features only ({len(filenames)} samples)")
        y_pred = cross_val_predict(clf, X_all, y, cv=cv, method="predict")
        y_prob = cross_val_predict(clf, X_all, y, cv=cv, method="predict_proba")

    elif mode == "grouped":
        X_all = np.hstack([X_num, token_col])
        preproc = ColumnTransformer([
            ("num", StandardScaler(), list(range(n_num))),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [n_num]),
        ], remainder="drop")
        clf = Pipeline([
            ("preproc", preproc),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        n_groups = len(set(tokens_list))
        n_splits = min(5, n_groups)
        cv = GroupKFold(n_splits=n_splits)
        print(f"Mode=grouped: running GroupKFold(n_splits={n_splits}) by token (groups={n_groups})")
        y_pred = cross_val_predict(clf, X_all, y, cv=cv, groups=np.array(tokens_list), method="predict")
        try:
            y_prob = cross_val_predict(clf, X_all, y, cv=cv, groups=np.array(tokens_list), method="predict_proba")
        except Exception:
            y_prob = np.zeros((len(y), 2))

    elif mode == "loog":
        # LeaveOneGroupOut by token (stricter generalization test)
        X_all = np.hstack([X_num, token_col])
        preproc = ColumnTransformer([
            ("num", StandardScaler(), list(range(n_num))),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [n_num]),
        ], remainder="drop")
        clf = Pipeline([
            ("preproc", preproc),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        logo = LeaveOneGroupOut()
        print(f"Mode=loog: running LeaveOneGroupOut by token (groups={len(set(tokens_list))})")
        y_pred = cross_val_predict(clf, X_all, y, cv=logo, groups=np.array(tokens_list), method="predict")
        try:
            y_prob = cross_val_predict(clf, X_all, y, cv=logo, groups=np.array(tokens_list), method="predict_proba")
        except Exception:
            y_prob = np.zeros((len(y), 2))

    elif mode == "mfcc":
        X_all = X_num
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        print(f"Mode=mfcc: running 5-fold stratified CV using MFCC-augmented features ({len(filenames)} samples)")
        y_pred = cross_val_predict(clf, X_all, y, cv=cv, method="predict")
        y_prob = cross_val_predict(clf, X_all, y, cv=cv, method="predict_proba")

    else:
        raise ValueError(f"Unknown mode {mode}")

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    metrics = {
        "mode": mode,
        "n_samples": int(len(filenames)),
        "n_numeric_features": int(n_num),
        "n_tokens": int(len(set(tokens_list))),
        "classification_report": report,
        "confusion_matrix": cm,
    }

    metrics_fp = RUNS_DIR / f"logreg_{lang_label}_metrics_{mode}.json"
    with open(metrics_fp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Train final model on all data and save
    clf.fit(X_all, y)
    model_fp = RUNS_DIR / f"logreg_{lang_label}_model_{mode}.pkl"
    joblib.dump(clf, model_fp)

    # Save predictions CSV
    probs = (y_prob[:, 1] if y_prob.ndim == 2 else [float(x[1]) for x in y_prob])
    out_fp = RUNS_DIR / f"logreg_{lang_label}_predictions_{mode}.csv"
    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("filename,duration_group,token,true_label,pred_label,pred_prob_long\n")
        for fn, dur, tok, true, pred, prob in zip(filenames, duration_list, tokens_list, y.tolist(), y_pred.tolist(), (probs if isinstance(probs, (list, np.ndarray)) else list(probs))):
            f.write(f"{fn},{dur},{tok},{int(true)},{int(pred)},{float(prob)}\n")

    print("Saved:")
    print(f" - metrics -> {metrics_fp}")
    print(f" - model -> {model_fp}")
    print(f" - predictions -> {out_fp}")


if __name__ == "__main__":
    main()
