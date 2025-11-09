"""Simple logistic regression classifier for vowel length (long vs short)

Usage (defaults):
	python tools/logistic_regression.py

This script expects a folder with precomputed spectrogram `.npy` files
at `vowel_length_gan_2025-08-24/processed_data` (relative to repo root).
Files whose filename contains the length marker 'ː' (U+02D0) are labeled
as long (1); all others are labeled short (0).

The pipeline: load .npy files -> extract features (PCA on flattened spectrograms)
-> train LogisticRegression with 5-fold cross-validation -> save model & metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

try:
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
	from sklearn.metrics import classification_report, confusion_matrix
	import joblib
except Exception as e:
	print("Missing dependency for training: scikit-learn and joblib are required.")
	print("Install into your venv, for example: python -m pip install scikit-learn joblib")
	raise


def find_npy_files(data_dir: Path):
	return sorted([p for p in data_dir.glob("*.npy") if p.is_file()])


def label_from_filename(name: str) -> int:
	# If the filename contains the IPA length marker U+02D0 'ː', mark as long
	return 1 if "\u02D0" in name or "ː" in name else 0


def load_features(files: list[Path], flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
	"""Load arrays and return a fixed-size summary feature vector per file.

	Many spectrograms have varying flattened lengths; to keep things robust we
	compute simple summary statistics per file (mean, std, min, max, median,
	10th and 90th percentiles). This yields a fixed 7-dim feature vector.
	"""
	X_list = []
	y_list = []
	for p in files:
		try:
			arr = np.load(p)
		except Exception as exc:
			print(f"Failed to load {p}: {exc}")
			continue
		# compute summary statistics
		arrf = arr.astype(np.float32)
		vals = arrf.ravel()
		feat = np.array([
			float(np.mean(vals)),
			float(np.std(vals)),
			float(np.min(vals)),
			float(np.max(vals)),
			float(np.median(vals)),
			float(np.percentile(vals, 10)),
			float(np.percentile(vals, 90)),
		], dtype=np.float32)
		X_list.append(feat)
		y_list.append(label_from_filename(p.name))
	if not X_list:
		raise RuntimeError("No valid .npy files found or all failed to load.")
	X = np.vstack(X_list)
	y = np.array(y_list, dtype=int)
	return X, y


def main(argv=None):
	p = argparse.ArgumentParser(description="Train logistic regression to classify vowel length")
	p.add_argument("--data-dir", default="vowel_length_gan_2025-08-24/processed_data", help="Path to .npy processed_data folder")
	p.add_argument("--pca", type=int, default=50, help="Number of PCA components (set 0 to skip PCA)")
	p.add_argument("--out", default="runs/logreg_model.pkl", help="Output path for trained model (joblib)")
	p.add_argument("--metrics", default="runs/logreg_metrics.json", help="Output path for metrics JSON")
	args = p.parse_args(argv)

	data_dir = Path(args.data_dir)
	if not data_dir.exists():
		print(f"Data directory not found: {data_dir}")
		sys.exit(2)

	files = find_npy_files(data_dir)
	print(f"Found {len(files)} .npy files in {data_dir}")

	X, y = load_features(files, flatten=True)
	print(f"Feature matrix shape: {X.shape}, labels: {np.bincount(y)}")

	os.makedirs(Path(args.out).parent, exist_ok=True)

	steps = []
	steps.append(("scaler", StandardScaler()))
	if args.pca and args.pca > 0:
		steps.append(("pca", PCA(n_components=min(args.pca, min(X.shape)))) )
	steps.append(("clf", LogisticRegression(max_iter=2000)))

	pipeline = Pipeline(steps)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	print("Running 5-fold cross-validation...")
	scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
	print(f"Cross-val accuracies: {scores}")
	mean_acc = float(np.mean(scores))

	# get cross-validated predictions for full classification report
	y_pred = cross_val_predict(pipeline, X, y, cv=cv)
	report = classification_report(y, y_pred, output_dict=True)
	cm = confusion_matrix(y, y_pred).tolist()

	# fit on full data and save model
	pipeline.fit(X, y)
	joblib.dump(pipeline, args.out)
	print(f"Saved trained model to {args.out}")

	metrics = {
		"n_samples": int(X.shape[0]),
		"n_features_raw": int(X.shape[1]),
		"cv_accuracy_mean": mean_acc,
		"cv_accuracy_all": [float(s) for s in scores],
		"confusion_matrix": cm,
		"classification_report": report,
	}
	with open(args.metrics, "w", encoding="utf8") as fh:
		json.dump(metrics, fh, indent=2, ensure_ascii=False)
	print(f"Saved metrics to {args.metrics}")
	print(json.dumps({"cv_mean_accuracy": mean_acc}, indent=2))


if __name__ == "__main__":
	main()

