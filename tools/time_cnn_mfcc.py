"""Time-aware CNN on MFCC frames.

Builds MFCC frame matrices from raw audio (found under the provided language folder), pads/truncates to fixed length,
trains a small Conv1D model using Keras (TensorFlow), and evaluates with GroupKFold by token (default) or LOOG if requested.

Saves: runs/logreg_<Lang>_metrics_timecnn.json, runs/logreg_<Lang>_predictions_timecnn.csv, runs/logreg_<Lang>_model_timecnn.h5

Usage examples:
  python tools/time_cnn_mfcc.py --viet-dir "vowel_length_gan_2025-08-24/Vietnamese/Vietnamese" --cv grouped

Note: Requires librosa and tensorflow available in the venv (we checked TensorFlow earlier).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
import joblib

# reuse helpers from logistic_regression_vietnamese
import sys
# make tools/ importable when running from repo root
TOOLS_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(TOOLS_DIR))
import logistic_regression_vietnamese as lrv

try:
    import numpy as _np
    # workaround for older librosa expecting deprecated numpy aliases
    if not hasattr(_np, 'complex'):
        _np.complex = complex
    # other deprecated aliases some audio libs still use
    if not hasattr(_np, 'float'):
        _np.float = float
    if not hasattr(_np, 'int'):
        _np.int = int
    import librosa
except Exception:
    raise RuntimeError("librosa is required for MFCC extraction")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    raise RuntimeError("TensorFlow/Keras is required for model training")


# Small callback to write the current learning rate into the TensorBoard logdir
class LRSummaryCallback(keras.callbacks.Callback):
    def __init__(self, logdir):
        super().__init__()
        self.logdir = str(logdir)
        try:
            self.writer = tf.summary.create_file_writer(self.logdir)
        except Exception:
            self.writer = None
        # CSV fallback: open a small CSV file to record epoch,lr for quick inspection
        try:
            self.csv_path = os.path.join(self.logdir, 'lr_log.csv')
            # write header if not exists
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w', encoding='utf-8') as fh:
                    fh.write('epoch,learning_rate\n')
        except Exception:
            self.csv_path = None

    def _get_lr(self):
        # TF2: optimizer may expose a decayed lr; fall back to attribute if needed
        opt = getattr(self, 'model', None) and getattr(self.model, 'optimizer', None)
        if opt is None:
            return None
        # try decayed lr
        try:
            lr_val = opt._decayed_lr(tf.float32).numpy()
            return float(lr_val)
        except Exception:
            pass
        # try common attribute names
        try:
            from tensorflow.keras import backend as K
            try:
                return float(K.get_value(opt.learning_rate))
            except Exception:
                return float(K.get_value(opt.lr))
        except Exception:
            # last resort: try attributes directly
            try:
                lr_attr = getattr(opt, 'learning_rate', getattr(opt, 'lr', None))
                if lr_attr is None:
                    return None
                try:
                    return float(lr_attr.numpy())
                except Exception:
                    return float(lr_attr)
            except Exception:
                return None

    def on_epoch_end(self, epoch, logs=None):
        lr = self._get_lr()
        # write TensorBoard scalar if writer available
        if lr is not None and self.writer is not None:
            try:
                with self.writer.as_default():
                    tf.summary.scalar('learning_rate', data=float(lr), step=epoch)
                    self.writer.flush()
            except Exception:
                pass
        # append to CSV fallback if lr is available
        if lr is not None and getattr(self, 'csv_path', None):
            try:
                with open(self.csv_path, 'a', encoding='utf-8') as fh:
                    fh.write(f"{int(epoch)},{float(lr)}\n")
            except Exception:
                pass

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "vowel_length_gan_2025-08-24" / "processed_data"
TB_ROOT = ROOT / "聲頻圖"
TB_ROOT.mkdir(exist_ok=True)


def build_dataset(local_viet_dir: Path, max_len: int = 200, sr: int = 16000, n_mfcc: int = 13, limit: int | None = None):
    tokens = lrv.gather_vietnamese_tokens(local_viet_dir)
    matches = lrv.find_matching_npy_files(DATA_DIR, tokens)
    if not matches:
        raise RuntimeError("No matching .npy files found for tokens")

    X_list = []
    filenames = []
    tokens_list = []
    duration_list = []

    for p, token, duration in matches:
        wave = lrv.find_audio_for_token(local_viet_dir, token)
        if wave is None:
            # skip if no raw audio
            # optionally print a warning
            # print(f"Warning: no raw audio for token {token}; skipping sample {p.name}")
            continue
        try:
            # load with a faster resampling algorithm to avoid slow numba/resampy paths
            y, _sr = librosa.load(str(wave), sr=sr, res_type='kaiser_fast')
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            # mfcc shape: (n_mfcc, T)
            T = mfcc.shape[1]
            if T >= max_len:
                mfcc = mfcc[:, :max_len]
            else:
                pad = np.zeros((n_mfcc, max_len - T), dtype=float)
                mfcc = np.hstack([mfcc, pad])
            # transpose to (max_len, n_mfcc)
            mfcc_t = mfcc.T.astype(np.float32)
            X_list.append(mfcc_t)
            filenames.append(p.name)
            tokens_list.append(token)
            duration_list.append(0 if duration == 'short' else 1)
            if limit is not None and len(X_list) >= limit:
                break
        except Exception as e:
            print(f"Error processing {wave}: {e}")
            continue

    if not X_list:
        raise RuntimeError("No samples with raw audio were found/loaded")

    X = np.stack(X_list, axis=0)  # (N, max_len, n_mfcc)
    y = np.array(duration_list, dtype=int)
    groups = np.array(tokens_list, dtype=object)
    return X, y, groups, filenames


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_timecnn(local_viet_dir: Path, cv: str = 'grouped', max_len: int = 200, epochs: int = 12, batch_size: int = 16, limit: int | None = None, tb_dir: Path | None = None):
    print(f"Building dataset for {local_viet_dir} (max_len={max_len})")
    X, y, groups, filenames = build_dataset(local_viet_dir, max_len=max_len, limit=limit)
    n_samples = X.shape[0]
    print(f"Loaded {n_samples} samples")

    # choose CV strategy
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if cv == 'grouped':
        n_splits = min(5, n_groups)
        cv_obj = GroupKFold(n_splits=n_splits)
        splits = cv_obj.split(X, y, groups)
    elif cv == 'stratified':
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = cv_obj.split(X, y)
    elif cv == 'loog':
        cv_obj = LeaveOneGroupOut()
        splits = cv_obj.split(X, y, groups)
    else:
        raise ValueError(f"Unknown cv {cv}")

    y_pred = np.zeros_like(y)
    y_prob = np.zeros((len(y), 2), dtype=float)

    fold = 0
    for train_idx, val_idx in splits:
        fold += 1
        print(f"Fold {fold}: train={len(train_idx)} val={len(val_idx)}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # class weights
        classes = np.unique(y_train)
        class_weight = compute_class_weight('balanced', classes=classes, y=y_train)
        cw = {int(c): float(w) for c, w in zip(classes, class_weight)}

        model = build_model(input_shape=X.shape[1:])
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        # assemble callbacks (optionally include TensorBoard)
        cbs = [es]
        if tb_dir is not None:
            fold_tb = tb_dir / f"fold_{fold}"
            fold_tb.mkdir(parents=True, exist_ok=True)
            tb_cb = keras.callbacks.TensorBoard(log_dir=str(fold_tb))
            cbs.append(tb_cb)
            # append LR logger so the learning rate is written to TB for later inspection
            try:
                cbs.append(LRSummaryCallback(fold_tb))
            except Exception:
                pass

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, class_weight=cw, callbacks=cbs, verbose=1)

        probs = model.predict(X_val, batch_size=batch_size).ravel()
        preds = (probs >= 0.5).astype(int)
        y_pred[val_idx] = preds
        y_prob[val_idx, 1] = probs
        y_prob[val_idx, 0] = 1.0 - probs

    # metrics
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()
    metrics = {
        'mode': 'timecnn',
        'cv': cv,
        'n_samples': int(n_samples),
        'n_tokens': int(n_groups),
        'classification_report': report,
        'confusion_matrix': cm,
    }

    lang_label = local_viet_dir.name
    metrics_fp = RUNS_DIR / f"logreg_{lang_label}_metrics_timecnn.json"
    with open(metrics_fp, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # train final model on all data and save
    final_model = build_model(input_shape=X.shape[1:])
    classes = np.unique(y)
    class_weight = compute_class_weight('balanced', classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, class_weight)}
    final_cbs = []
    if tb_dir is not None:
        final_tb = tb_dir / "final"
        final_tb.mkdir(parents=True, exist_ok=True)
        final_cbs.append(keras.callbacks.TensorBoard(log_dir=str(final_tb)))
        try:
            final_cbs.append(LRSummaryCallback(final_tb))
        except Exception:
            pass
    final_model.fit(X, y, epochs=epochs, batch_size=batch_size, class_weight=cw, verbose=1, callbacks=final_cbs)
    model_fp = RUNS_DIR / f"logreg_{lang_label}_model_timecnn.h5"
    final_model.save(str(model_fp))

    # save predictions CSV
    out_fp = RUNS_DIR / f"logreg_{lang_label}_predictions_timecnn.csv"
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write('filename,duration_group,token,true_label,pred_label,pred_prob_long\n')
        for fn, tok, true, pred, prob in zip(filenames, groups, y.tolist(), y_pred.tolist(), y_prob[:, 1].tolist()):
            f.write(f"{fn},{'short' if true==0 else 'long'},{tok},{int(true)},{int(pred)},{float(prob)}\n")

    print('Saved:')
    print(' - metrics ->', metrics_fp)
    print(' - model ->', model_fp)
    print(' - predictions ->', out_fp)

    return metrics_fp, out_fp, RUNS_DIR / f"logreg_{lang_label}_predictions_timecnn_per_token.csv"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viet-dir', required=True, help='Path to language folder under vowel_length_gan_2025-08-24/Vietnamese')
    parser.add_argument('--cv', choices=['grouped', 'stratified', 'loog'], default='grouped')
    parser.add_argument('--max-len', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None, help='Optional: limit number of samples (for fast debug runs)')
    parser.add_argument('--tb-dir', type=str, default=None, help='Optional: directory to write TensorBoard logs (default: 聲頻圖/timecnn_<lang>_<ts>)')
    args = parser.parse_args()

    viet_dir = Path(args.viet_dir)
    if not viet_dir.exists():
        print('Provided --viet-dir does not exist:', viet_dir)
        raise SystemExit(1)
    # prepare TensorBoard directory
    if args.tb_dir:
        tb_dir = Path(args.tb_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
    else:
        # prefer the repository '聲頻圖' folder, but TensorFlow C++ APIs sometimes fail on
        # non-ASCII or very long Windows paths; fall back to an ASCII-safe runs/tb/ path
        use_tb_root = TB_ROOT
        if not all(ord(c) < 128 for c in str(TB_ROOT)):
            # fallback to runs/tb
            use_tb_root = RUNS_DIR / "tb"
            use_tb_root.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        tb_dir = use_tb_root / f"timecnn_{viet_dir.name}_{ts}"
        tb_dir.mkdir(parents=True, exist_ok=True)

    metrics_fp, preds_fp, _ = evaluate_timecnn(viet_dir, cv=args.cv, max_len=args.max_len, epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, tb_dir=tb_dir)
    # generate per-token CSV and plot using existing tool
    try:
        # try importing as a top-level module (tools dir was inserted on sys.path)
        import plot_per_token as plot_per_token_mod
        plot_per_token_mod.run(preds_fp)
    except Exception:
        try:
            # fallback: package-style import if tools is a package
            from tools import plot_per_token as plot_per_token_mod2
            plot_per_token_mod2.run(preds_fp)
        except Exception as e:
            print('Could not generate per-token plot:', e)
