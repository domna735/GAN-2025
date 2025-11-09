"""Inspect learning rate scalars from TensorBoard event folders and plot/save results.

Usage:
  python tools/inspect_lr.py --tb-root runs/tb --out-dir runs/plots

This script searches subfolders in `--tb-root` for TensorBoard event files,
extracts scalar tags that look like learning rate (contains 'lr' or 'learning_rate'),
and writes CSV + PNG plots per run.
"""
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


def find_event_dirs(tb_root):
    # Look for subdirectories with event files
    dirs = []
    for root, dnames, fnames in os.walk(tb_root):
        for f in fnames:
            if f.startswith('events.out.tfevents'):
                dirs.append(root)
                break
    return sorted(set(dirs))


def extract_scalars(event_dir):
    if EventAccumulator is None:
        raise RuntimeError('tensorboard.event_accumulator.EventAccumulator not available; run in venv with tensorboard installed')
    ea = EventAccumulator(event_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    # pick candidate LR tags
    lr_tags = [t for t in tags if 'lr' in t.lower() or 'learning_rate' in t.lower()]
    results = {}
    for tag in lr_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        results[tag] = (steps, vals)
    return results


def save_plot_csv(out_dir, run_name, scalars):
    os.makedirs(out_dir, exist_ok=True)
    for tag, (steps, vals) in scalars.items():
        df = pd.DataFrame({'step': steps, 'value': vals})
        csv_path = os.path.join(out_dir, f'lr_{run_name}_{safe_name(tag)}.csv')
        df.to_csv(csv_path, index=False)
        # plot
        plt.figure()
        plt.plot(df['step'], df['value'], marker='o')
        plt.xlabel('step')
        plt.ylabel('learning_rate')
        plt.title(f'{run_name} - {tag}')
        png_path = os.path.join(out_dir, f'lr_{run_name}_{safe_name(tag)}.png')
        plt.grid(True)
        plt.savefig(png_path)
        plt.close()


def safe_name(s):
    return ''.join(c if c.isalnum() else '_' for c in s)[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tb-root', default='runs/tb', help='TensorBoard root folder')
    p.add_argument('--out-dir', default='runs/plots', help='Output folder for CSVs and PNGs')
    args = p.parse_args()

    dirs = find_event_dirs(args.tb_root)
    if not dirs:
        print('No TensorBoard event dirs found under', args.tb_root)
        return
    print('Found event dirs:', dirs)
    for d in dirs:
        run_name = os.path.basename(d.rstrip('/\\'))
        try:
            scalars = extract_scalars(d)
        except Exception as e:
            print('Failed to read', d, '->', e)
            continue
        if not scalars:
            print('No LR-like scalar tags in', d)
            continue
        save_plot_csv(args.out_dir, run_name, scalars)
        print('Saved LR plots for', run_name)

    # Fallback: look for per-run lr_log.csv files (written by training callback)
    lr_csvs = glob.glob(os.path.join(args.tb_root, '**', 'lr_log.csv'), recursive=True)
    for csv_fp in lr_csvs:
        try:
            df = pd.read_csv(csv_fp)
            # derive a run name from parent dirs
            run_name = os.path.basename(os.path.dirname(csv_fp))
            out_csv = os.path.join(args.out_dir, f'lr_{run_name}_fallback.csv')
            os.makedirs(args.out_dir, exist_ok=True)
            df.to_csv(out_csv, index=False)
            # plot
            plt.figure()
            plt.plot(df['epoch'], df['learning_rate'], marker='o')
            plt.xlabel('epoch')
            plt.ylabel('learning_rate')
            plt.title(f'{run_name} - lr_log.csv')
            plt.grid(True)
            png_path = os.path.join(args.out_dir, f'lr_{run_name}_fallback.png')
            plt.savefig(png_path)
            plt.close()
            print('Saved fallback LR plot for', run_name, '->', out_csv)
        except Exception as e:
            print('Failed to process fallback csv', csv_fp, '->', e)


if __name__ == '__main__':
    main()
