"""List scalar tags in TensorBoard event subfolders under runs/tb.

Usage:
  .venv_cpu\Scripts\python.exe tools\list_tb_scalars.py

Prints each event dir and the scalar tags found there. Also highlights any tags
that look like learning-rate ('lr' or 'learning_rate').
"""
import os
import glob
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print('Could not import EventAccumulator:', e)
    raise


def find_event_dirs(tb_root='runs/tb'):
    dirs = []
    for root, dnames, fnames in os.walk(tb_root):
        for f in fnames:
            if f.startswith('events.out.tfevents'):
                dirs.append(root)
                break
    return sorted(set(dirs))


def main():
    tb_root = 'runs/tb'
    dirs = find_event_dirs(tb_root)
    if not dirs:
        print('No event dirs found under', tb_root)
        return
    print(f'Found {len(dirs)} event dirs')
    for d in dirs:
        try:
            ea = EventAccumulator(d)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
        except Exception as e:
            print('Failed to read', d, '->', e)
            continue
        if not tags:
            print(d, ': (no scalar tags)')
            continue
        lr_like = [t for t in tags if 'lr' in t.lower() or 'learning_rate' in t.lower()]
        print(d, ':', len(tags), 'scalar tags; lr-like:', lr_like)


if __name__ == '__main__':
    main()
