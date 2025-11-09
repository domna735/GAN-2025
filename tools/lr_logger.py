"""Learning rate & loss logging callback utilities.

Provides a simple CSV logger for tracking LR vs step/epoch and core losses.
Integrate with training loop in train_ciwgan.py.
"""
from __future__ import annotations
import csv
import time
from pathlib import Path

class LRScheduleLogger:
    def __init__(self, csv_path: str, fieldnames=None):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or ["timestamp", "epoch", "global_step", "lr", "critic_loss", "gen_loss", "info_cat", "info_dur"]
        self._init_file()
    def _init_file(self):
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.fieldnames)
    def log(self, epoch: int, global_step: int, lr: float, critic_loss: float, gen_loss: float, info_cat: float, info_dur: float):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([time.time(), epoch, global_step, lr, critic_loss, gen_loss, info_cat, info_dur])

if __name__ == "__main__":
    # smoke test
    logger = LRScheduleLogger("runs/lr_schedule.csv")
    logger.log(0, 1, 2e-4, 0.0, 0.0, 0.0, 0.0)
    print("LR logger initialized:", logger.path)
