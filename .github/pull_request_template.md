# Pull Request

## Summary
Short explanation of the change.

## Type of change
- [ ] Feature
- [ ] Bug fix
- [ ] Refactor / cleanup
- [ ] Docs
- [ ] Experiment

## Related Issue
Closes #<issue-number> (or "N/A").

## How to test
Commands or steps to reproduce / validate (small sample):

```powershell
. .\.venv_cpu\Scripts\Activate.ps1
python tools\time_cnn_mfcc.py --viet-dir "vowel_length_gan_2025-08-24\Vietnamese\Vietnamese" --cv grouped --max-len 80 --epochs 1 --batch-size 8 --limit 40
```

## Checklist
- [ ] Code runs locally (describe interpreter / venv)
- [ ] Added / updated docs (process log or README)
- [ ] Metrics JSON/CSV saved (if experiment)
- [ ] No large data accidentally committed
- [ ] Git LFS installed if adding binaries

## Screenshots / Logs (optional)
Paste key log lines or attach images.

## Notes
Anything the reviewer should pay special attention to (edge cases, follow-ups, risk).
