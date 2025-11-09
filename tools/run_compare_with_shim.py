"""Small wrapper to run `tools/compare_vot_intensity.py` after restoring numpy aliases
that older librosa versions expect. Call this script with the same args you would
pass to `compare_vot_intensity.py`.

Example:
.venv_cpu\Scripts\python.exe tools\run_compare_with_shim.py --orig-dir "vowel_length_gan_2025-08-24/Vietnamese/Vietnamese" --gen-dir runs/gen/griffinlim --out runs/compare/griffinlim_vn_compare.csv
"""
import sys
import numpy as np
import runpy

# restore deprecated numpy aliases if missing
for _name, _val in (('complex', complex), ('float', float), ('int', int), ('bool', bool), ('object', object)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)

if __name__ == '__main__':
    # forward args to the compare script
    # runpy will execute the script as __main__ using current sys.argv
    runpy.run_path('tools/compare_vot_intensity.py', run_name='__main__')
