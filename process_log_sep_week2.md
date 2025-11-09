2025-09-06 — Weekly update (week 2)

Summary

- Created a clean CPU-only Python environment and installed the audio / ML dependencies needed to run `wavegan-master` on Windows without WSL/GPU.
- Fixed the earlier numpy build/Meson Unicode error by using a Python runtime with available prebuilt wheels (created a new venv, `.venv_cpu`).
- Installed `tensorflow==2.17.1` (intel build) and verified TensorFlow + NumPy import and a tiny tensor operation on CPU.

Actions taken (chronological)

- Removed problematic/incomplete venvs where needed and created a fresh venv for CPU work:

```powershell
py -3 -m venv .venv_cpu
```

- Upgraded packaging tools and installed CPU packages (record of exact install used):

```powershell
.\.venv_cpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv_cpu\Scripts\python.exe -m pip install "tensorflow==2.17.1" "numpy==1.26.4" "scipy==1.11.4" "librosa==0.8.1" "soundfile==0.12.1" "audioread==3.0.1" "matplotlib==3.8.4"
```

- Observed and resolved build error for `numpy` that earlier attempted a Meson build and failed with a UnicodeDecodeError when running under Python 3.13 in the non-ASCII project path. Workarounds applied:
	- Use a Python interpreter where prebuilt wheels exist (created `.venv_cpu` with `py -3`, which used the installed CPython 3.9 on this machine).
	- Alternatively recommended: copy project to an ASCII-only path (e.g., `C:\Projects\GAN2025`) to avoid Meson/configparse encoding issues.

Verification

- Installed packages (partial listing): tensorflow 2.17.1, numpy 1.26.4, scipy 1.11.4, librosa 0.8.1, soundfile 0.12.1, audioread 3.0.1, matplotlib 3.8.4.
- Quick checks run inside `.venv_cpu`:

```powershell
.\.venv_cpu\Scripts\python.exe -c "import numpy as np; print('numpy', np.__version__)"
# -> numpy 1.26.4

.\.venv_cpu\Scripts\python.exe -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
# -> tensorflow 2.17.1

# Fast runtime smoke test (disabled oneDNN and limited threads in session):
$env:TF_CPP_MIN_LOG_LEVEL = '2'; $env:TF_ENABLE_ONEDNN_OPTS = '0'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
.\.venv_cpu\Scripts\python.exe - <<'PY'
import tensorflow as tf
import numpy as np
print('tf', tf.__version__)
print('numpy', np.__version__)
print('GPUs', tf.config.list_physical_devices('GPU'))
print('small op:', (tf.constant([1.0,2.0,3.0]) * 2).numpy())
PY
# -> tf 2.17.1, numpy 1.26.4, GPUs [] (CPU only), small op: [2. 4. 6.]
```

Notes / diagnosis

- Root cause of initial failure: pip fell back to building numpy from source when a suitable wheel wasn't matched for the active Python. The Meson build process failed while parsing machine files due to a UnicodeDecodeError — likely caused by non-ASCII characters in the project path or temp files. Building numpy from source on Windows requires the Visual C++ toolchain and Meson/Ninja; avoid by using a Python version with prebuilt wheels or by installing the build toolchain.
- PowerShell line-continuation differences: avoid PowerShell backticks inside cmd.exe and do not quote executable paths without the PowerShell call operator (&) when required.
- The project still includes `.venv_gpu` (tools) and a WSL-based GPU setup; for GPU training we continue to recommend WSL2 with a Linux TF+CUDA setup (cleanest path for recent TF versions).

Next steps (short)

1. Run inference + preview to produce WAVs (PowerShell, from project root):

```powershell
& ".\.venv_cpu\Scripts\Activate.ps1"
& ".\.venv_cpu\Scripts\python.exe" .\wavegan-master\train_specgan.py infer runs\specgan_wsl --data_moments_fp runs\specgan\moments.pkl
& ".\.venv_cpu\Scripts\python.exe" .\wavegan-master\train_specgan.py preview runs\specgan_wsl
Start-Process ".\runs\specgan_wsl\preview\00050024.wav"
```

2. Start TensorBoard (separate terminal):

```powershell
& ".\.venv_cpu\Scripts\python.exe" -m tensorboard --logdir runs\specgan_wsl --port 6006
# then open http://localhost:6006
```

3. If preview fails with key/name mismatches, run infer before preview (re-export inference graph), or do a tiny resume train to refresh inference graph and checkpoint pairing.

4. Optional: move project to `C:\Projects\GAN2025` to avoid future Meson/encoding problems.

Entry prepared by: automated assistant — update recorded 2025-09-06.

2025-09-06 14:20 — TensorBoard started (local)

Command run (from project root, activated `.venv_cpu`):

```powershell
& ".\.venv_cpu\Scripts\tensorboard.exe" --logdir runs\specgan_wsl --port 6006
```

Observed output:

```
D:\東南亞長短元音對立論文\GAN 2025\.venv_cpu\lib\site-packages\tensorboard\default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
	import pkg_resources
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.17.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Note: Browser reported "localhost refused to connect" earlier because TensorBoard was not running; starting via the `tensorboard.exe` resolved it. If exposure to other machines is required, run with `--bind_all` and adjust firewall rules.

