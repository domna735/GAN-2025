# Start TensorBoard pointing to ASCII-safe runs/tb directory (PowerShell)
# Activate your venv first (choose .venv_cpu or .venv_gpu)
# Example:
# .\.venv_gpu\Scripts\Activate.ps1
# Then run this script (or call the command below):

param(
    [string]$LogDir = "runs\tb",
    [int]$Port = 6006
)

Write-Host "Starting TensorBoard with logdir=$LogDir on port $Port"
python -m tensorboard.main --logdir $LogDir --bind_all --port $Port
