<#
Run time-aware CNN sequentially for the three language folders under vowel_length_gan_2025-08-24/Vietnamese
Usage (PowerShell):
    .\.venv_gpu\Scripts\Activate.ps1
  .\tools\run_timecnn_for_all.ps1 -Epochs 20 -MaxLen 200 -BatchSize 16

This script runs `tools/time_cnn_mfcc.py` for each language (Cantonese, Thai, Vietnamese)
and writes TensorBoard logs to an ASCII-safe path under `runs/tb/` and saves stdout/stderr to `runs/logs/`.
#>
param(
    [int]$Epochs = 12,
    [int]$MaxLen = 200,
    [int]$BatchSize = 16,
    [int]$Limit = 0  # set >0 for quick debug (samples limit)
)

$repo = (Get-Location).Path
$base = Join-Path $repo 'vowel_length_gan_2025-08-24\Vietnamese'
$languages = @('Cantonese','Thai','Vietnamese')

$ts = (Get-Date).ToString('yyyyMMddTHHmmss')
$logroot = Join-Path $repo 'runs\logs'
$tbroot = Join-Path $repo 'runs\tb'

if (-not (Test-Path $logroot)) { New-Item -ItemType Directory -Path $logroot | Out-Null }
if (-not (Test-Path $tbroot)) { New-Item -ItemType Directory -Path $tbroot | Out-Null }

foreach ($lang in $languages) {
    $lang_dir = Join-Path $base $lang
    if (-not (Test-Path $lang_dir)) {
        Write-Warning "Language folder not found: $lang_dir - skipping"
        continue
    }
    $run_ts = (Get-Date).ToString('yyyyMMddTHHmmss')
    $tb_dir = Join-Path $tbroot "timecnn_${lang}_$run_ts"
    New-Item -ItemType Directory -Force -Path $tb_dir | Out-Null

    $log_fp = Join-Path $logroot "timecnn_${lang}_$run_ts.log"
    $cmd = "python tools\time_cnn_mfcc.py --viet-dir `"$lang_dir`" --cv grouped --max-len $MaxLen --epochs $Epochs --batch-size $BatchSize --tb-dir `"$tb_dir`""
    if ($Limit -gt 0) { $cmd += " --limit $Limit" }

    Write-Host "Running: $cmd"
    Write-Host "Logs -> $log_fp"
    # Run synchronously and redirect output to log file
    & cmd /c $cmd *> $log_fp

    Write-Host "Completed run for $lang (logs: $log_fp, tb: $tb_dir)"
}

Write-Host "All runs finished. Use the start_tensorboard.ps1 script to view runs/tb/ or point TensorBoard to runs/tb/."