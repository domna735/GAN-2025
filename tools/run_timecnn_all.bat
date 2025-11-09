@echo off
REM run_timecnn_all.bat — activates .venv_cpu and runs the Python wrapper for all languages
REM Usage examples:
REM   tools\run_timecnn_all.bat --epochs 1 --max-len 100 --batch-size 16 --limit 60
REM   tools\run_timecnn_all.bat --epochs 30 --max-len 200 --batch-size 16

SET SCRIPT_DIR=%~dp0
CALL "%SCRIPT_DIR%..\.venv_cpu\Scripts\activate.bat"
python tools\run_timecnn_all.py %*
pause
