@echo off
REM start_tensorboard.bat — activates .venv_cpu and starts TensorBoard (cmd-friendly)
REM Usage: tools\start_tensorboard.bat

REM Resolve script directory
SET SCRIPT_DIR=%~dp0

REM Activate venv (cmd activator avoids PowerShell ExecutionPolicy issues)
CALL "%SCRIPT_DIR%..\.venv_cpu\Scripts\activate.bat"

REM Ensure pip and tensorboard are installed in the venv (safe no-op if already present)
python -m pip install --upgrade pip
python -m pip install tensorboard

REM Start TensorBoard pointing at runs/tb on port 6006
python -m tensorboard.main --logdir runs/tb --bind_all --port 6006
pause
