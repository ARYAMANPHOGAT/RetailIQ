@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo   RetailIQ - one-time setup
echo ================================================

rem Prefer the Windows "py" launcher, otherwise "python"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 or 3.13 ^(64-bit^) from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation. Then run setup.bat again.
  pause & exit /b 1
)
%PY% -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)"
if errorlevel 1 (
  echo RetailIQ needs Python 3.12 or newer. Found:
  %PY% --version
  pause & exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

echo Installing packages (first time can take a few minutes) ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. See README.md, section "Troubleshooting".
  pause & exit /b 1
)

echo Checking the saved models ...
python -c "import ml_pipeline as m; s=m.load_artifacts(); print('Model status:', s['state']); raise SystemExit(0 if s['state']=='ready' else 3)"
if errorlevel 3 (
  echo Saved models could not be loaded on this machine - retraining now ^(about 1 minute^) ...
  python train.py
  if errorlevel 1 ( echo Training failed - see the messages above. & pause & exit /b 1 )
)

echo.
echo Setup complete. Start the app by double-clicking run_retailiq.bat
pause
