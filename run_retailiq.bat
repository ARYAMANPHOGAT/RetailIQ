@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo Virtual environment not found. Run setup.bat first ^(trying the system Python instead^).
)
echo Starting RetailIQ ... it opens in your browser at http://localhost:8501
echo Keep this window open while using the app. Press Ctrl+C or close the window to stop.
python -m streamlit run app.py
pause
