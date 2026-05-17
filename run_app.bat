@echo off
cd /d "%~dp0"
echo =======================================================
echo Starting Heat Exchanger Design Assistant...
echo Please wait while the required packages are being loaded.
echo =======================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app_corrected.py_2.py --global.developmentMode=false --server.headless=false
) else (
    python -m streamlit run app_corrected.py_2.py --global.developmentMode=false --server.headless=false
)
pause
