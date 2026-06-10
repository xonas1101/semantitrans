@echo off
REM Optional Windows convenience. The Python entry points are the real
REM interface and are identical to Linux/macOS. Requires Python 3.11-3.13.
REM For CUDA, install the matching torch build from https://pytorch.org first.

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv
) else (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if "%1"=="--optional" (
  python -m pip install -r requirements-optional.txt
  python -m spacy download en_core_web_sm
)

echo.
echo Setup complete. Activate with:  .venv\Scripts\activate.bat
python verify_setup.py
