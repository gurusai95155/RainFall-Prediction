@echo off
echo ========================================================
echo   Starting Rainfall Prediction System (MLflow & Flask)
echo ========================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Python virtual environment not found. Creating .venv...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv 2>nul || py -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

if not exist "model.pkl" (
    echo [INFO] Training model and registering in MLflow...
    python train.py
)

echo [INFO] Launching Web App on http://127.0.0.1:8080 ...
python -u app.py
pause
