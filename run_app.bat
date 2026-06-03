@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PY=%~dp0.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [LOI] Chua co .venv
    echo Chay trong PowerShell:
    echo   cd /d "%~dp0"
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "models\pipeline.joblib" (
    echo [LOI] Chua co mo hinh. Chay train truoc:
    echo   "%PY%" -m src.train
    pause
    exit /b 1
)

"%PY%" -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Dang cai streamlit...
    "%PY%" -m pip install -r requirements.txt
)

echo.
echo ========================================
echo  Dang khoi dong Streamlit...
echo  KHONG DONG cua so nay khi dang dung app
echo  Mo trinh duyet: http://localhost:8501
echo ========================================
echo.

"%PY%" -m streamlit run app.py --server.headless false

if errorlevel 1 (
    echo.
    echo [LOI] Streamlit khong chay duoc. Thu cong:
    echo   cd /d "%~dp0"
    echo   .venv\Scripts\activate
    echo   python -m streamlit run app.py
    pause
)
