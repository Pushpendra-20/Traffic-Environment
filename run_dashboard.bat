@echo off
setlocal
cd /d "%~dp0"

set "ENV_DIR=%CD%\.venv"
set "PY=%ENV_DIR%\Scripts\python.exe"

if not exist "%PY%" (
  echo Creating the project Python environment...
  py -3.12 -m venv "%ENV_DIR%" 2>nul
  if errorlevel 1 python -m venv "%ENV_DIR%"
  if errorlevel 1 goto :failed

  echo Installing dashboard packages (first launch only)...
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :failed
  "%PY%" -m pip install -r "%CD%\requirements.txt"
  if errorlevel 1 goto :failed
)

"%PY%" -m streamlit run "%CD%\app.py"
if errorlevel 1 goto :failed
goto :eof

:failed
echo.
echo The Traffic and Environment dashboard could not start. Review the error above.
pause
exit /b 1
