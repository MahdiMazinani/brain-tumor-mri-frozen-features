@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" goto usage
if "%~2"=="" goto usage
set "PY=%~dp0venv311\Scripts\python.exe"
if not exist "%PY%" (echo Missing venv311 Python & exit /b 2)
"%PY%" ADAPTIVE_RUN.py check
if errorlevel 1 exit /b 1
"%PY%" ADAPTIVE_RUN.py prepare --data "%~1" --out "%~2\manifest"
if errorlevel 1 exit /b 1
"%PY%" ADAPTIVE_RUN.py develop --manifest "%~2\manifest" --run "%~2\development"
exit /b %errorlevel%
:usage
echo Usage: RUN_ALL_ADAPTIVE.bat "dataset-folder" "new-experiment-folder"
echo Runs prepare and validation development only. Final test is separate.
exit /b 2
