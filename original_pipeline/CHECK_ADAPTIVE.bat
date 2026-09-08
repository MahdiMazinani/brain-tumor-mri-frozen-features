@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0venv311\Scripts\python.exe"
if not exist "%PY%" (echo Missing venv311 Python & exit /b 2)
"%PY%" test_ADAPTIVE.py
if errorlevel 1 exit /b 1
"%PY%" ADAPTIVE_RUN.py check
exit /b %errorlevel%
