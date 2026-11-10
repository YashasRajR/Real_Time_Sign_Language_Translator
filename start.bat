@echo off
REM Starts the sign translator (run the setup in README.md once first).
cd /d "%~dp0"
call venv\Scripts\activate
python translate.py %*
pause
