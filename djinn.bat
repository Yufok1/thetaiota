@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"
REM Unified CLI shim: use without calling python explicitly
python cli_control.py %*
exit /b %errorlevel%


