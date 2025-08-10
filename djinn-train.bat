@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"
REM Train the local tiny LM on project text
python train_tiny_lm.py --epochs 100 --lr 3e-4 --context 128 --batch 8
exit /b %errorlevel%


