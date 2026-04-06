@echo off
REM Setup script for Quantum-Enhanced Drug Discovery System
REM This script creates a virtual environment and installs all dependencies

echo ============================================
echo Quantum Drug Discovery - Setup Script
echo ============================================
echo.

REM Create virtual environment
echo [1/3] Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [3/3] Installing dependencies from requirements.txt...
echo This may take 5-10 minutes depending on your internet speed...
pip install -r requirements.txt

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Activate virtual environment:
echo    .\venv\Scripts\activate
echo.
echo 2. Run the quick demo:
echo    python quickstart.py
echo.
echo 3. Or run interactive demo:
echo    python demo.py
echo.
echo 4. For full training:
echo    python train.py --help
echo.
pause
