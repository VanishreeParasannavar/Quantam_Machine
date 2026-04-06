#!/usr/bin/env pwsh
# Setup script for Quantum-Enhanced Drug Discovery System (PowerShell)
# This script creates a virtual environment and installs all dependencies

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Quantum Drug Discovery - Setup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Create virtual environment
Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install requirements
Write-Host "[3/3] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes depending on your internet speed..." -ForegroundColor Gray
pip install -r requirements.txt

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate virtual environment:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run the quick demo:" -ForegroundColor White
Write-Host "   python quickstart.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or run interactive demo:" -ForegroundColor White
Write-Host "   python demo.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. For full training:" -ForegroundColor White
Write-Host "   python train.py --help" -ForegroundColor Gray
Write-Host ""
