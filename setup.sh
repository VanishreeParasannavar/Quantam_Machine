#!/bin/bash
# Setup script for Quantum-Enhanced Drug Discovery System
# Usage: bash setup.sh (on macOS/Linux)

echo "============================================"
echo "Quantum Drug Discovery - Setup Script"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "[2/3] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install requirements
echo "[3/3] Installing dependencies from requirements.txt..."
echo "This may take 5-10 minutes depending on your internet speed..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo ""
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verify installation:"
echo "   python verify_install.py"
echo ""
echo "3. Run the quick demo:"
echo "   python quickstart.py"
echo ""
echo "4. Or run interactive demo:"
echo "   python demo.py"
echo ""
echo "5. For full training:"
echo "   python train.py --help"
echo ""
