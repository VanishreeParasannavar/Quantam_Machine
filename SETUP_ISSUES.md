# Setup & Import Error Resolution Guide

## Pylance Import Errors - Why They Occur

You're seeing import errors from Pylance because **Python dependencies haven't been installed yet**. This is completely normal after downloading a new project.

```
✗ Import "torch" could not be resolved
✗ Import "numpy" could not be resolved
✗ Import "pandas" could not be resolved
... (and many others)
```

**This is NOT a problem with the code** - it means we need to install the packages.

---

## Solution: Install Dependencies

### Option 1: Quick Setup Script (Recommended - 1 command)

#### Windows (Command Prompt)
```bash
setup.bat
```

#### Windows (PowerShell)
```powershell
.\setup.ps1
```

#### macOS/Linux
```bash
bash setup.sh
```

**This script will:**
- ✅ Create virtual environment
- ✅ Install all 19 dependencies
- ✅ Configure Python path
- ✅ Fix all import errors

**Time**: 5-10 minutes depending on internet speed

---

### Option 2: Manual Installation

#### Windows (Command Prompt)
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Verify Installation

After installation completes, verify everything:

```bash
python verify_install.py
```

Expected output:
```
✓ Python 3.12.10
✓ NumPy
✓ PyTorch
✓ PyTorch Geometric
✓ PennyLane
✓ RDKit
... (and others)

✅ All checks passed! System is ready to use.
```

---

## Configure VS Code (Optional)

VS Code usually auto-detects the virtual environment. If it doesn't:

1. **Open Command Palette**: `Ctrl+Shift+P`
2. **Type**: "Python: Select Interpreter"
3. **Choose**: Your newly created virtual environment

You should see:
```
./venv/bin/python (recommended)
```

The import errors should then disappear within a few seconds.

---

## Package Installation Details

### What Each Package Does

| Package | Purpose |
|---------|---------|
| **torch** | Deep learning framework |
| **torch-geometric** | Graph neural networks |
| **pennylane** | Quantum computing |
| **rdkit** | Molecular processing |
| **numpy** | Numerical computing |
| **pandas** | Data handling |
| **matplotlib** | Data visualization |
| **tqdm** | Progress bars |
| **scikit-learn** | Machine learning utilities |
| + 10 more | Supporting libraries |

### Installation Files

- `requirements.txt` - Full package list  
- `setup.bat` - Windows batch script
- `setup.ps1` - PowerShell script
- `.vscode/settings.json` - VS Code configuration
- `pyrightconfig.json` - Pylance configuration

---

## Troubleshooting

### Issue: "pip command not found"
```bash
# Use Python module:
python -m pip install -r requirements.txt
```

### Issue: "Permission denied" on macOS/Linux
```bash
# Make setup executable:
chmod +x setup.sh
./setup.sh
```

### Issue: Virtual environment doesn't activate
```bash
# Check syntax (note the dot and space):
source venv/bin/activate      # macOS/Linux
.\venv\Scripts\activate       # Windows CMD
.\venv\Scripts\Activate.ps1   # Windows PowerShell
```

### Issue: "Module still not found after pip install"
```bash
# Verify virtual environment is active:
which python              # macOS/Linux
where python             # Windows
# Should show path to venv/bin/python or venv\Scripts\python.exe

# If not active, activate it first
```

### Issue: Slow download/timeout
```bash
# Try installing one package at a time:
pip install torch
pip install torch-geometric
# ... etc
```

---

## What Happens After Installation

Once installed, you can immediately:

```bash
# Quick 5-minute demo
python quickstart.py

# Interactive walkthrough
python demo.py

# Full training with benchmarks
python train.py --help
```

---

## File Locations

After setup, your project will have:

```
Quantam_Machine/
├── venv/                    # Virtual environment (newly created)
├── data/                    # Datasets (auto-downloads on first run)
├── results/                 # Training outputs (auto-created)
├── src/                     # Source code (unchanged)
├── .vscode/
│   ├── settings.json        # VS Code Python config
│   └── launch.json          # Debug configurations
├── pyrightconfig.json       # Pylance config
└── ... (other files)
```

---

## Next Steps After Setup

1. ✅ Run installation script
2. ✅ Verify with `python verify_install.py`
3. ✅ Try `python quickstart.py`
4. ✅ Read `GETTING_STARTED.md`
5. ✅ Explore `demo.py`

---

## Common Questions

**Q: Do I need GPU?**  
A: No, it defaults to CPU. GPU speeds up training 5-10x.

**Q: How much disk space?**  
A: ~2GB for venv + 5GB for datasets = 7GB total

**Q: Can I skip the virtual environment?**  
A: Install to system Python, but NOT recommended (can cause conflicts)

**Q: What if pip is slow?**  
A: Use a faster mirror or install packages incrementally

---

## Support

If you still see import errors after completing setup:

1. Check virtual environment is active: `python -c "import sys; print(sys.prefix)"`
2. Reinstall problematic package: `pip install --force-reinstall torch`
3. Run verification: `python verify_install.py`
4. Check Python version: `python --version` (must be 3.8+)

---

**You should be able to run code now!** 🚀

Start with: `python quickstart.py`

