# QUICK FIX: Pylance Import Errors

## Problem
You see Pylance import errors in VS Code:
```
✗ Import "torch" could not be resolved
✗ Import "numpy" could not be resolved
... (40+ more errors)
```

## Solution (Pick One)

### 🚀 **FASTEST: Run Setup Script (1 command)**

**Windows (CMD)**:
```bash
setup.bat
```

**Windows (PowerShell)**:
```powershell
.\setup.ps1
```

**macOS/Linux**:
```bash
bash setup.sh
```

Time: 5-10 minutes | Result: All errors fixed ✅

---

### **MANUAL: Install Dependencies**

**Windows**:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Time: 5-10 minutes | Result: All errors fixed ✅

---

### **VERIFY: Check Installation**

After setup completes:
```bash
python verify_install.py
```

Expected output:
```
✓ Python 3.12.10
✓ NumPy
✓ PyTorch
✓ All packages...

✅ All checks passed!
```

If you see ✅, you're done! 🎉

---

### **CONFIGURE: VS Code (Optional)**

If errors persist:

1. Press: `Ctrl+Shift+P`
2. Type: `Python: Select Interpreter`
3. Choose: `./venv/Scripts/python` (your new environment)
4. Wait: 5 seconds for Pylance to update

---

## What Happens After Setup

✅ Pylance errors disappear  
✅ Syntax highlighting works  
✅ Code intellisense available  
✅ You can run scripts  

Try it:
```bash
python quickstart.py
```

---

## If You Get Stuck

1. **Venv not activating?**
   - Windows CMD: `.\venv\Scripts\activate`
   - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
   - macOS/Linux: `source venv/bin/activate`

2. **pip not found?**
   - Use: `python -m pip install -r requirements.txt`

3. **Still seeing errors?**
   - Run: `python verify_install.py`
   - Check: Is venv active? (prompt should show ①(venv))

4. **Need more help?**
   - Read: `SETUP_ISSUES.md` (comprehensive troubleshooting)
   - Read: `IMPORT_ERRORS_RESOLVED.md` (detailed explanation)

---

## TL;DR

**Copy-paste this:**

```bash
# Windows
setup.bat

# OR macOS/Linux
bash setup.sh
```

**Then check:**

```bash
python verify_install.py
```

**Done!** Your import errors are fixed. 🚀

---

## 5-Minute Quick Start (After Setup)

```bash
python quickstart.py
```

This will:
- Load molecular dataset
- Train hybrid quantum-classical model  
- Show results
- Done in ~10 minutes

---

**⏱️ Total Time**: 5 min setup + 10 min demo = 15 minutes to see it working

**🎯 Result**: Fully functional quantum ML system for drug discovery

