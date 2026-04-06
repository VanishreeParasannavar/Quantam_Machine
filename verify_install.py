#!/usr/bin/env python3
"""
Installation verification script for Quantum Drug Discovery System
Run this to check if all dependencies are correctly installed
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ Warning: Python 3.8+ is recommended")
        return False
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (not installed)")
        return False

def check_all_packages():
    """Check all required packages"""
    packages = {
        'NumPy': 'numpy',
        'PyTorch': 'torch',
        'PyTorch Geometric': 'torch_geometric',
        'PennyLane': 'pennylane',
        'RDKit': 'rdkit',
        'Pandas': 'pandas',
        'Matplotlib': 'matplotlib',
        'Scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'Requests': 'requests'
    }
    
    missing = []
    for name, import_name in packages.items():
        if not check_package(name, import_name):
            missing.append(name)
    
    return missing

def check_venv():
    """Check if virtual environment is active"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"✓ Virtual environment active: {sys.prefix}")
        return True
    else:
        print("⚠ Virtual environment not active")
        print("  Run: . venv/bin/activate  (macOS/Linux)")
        print("  Or:  .\\venv\\Scripts\\activate  (Windows)")
        return False

def check_project_structure():
    """Check if project files exist"""
    required_files = [
        'src/config.py',
        'src/utils.py',
        'src/gnn_encoder.py',
        'src/quantum_circuit.py',
        'src/hybrid_model.py',
        'src/data_loader.py',
        'src/trainer.py',
        'src/benchmark.py',
        'train.py',
        'quickstart.py',
        'demo.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"✗ {file_path}")
        else:
            print(f"✓ {file_path}")
    
    return missing_files

def main():
    """Run all checks"""
    print("=" * 60)
    print("Quantum Drug Discovery - Installation Verification")
    print("=" * 60)
    print()
    
    print("[1/3] Python Version:")
    check_python_version()
    print()
    
    print("[2/3] Package Installation:")
    missing_packages = check_all_packages()
    print()
    
    print("[3/3] Project Structure:")
    missing_files = check_project_structure()
    print()
    
    print("=" * 60)
    if missing_packages:
        print(f"❌ Missing {len(missing_packages)} package(s):")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print()
        print("To install missing packages:")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)
    elif missing_files:
        print(f"❌ Missing {len(missing_files)} file(s)")
        sys.exit(1)
    else:
        print("✅ All checks passed! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run quick demo:     python quickstart.py")
        print("  2. Run interactive:    python demo.py")
        print("  3. Full training:      python train.py --help")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
