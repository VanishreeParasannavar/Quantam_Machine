# Getting Started Guide

Welcome to the Quantum-Enhanced Drug Discovery System! This guide will help you get up and running quickly.

## Prerequisites Check

Before starting, verify you have:
- Python 3.8 or higher: `python --version`
- pip package manager: `pip --version`
- ~8GB RAM available
- 5GB+ disk space

## Step 1: Environment Setup (5 minutes)

### Quick Setup (Recommended)

**Windows (Command Prompt)**:
```bash
setup.bat
```

**Windows (PowerShell)**:
```powershell
.\setup.ps1
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Manual Setup

**Windows (Command Prompt)**:
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Windows (PowerShell)**:
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure VS Code (Optional but Recommended)

After creating the virtual environment, VS Code should automatically detect it. If not:

1. Open Command Palette: `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `./venv/bin/python` or `./venv/Scripts/python`

A `pyrightconfig.json` file has been created to help Pylance resolve imports correctly.

## Step 2: Verify Installation (5 minutes)

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pennylane; print(f'PennyLane: {pennylane.__version__}')"
python -c "from rdkit import Chem; print('RDKit: OK')"

# Run tests
python -m pytest tests/test_models.py -v
```

## Step 3: Run Quick Demo (10 minutes)

```bash
# Option A: 10-epoch quick demo
python quickstart.py

# Option B: Interactive walkthrough
python demo.py

# Option C: Full benchmark
python train.py --benchmark --epochs 50 --dataset ESOL
```

## Common Issues & Solutions

### Issue: ImportError for torch_geometric
**Solution:**
```bash
pip install torch-geometric
```

### Issue: CUDA/GPU not detected
**Solution:** This is normal - the code defaults to CPU. To force GPU:
```bash
# Python script
import torch
print(torch.cuda.is_available())  # Should show True if available
```

### Issue: Memory error during training
**Solution:** Reduce batch size
```bash
python train.py --batch-size 16 --epochs 20
```

### Issue: Dataset download fails
**Solution:** Manual download (datasets cached automatically on first run)
```bash
# Datasets cache to: ./data/
# They download automatically from MoleculeNet on first run
```

## Next Commands

After verification, try:

```bash
# 1. Basic training (ESOL dataset)
python train.py --experiment hybrid --epochs 20

# 2. Compare quantum vs classical
python train.py --benchmark --epochs 30 --dataset ESOL

# 3. Try different dataset
python train.py --experiment hybrid --dataset HIV --epochs 20

# 4. Analyze noise effects
python train.py --benchmark --analyze-noise --epochs 20

# 5. Run tests
python -m pytest tests/ -v
```

## Project Structure Reference

```
├── train.py          ← Main entry point
├── quickstart.py     ← Quick 10-epoch demo
├── demo.py          ← Interactive tutorial
├── src/
│   ├── config.py     ← Configuration options
│   ├── data_loader.py    ← Dataset loading
│   ├── gnn_encoder.py    ← Graph neural networks
│   ├── quantum_circuit.py ← Quantum circuits
│   ├── hybrid_model.py    ← Main model
│   ├── trainer.py    ← Training loop
│   └── benchmark.py  ← Comparison tools
├── data/             ← Downloaded datasets (auto-created)
├── results/          ← Training results (auto-created)
└── tests/            ← Unit tests
```

## Expected Output

### After `python quickstart.py`:
```
🧬 Quantum-Enhanced Drug Discovery - Quick Start
================================================================================

[1/4] Loading ESOL dataset...
Successfully processed 1000 molecules

[2/4] Building Hybrid Quantum-Classical Model...
Model Parameters: 1,245,632

[3/4] Training (10 epochs for demo)...
Epoch 1/10
  Train Loss: 0.234567
  Val Loss: 0.198765
...

[4/4] Evaluating on test set...
Test Results:
  loss: 0.156789
  mae: 0.287654
  rmse: 0.403210
  r2: 0.823456

Quick Start Complete! ✓
```

## Training Timeline

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| Quick demo (10 epochs) | 15-30 min | 3-5 min |
| Standard training (50 epochs) | 1-2 hours | 10-15 min |
| Full benchmark (100 epochs) | 3-6 hours | 25-40 min |

## Next Steps

1. ✅ Run quickstart.py
2. ✅ Explore demo.py
3. ✅ Try different datasets:
   ```bash
   python train.py --dataset HIV --epochs 30
   python train.py --dataset Tox21 --epochs 30
   ```
4. ✅ Run benchmark:
   ```bash
   python train.py --benchmark --epochs 50
   ```
5. ✅ Modify configurations in `src/config.py`
6. ✅ Read full README.md for advanced usage

## Help & Debugging

### Verbose logging
```bash
export LOG_LEVEL=DEBUG
python train.py --experiment hybrid
```

### Check data
```python
from src.data_loader import DrugDiscoveryDataLoader
loader = DrugDiscoveryDataLoader('ESOL')
train_set, val_set, test_set = loader.load_dataset()
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
```

### Inspect model
```python
from src.config import get_experiment_config
from src.hybrid_model import HybridQGNNModel

config = get_experiment_config('hybrid')
model = HybridQGNNModel(config.gnn_config, config.vqc_config, config.hybrid_config)
print(model)  # See full architecture
```

## Tips & Tricks

1. **Faster iteration**: Use smaller datasets first
   ```bash
   python quickstart.py  # Uses ESOL with 10 epochs
   ```

2. **GPU acceleration**: Install CUDA and cuDNN, then:
   ```python
   torch.cuda.is_available()  # Should be True
   ```

3. **Reproducibility**: Always set seed
   ```python
   from src.utils import set_seed
   set_seed(42)
   ```

4. **Monitor memory**: Check RAM usage
   ```python
   import psutil
   print(f"Memory: {psutil.virtual_memory().percent}%")
   ```

## Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed: `pip install -r requirements.txt`
- [ ] Tests passing: `pytest tests/ -v`
- [ ] Can import main modules: `python -c "from src import *"`
- [ ] Can download dataset: Run any train.py command (tests download)
- [ ] GPU available (optional): Check `torch.cuda.is_available()`

**You're all set! 🚀 Start with `python quickstart.py`**
