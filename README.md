# 🧬 Quantum-Enhanced Drug Discovery: Hybrid Quantum-Classical Models for Molecular Property Prediction

A cutting-edge implementation of **hybrid quantum-classical neural networks** for predicting molecular properties using **Variational Quantum Circuits (VQC)** and **Graph Neural Networks (GNN)**.

## 🚀 Project Overview

This system combines:
- **Classical ML**: Graph Neural Networks (PyTorch Geometric) for molecular feature extraction
- **Quantum Computing**: Variational Quantum Circuits (PennyLane) for quantum feature encoding
- **Benchmarking**: Comprehensive comparison with classical-only baselines

### Key Features
✅ Hybrid QGNN architecture for molecular property prediction  
✅ Support for multiple datasets (ESOL, Tox21, HIV, BBBP)  
✅ Full training pipeline with early stopping & checkpointing  
✅ Quantum-classical benchmark analysis  
✅ Noise effect simulation on quantum circuits  
✅ Reproducible results with seed management  

---

## 📋 System Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB+ RAM
- ~5GB disk space for datasets

---

## 🛠️ Installation

### Quick Setup (1 command!)

**Windows (any terminal)**:
```bash
setup.bat
```

or with PowerShell:
```powershell
.\setup.ps1
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### What This Does
- Creates isolated Python virtual environment
- Installs all 19 dependencies from `requirements.txt`
- Configures Pylance for import resolution

### 1. Clone/Setup Repository
```bash
cd e:\Quantam_Machine
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch==2.0.0` - Deep learning framework
- `torch-geometric==2.3.0` - Graph neural networks
- `pennylane==0.30.0` - Quantum computing framework
- `rdkit==2023.03.1` - Molecular processing
- `qiskit==0.40.0` - IBM Quantum framework

---

## 📁 Project Structure

```
Quantam_Machine/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utility functions
│   ├── data_loader.py         # Dataset loading (ESOL, Tox21, etc.)
│   ├── gnn_encoder.py         # Graph Neural Network encoders
│   ├── quantum_circuit.py     # Variational Quantum Circuit layers
│   ├── hybrid_model.py        # Hybrid Quantum-Classical model
│   ├── trainer.py             # Training pipeline
│   └── benchmark.py           # Benchmarking utilities
├── data/                      # Downloaded datasets
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── results/                   # Training results & checkpoints
├── train.py                   # Main training script
├── demo.py                    # Interactive demonstration
├── quickstart.py              # Quick start example
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Option 1: Quick Demo (5 minutes)
```bash
python quickstart.py
```
- Trains model for 10 epochs on ESOL dataset
- Provides baseline performance metrics

### Option 2: Interactive Demo
```bash
python demo.py
```
- Step-by-step walkthrough of model components
- Forward pass demonstration
- Molecule property prediction examples

### Option 3: Full Benchmark
```bash
python train.py --benchmark --epochs 100 --dataset ESOL
```
- Complete training with quantum and classical models
- Detailed performance comparison
- Visualization plots

---

## 📊 Supported Datasets

| Dataset | Task | Target | Samples |
|---------|------|--------|---------|
| **ESOL** | Regression | Log Solubility | ~1,144 |
| **Tox21** | Multi-task Classification | 12 toxicity endpoints | ~7,831 |
| **HIV** | Classification | HIV activity | ~41,127 |
| **BBBP** | Classification | Blood-Brain Barrier Permeability | ~2,053 |

All datasets are automatically downloaded from MoleculeNet.

---

## 🏋️ Training

### Basic Training
```python
from src.config import get_experiment_config
from src.data_loader import DrugDiscoveryDataLoader
from src.hybrid_model import HybridQGNNModel
from src.trainer import Trainer
from src.utils import get_device, set_seed

# Setup
set_seed(42)
device = get_device()

# Load configuration and data
config = get_experiment_config('hybrid')
data_loader = DrugDiscoveryDataLoader('ESOL', batch_size=32)

train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()
test_loader = data_loader.get_test_loader()

# Create model
model = HybridQGNNModel(
    config.gnn_config,
    config.vqc_config,
    config.hybrid_config
)

# Train
trainer = Trainer(model, config, device)
trainer.fit(train_loader, val_loader, num_epochs=100)

# Evaluate
test_metrics = trainer.test(test_loader)
print(test_metrics)
```

### Command Line Training
```bash
# Hybrid quantum-classical model (ESOL)
python train.py --experiment hybrid --dataset ESOL --epochs 100 --batch-size 32

# Classical baseline comparison
python train.py --experiment gnn_baseline --dataset ESOL --epochs 100

# Multi-task classification (Tox21)
python train.py --experiment tox21 --dataset Tox21 --epochs 100

# Full benchmark
python train.py --benchmark --analyze-noise --epochs 100
```

### Configuration

Modify `src/config.py` to customize:

```python
from src.config import ExperimentConfig, GNNConfig, VQCConfig, HybridModelConfig, TrainingConfig

# Custom configuration
config = ExperimentConfig(
    gnn_config=GNNConfig(
        hidden_dim=256,
        num_layers=4,
        dropout=0.2
    ),
    vqc_config=VQCConfig(
        n_qubits=8,
        n_layers=3,
        shots=1000
    ),
    training_config=TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        num_epochs=200,
        early_stopping_patience=20
    ),
    dataset_config=DatasetConfig(name="ESOL")
)
```

---

## 🔬 Model Architecture

### Hybrid Quantum-Classical GNN

```
Input: Molecular SMILES/Graph
    ↓
[Classical GNN Encoder]
    • Graph Convolution Layers
    • Node embeddings extraction
    • Graph-level pooling
    ↓ (Graph embeddings)
[Quantum Circuit Layer]
    • Feature encoding to quantum states
    • Variational rotations
    • Entangling gates (CNOT)
    • Measurement (expectation values)
    ↓ (Quantum-processed features)
[Classical Output Head]
    • Dense layers
    • Task-specific activation
    ↓
Output: Molecular property prediction
```

### Key Components

#### 1. **GNN Encoder** (`gnn_encoder.py`)
- **GCNConv**: Graph Convolution Networks
- **GATConv**: Graph Attention Networks
- **Pooling**: Global mean/max pooling for graph-level representation

#### 2. **Quantum Circuit** (`quantum_circuit.py`)
- **Angle Encoding**: Classical features → quantum angles
- **Parametrized Rotations**: RX, RY, RZ gates
- **Entanglement**: CNOT and CRZ gates
- **Measurement**: Pauli-Z expectation values

#### 3. **Hybrid Model** (`hybrid_model.py`)
- Combines GNN + VQC for end-to-end learning
- Optional quantum layer for ablation studies
- Supports ensemble models

---

## 📈 Benchmark Results Example

```
MODEL PERFORMANCE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hybrid Quantum-Classical Model:
  loss: 0.152341
  mae: 0.285642
  rmse: 0.401234
  r2: 0.812456

Classical GNN Baseline:
  loss: 0.189234
  mae: 0.315789
  rmse: 0.445123
  r2: 0.765432

Quantum Advantage:
  Loss:  -19.5% (better)
  MAE:   -9.6% (better)
  RMSE:  -9.8% (better)
  R²:    +6.1% (better)

Model Complexity:
  Quantum Parameters:    1,245,632
  Classical Parameters:  2,048,576
  Ratio:                 0.61x (more efficient)
```

---

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/ -v

# Or run specific test
python -m pytest tests/test_models.py::TestHybridModel -v
```

Tests cover:
- GNN encoder forward passes
- Quantum circuit execution
- Hybrid model pipeline
- Molecular processing utilities
- Dataset loading

---

## 📊 Experiments

### Default Experiments

1. **Hybrid Model (ESOL)**
   ```bash
   python train.py --experiment hybrid --dataset ESOL
   ```
   Combines GNN + VQC for solubility prediction

2. **Classical Baseline (ESOL)**
   ```bash
   python train.py --experiment gnn_baseline --dataset ESOL
   ```
   Classical-only GNN for comparison

3. **Multi-task Learning (Tox21)**
   ```bash
   python train.py --experiment tox21 --dataset Tox21
   ```
   12-task toxicity prediction

4. **Full Benchmark**
   ```bash
   python train.py --benchmark --analyze-noise
   ```
   Complete quantum vs classical comparison

### Custom Experiments

Create custom experiments in `src/config.py`:
```python
def get_experiment_config(experiment_type):
    if experiment_type == "custom":
        return ExperimentConfig(
            gnn_config=GNNConfig(...),
            vqc_config=VQCConfig(...),
            ...
        )
```

---

## 🔄 Quantum Noise Analysis

The system includes simulation of quantum noise effects:

```python
from src.benchmark import BenchmarkRunner

benchmark = BenchmarkRunner(config)
noise_analysis = benchmark.analyze_quantum_noise_effects(
    test_loader,
    shots_list=[100, 500, 1000, 4000]
)
```

This simulates measurement noise by varying the number of quantum shots (circuit repetitions).

---

## 💾 Checkpointing & Resuming

### Save Checkpoints
```python
trainer.save_checkpoint('my_model', is_best=True)
```

### Load Checkpoint
```python
trainer.load_checkpoint('./checkpoints/my_model_best.pt')
```

### Results Structure
```
results/
├── hybrid_model_best.pt           # Model weights
├── classical_model_best.pt        # Classical baseline
├── benchmark_results.json         # Performance metrics
├── benchmark_comparison.png       # Visualization
└── training_history.json          # Training curves
```

---

## 🎨 Visualization

Benchmark automatically generates:
- **Training loss curves** (Quantum vs Classical)
- **Test metric comparison** (Bar plots)
- **Model complexity analysis**
- **Quantum advantage heatmap**

Example:
```bash
python train.py --benchmark
# Results saved to results/benchmark_comparison.png
```

---

## 📚 Documentation

### Key Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration management |
| `utils.py` | Molecular processing, normalization |
| `data_loader.py` | Dataset loading & preprocessing |
| `gnn_encoder.py` | Graph neural networks |
| `quantum_circuit.py` | Quantum circuits |
| `hybrid_model.py` | Hybrid model architecture |
| `trainer.py` | Training & evaluation |
| `benchmark.py` | Benchmarking utilities |

### Example: Custom Model

```python
from src.hybrid_model import HybridQGNNModel
from src.config import GNNConfig, VQCConfig, HybridModelConfig

# Custom configuration
gnn_cfg = GNNConfig(hidden_dim=512, num_layers=6)
vqc_cfg = VQCConfig(n_qubits=16, n_layers=5)
hybrid_cfg = HybridModelConfig(output_dim=12, task_type="classification")

model = HybridQGNNModel(gnn_cfg, vqc_cfg, hybrid_cfg)
```

---

## 🔍 Debugging

Enable verbose logging:

```python
import logging
from src.utils import setup_logging

setup_logging(level=logging.DEBUG)
```

Or via command line:
```bash
# Set LOG_LEVEL environment variable
export LOG_LEVEL=DEBUG
python train.py --experiment hybrid
```

---

## 🐛 Known Issues & Workarounds

| Issue | Workaround |\n|-------|------------|
| Qiskit simulator not available | Falls back to PennyLane default.qubit |\n| OOM on GPU | Reduce batch_size or use CPU |\n| Slow quantum circuit | Reduce n_qubits or n_layers |\n| Dataset download fails | Manual download from MoleculeNet |\n

---

## 🚀 Next Steps

1. **Quantum Hardware**: Deploy on IBM Quantum, Rigetti, or other providers
2. **Hyperparameter Search**: Grid/random search for optimal configurations
3. **Larger Datasets**: Scale to HIV and Tox21 datasets
4. **Advanced Circuits**: Implement QAOA, VQE variants
5. **Transfer Learning**: Pre-trained GNN → quantum circuits
6. **Ensemble Methods**: Multiple quantum architectures

---

## 📚 References

### Papers
- Quantum Machine Learning: https://arxiv.org/abs/2006.09904
- Graph Neural Networks: https://arxiv.org/abs/2003.00330
- Variational Quantum Circuits: https://arxiv.org/abs/1712.09762

### Frameworks
- PennyLane: https://pennylane.ai/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- RDKit: https://www.rdkit.org/
- Qiskit: https://qiskit.org/

### Datasets
- MoleculeNet: https://moleculenet.org/
- ChEMBL: https://www.ebi.ac.uk/chembl/

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More quantum circuit architectures
- Additional datasets
- Better noise models
- Performance optimizations
- Better documentation

---

## 📄 License

This project is provided as-is for research and educational purposes.

---

## 👨‍💻 Author & Support

**Quantum ML Lab** - Advanced Quantum Machine Learning Research

For issues, questions, or suggestions:
1. Check existing documentation
2. Review code comments and docstrings
3. Create detailed bug reports with reproducibility steps

---

## 🎯 Citation

If you use this system in research, please cite:
```bibtex
@software{quantum_drug_discovery_2024,
  title={Quantum-Enhanced Drug Discovery: Hybrid Quantum-Classical Models for Molecular Property Prediction},
  author={Quantum ML Lab},
  year={2024},
  url={https://github.com/...}
}
```

---

## 📞 Contact & Feedback

Share your results, improvements, or innovations!

**Happy Quantum Computing! 🚀🧬**
