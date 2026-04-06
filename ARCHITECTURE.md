# Architecture & Design Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           QUANTUM-ENHANCED DRUG DISCOVERY SYSTEM            │
└─────────────────────────────────────────────────────────────┘

INPUT LAYER:
└─ SMILES String → RDKit Molecular Parser

PREPROCESSING LAYER:
├─ Node Feature Extraction (58-dim vectors per atom)
├─ Edge Feature Extraction (12-dim vectors per bond)
├─ Molecular Descriptors (MW, LogP, etc.)
└─ Graph Normalization & Padding

CLASSICAL-QUANTUM LAYERS:
┌──────────────────────────────────────────┐
│   Graph Neural Network (Classical)       │
│  ├─ GCN/GAT Convolution Layers          │
│  ├─ Batch Normalization                  │
│  ├─ Dropout Regularization              │
│  └─ Global Graph Pooling                │
│                ↓                         │
│  Output: Graph-level embeddings         │
│  Dimension: 128                         │
└──────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────┐
│ Variational Quantum Circuit (Quantum)    │
│  ├─ Feature Angle Encoding              │
│  ├─ Parametrized Rotation Gates         │
│  ├─ Entangling Layers (CNOT/CRZ)       │
│  └─ Quantum Measurement (Pauli-Z)      │
│                                          │
│  Qubits: 8                              │
│  Layers: 3-5                            │
│  Measurement Mode: Expectation Values   │
│                ↓                         │
│  Output: Quantum-processed features    │
│  Dimension: 128                         │
└──────────────────────────────────────────┘
                  ↓
OUTPUT LAYER:
├─ Feature Combination (Classical + Quantum)
├─ Dense Neural Layers
├─ Task-specific Activation
└─ Property Prediction (1D or Multi-task)

COMPARISON LAYER:
└─ Classical GNN Baseline (for quantum advantage analysis)
```

---

## Component Details

### 1. Molecular Input Processing

**File**: `utils.py` - `mol_to_graph_data()`

**Process**:
```python
SMILES String
    ↓
RDKit Molecule Object
    ↓
├─ Node Features (per atom)
│  ├─ Atom type (one-hot: C,N,O,S,F,Cl,Br,I,P)
│  ├─ Degree
│  ├─ Formal charge
│  ├─ Aromaticity
│  ├─ Hybridization
│  ├─ Total hydrogens
│  └─ Connectivity
│
├─ Edge Features (per bond)
│  ├─ Bond type (one-hot: Single,Double,Triple,Aromatic)
│  └─ Bond properties
│
└─ Molecular Descriptors
   ├─ Molecular Weight
   ├─ LogP (Lipophilicity)
   ├─ Rotatable bonds
   ├─ H-bond donors/acceptors
   ├─ Aromatic rings
   └─ Atom/bond count
```

**Output**: Undirected molecular graph with node/edge/global features

### 2. Graph Neural Network Encoder

**File**: `gnn_encoder.py`

#### Architecture Option A: GCNEncoder
```
Input: Node features [N, 58]
    ↓
GCN Layer 1 (58→128) + BatchNorm + ReLU
    ↓
GCN Layer 2 (128→128) + BatchNorm + ReLU
    ↓
GCN Layer 3 (128→128) + BatchNorm + ReLU + MultiheadAttention
    ↓
Global Mean Pooling
    ↓
Output: Graph embedding [B, 128]
```

#### Architecture Option B: Attention-based GNN
```
Input: Node features [N, 58]
    ↓
GAT Layer 1 (58→128, 4 heads)
    ↓
GAT Layer 2 (512→128, 4 heads)
    ↓
GAT Layer 3 (512→128, 1 head)
    ↓
Global Mean Pooling
    ↓
Output: Graph embedding [B, 128]
```

**Key Features**:
- Learns to weight important atoms
- Handles variable graph sizes
- Skip connections (residual)
- Graph-level representation

### 3. Variational Quantum Circuit

**File**: `quantum_circuit.py` - `QuantumCircuitLayer`

#### Circuit Structure (per layer)
```
Input features x[0:8]
    ↓
[Encoding Layer]
├─ RX(x[i]) on qubit i        # Angle encoding
└─ RZ(θ_enc[i]) on qubit i   # Learned encoding params

    ↓
[Variational Layer]
├─ RX(θ_var[i,0]) on qubit i  # Learned rotations
├─ RY(θ_var[i,1]) on qubit i
├─ RZ(θ_var[i,2]) on qubit i
└─ CNOT ladder (entanglement)

    ↓
[Measurement]
├─ Return ⟨Z₀⟩ (Pauli-Z expectation on qubit 0)
└─ Backprop through measurement
```

**Parameters**:
- Encoding params: [n_layers, n_qubits]
- Variational params: [n_layers, n_qubits, 3]
- Entangling params: [n_layers, entangling_layers, ⌊n_qubits/2⌋]

**Backends**:
- `default.qubit` (PennyLane simulator)
- `qiskit_aer` (IBM Qiskit simulator)
- Real quantum hardware (with proper backend)

### 4. Hybrid Integration

**File**: `hybrid_model.py`

```python
class HybridQGNNModel(nn.Module):
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Classical GNN processing
        node_emb, graph_emb = self.gnn_encoder(x, edge_index, edge_attr, batch)
        # graph_emb shape: [batch_size, 128]
        
        # To quantum space
        quantum_input = tanh(self.to_quantum(graph_emb))  # Normalize to [-1,1]
        # quantum_input shape: [batch_size, 8]
        
        # Quantum processing
        quantum_output = self.quantum_layer(quantum_input)  # Execute VQC
        # quantum_output shape: [batch_size, 1]
        
        # Back from quantum
        quantum_features = self.from_quantum(quantum_output)
        # quantum_features shape: [batch_size, 128]
        
        # Combine
        combined = graph_emb + 0.5 * quantum_features
        
        # Classical output head
        output = self.output_head(combined)  # → property prediction
        return output, intermediates
```

### 5. Training Pipeline

**File**: `trainer.py`

```
For each epoch:
    │
    ├─ [Training Phase]
    │  ├─ Load batch: molecules → graphs
    │  ├─ Forward: GNN → Quantum → Output
    │  ├─ Compute loss: MSE (regression) or BCE (classification)
    │  ├─ Backward: Compute gradients
    │  ├─ Update: Optimizer step (AdamW)
    │  └─ Record: Training metrics
    │
    ├─ [Validation Phase]
    │  ├─ Disable dropout/BN training mode
    │  ├─ Evaluate: Predictions vs targets
    │  ├─ Compute: MAE, RMSE, R² (regression) or Accuracy (classification)
    │  ├─ Check: Early stopping condition
    │  └─ Record: Validation metrics
    │
    └─ [Learning Rate Scheduling]
       └─ Reduce LR if validation loss plateaus (ReduceLROnPlateau)
```

### 6. Data Loading Pipeline

**File**: `data_loader.py`

```
Dataset Request (ESOL)
    ↓
Check cache (./data/ESOL_processed.pkl)
    ├─ YES: Load cached data (fast)
    └─ NO: Download & process
        ├─ Download CSV from MoleculeNet
        ├─ Parse SMILES strings
        ├─ Convert to molecular graphs
        ├─ Extract features
        ├─ Normalize
        └─ Cache for future use
    ↓
Split: Train (80%) → Val (10%) → Test (10%)
    ↓
Create DataLoaders with batching
    ↓
On access: Batch collation with graph padding
```

### 7. Benchmarking

**File**: `benchmark.py` - `BenchmarkRunner`

```
Compare: Quantum Model vs Classical Model
    │
    ├─ Train Hybrid Quantum-Classical Model
    │  └─ Test & record metrics
    │
    ├─ Train Classical GNN Baseline
    │  └─ Test & record metrics
    │
    ├─ Analyze Quantum Advantage
    │  ├─ Compare: MSE, R², Accuracy
    │  ├─ Calculate: % improvement
    │  └─ Identify: Winner per metric
    │
    ├─ Compare Complexity
    │  ├─ Count parameters
    │  ├─ Compare model size
    │  └─ Analyze efficiency
    │
    ├─ Generate Plots
    │  ├─ Training curves
    │  ├─ Test metrics bar charts
    │  ├─ Model complexity
    │  └─ Quantum advantage heatmap
    │
    └─ Save Results (JSON + PNG)
```

---

## Data Flow Example: Single Molecule

```python
# Input: Ibuprofen SMILES
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"

# Step 1: SMILES → RDKit Mol object
mol = Chem.MolFromSmiles(smiles)
n_atoms = 13, n_bonds = 14

# Step 2: Extract features
node_features.shape = (13, 58)       # 13 atoms, 58 features each
edge_index.shape = (2, 28)           # 14 bonds × 2 (undirected)
edge_features.shape = (28, 12)       # 28 edges, 12 features each

# Step 3: Batch & pad (for GPU efficiency)
# Pad to max_nodes=50
node_features_padded.shape = (50, 58)
edge_index_padded.shape = (2, 200)
batch_idx = [0, 0, ..., 0]          # All from same molecule

# Step 4: GNN processing
# Input: node_features_padded, edge_index_padded
# Process through 3 GCN layers
gnn_output.shape = (128,)            # Graph-level embedding

# Step 5: Quantum processing
# Input: tanh(normalize(gnn_output)) → [8 values for 8 qubits]
# Execute VQC with 3 layers, measure on qubit 0
quantum_output = ⟨Z₀⟩ = some value in [-1, 1]
quantum_output.shape = (1,)

# Step 6: Combine & predict
combined = gnn_output + 0.5 * denormalize(quantum_output)
# Dense layers with dropout
prediction = output_head(combined)
prediction = 0.45                    # log solubility (normalized)

# Step 7: Denormalize
actual_solubility = denormalize(0.45) ≈ -3.5 (mol/L scale)
```

---

## Configuration Hierarchy

```
ExperimentConfig (top level)
├── gnn_config: GNNConfig
│   ├── hidden_dim: 128
│   ├── num_layers: 3
│   ├── dropout: 0.1
│   └── num_node_features: 58
│
├── vqc_config: VQCConfig
│   ├── n_qubits: 8
│   ├── n_layers: 3
│   ├── shots: 1000
│   └── simulator: "default.qubit"
│
├── hybrid_config: HybridModelConfig
│   ├── output_dim: 1
│   ├── task_type: "regression"
│   └── use_quantum: True
│
├── training_config: TrainingConfig
│   ├── learning_rate: 0.001
│   ├── batch_size: 32
│   ├── num_epochs: 100
│   └── early_stopping_patience: 15
│
└── dataset_config: DatasetConfig
    ├── name: "ESOL"
    ├── path: "./data"
    ├── max_nodes: 50
    └── normalize_targets: True
```

---

## Error Handling & Robustness

### Input Validation
```
SMILES String → RDKit Parser
    ├─ Valid SMILES? → Continue
    └─ Invalid? → Skip molecule, log warning

Molecular Graph → Feature Extraction
    ├─ Expected features? → Continue
    └─ Unexpected? → Use defaults, log warning
```

### Runtime Error Handling
```python
try:
    # Process dangerous operations
    quantum_output = quantum_circuit(features)
except QuantumError:
    # Fallback to classical features
    quantum_output = classical_fallback(features)
    
try:
    # Training step
    loss.backward()
    optimizer.step()
except NumericalError:
    # Clip gradients, reduce learning rate
    clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Monitoring
- Early stopping prevents overfitting
- Learning rate scheduling adapts to plateaus
- Checkpointing saves best model
- Logging tracks all key events

---

## Performance Optimization

### Memory Efficiency
- Graph padding to fixed size (no fragmentation)
- Batch collation (minimize padding waste)
- Gradient accumulation option
- Model compression (pruning optional)

### Computation Speed
- GPU acceleration (CUDA)
- Quantum circuit caching
- Vectorized operations (PyTorch)
- Quantization (future optimization)

### Scalability
- Distributed training ready (DataParallel/DistributedDataParallel)
- Lazy data loading
- Streaming gradients
- Ensemble models

---

## Extension Points

### Add New GNN Encoder
```python
# In gnn_encoder.py
class GraphSAGEEncoder(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([SAGEConv(...) for _ in range(config.num_layers)])
    
    def forward(self, x, edge_index, batch):
        # Implementation
        return node_embeddings, graph_embedding
```

### Add New Quantum Circuit
```python
# In quantum_circuit.py
class QAOACircuit(nn.Module):
    def __init__(self, config):
        # Initialize QAOA parameters
    
    def forward(self, x):
        # QAOA implementation
        return measured_value
```

### Add New Dataset
```python
# In data_loader.py
DATASETS['MyDataset'] = {
    'url': 'https://...',
    'task': 'regression',
    'target_col': 'property'
}
```

---

## Testing Strategy

```
Unit Tests (test_models.py)
├─ Component Tests
│  ├─ GNNEncoder forward pass
│  ├─ QuantumCircuit execution
│  ├─ HybridModel pipeline
│  └─ ClassicalBaseline inference
│
├─ Utility Tests
│  ├─ SMILES parsing
│  ├─ Graph creation
│  ├─ Feature normalization
│  └─ Data loading
│
└─ Integration Tests
   ├─ Full training loop
   ├─ Validation evaluation
   ├─ Test evaluation
   └─ Checkpoint save/load
```

---

## Future Enhancements

1. **Quantum Hardware**: Deploy on real quantum processors
2. **Advanced Circuits**: QAOA, VQE, Parameterized Quantum Kernels
3. **Transfer Learning**: Pre-trained GNN encoders
4. **Multi-task Learning**: Joint training on multiple properties
5. **Federated Learning**: Distributed training across institutions
6. **AutoML**: Hyperparameter optimization
7. **Interpretability**: Circuit visualization & impact analysis
8. **Uncertainty Quantification**: Bayesian variants

---

**This architecture provides a solid foundation for quantum machine learning on drug discovery tasks.**

