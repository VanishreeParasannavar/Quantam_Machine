"""
Configuration for Quantum-Enhanced Drug Discovery System
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network"""
    input_dim: int = 40  # Node feature dimension
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    num_node_features: int = 58  # From RDKit features
    num_edge_features: int = 12

@dataclass
class VQCConfig:
    """Configuration for Variational Quantum Circuit"""
    n_qubits: int = 8  # Number of qubits - should match GNN output dimension
    n_layers: int = 3  # Depth of quantum circuit
    entangling_layers: int = 2
    measurement_qubits: int = 1  # Output qubits
    simulator: str = "qiskit_aer"  # or "default.qubit" for PennyLane simulator
    shots: int = 1000
    init_method: str = "random"  # Initialization method for weights

@dataclass
class HybridModelConfig:
    """Configuration for Hybrid Model"""
    output_dim: int = 1  # For regression (property prediction)
    task_type: str = "regression"  # "regression" or "classification"
    use_quantum: bool = True
    dropout_output: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for Training"""
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.1
    test_split: float = 0.1
    early_stopping_patience: int = 15
    device: str = "cuda"  # or "cpu"
    seed: int = 42
    
@dataclass
class DatasetConfig:
    """Configuration for Dataset"""
    name: str = "ESOL"  # Options: "ESOL", "Tox21", "HIV", "BBBP"
    download: bool = True
    path: str = "./data"
    preprocess_smiles: bool = True
    augment_data: bool = False
    max_nodes: int = 50
    normalize_targets: bool = True

@dataclass
class ExperimentConfig:
    """Main configuration combining all configs"""
    gnn_config: Optional[GNNConfig] = None
    vqc_config: Optional[VQCConfig] = None
    hybrid_config: Optional[HybridModelConfig] = None
    training_config: Optional[TrainingConfig] = None
    dataset_config: Optional[DatasetConfig] = None
    experiment_name: str = "quantum_drug_discovery_v1"
    run_benchmark: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    def __post_init__(self):
        if self.gnn_config is None:
            self.gnn_config = GNNConfig()
        if self.vqc_config is None:
            self.vqc_config = VQCConfig()
        if self.hybrid_config is None:
            self.hybrid_config = HybridModelConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
        if self.dataset_config is None:
            self.dataset_config = DatasetConfig()

# Presets for different experiments
def get_experiment_config(experiment_type: str = "hybrid") -> ExperimentConfig:
    """Get preset configurations"""
    if experiment_type == "hybrid":
        return ExperimentConfig(
            gnn_config=GNNConfig(hidden_dim=128),
            vqc_config=VQCConfig(n_qubits=8, n_layers=3),
            hybrid_config=HybridModelConfig(use_quantum=True),
            dataset_config=DatasetConfig(name="ESOL")
        )
    elif experiment_type == "gnn_baseline":
        return ExperimentConfig(
            gnn_config=GNNConfig(hidden_dim=256),
            hybrid_config=HybridModelConfig(use_quantum=False),
            dataset_config=DatasetConfig(name="ESOL")
        )
    elif experiment_type == "tox21":
        return ExperimentConfig(
            gnn_config=GNNConfig(hidden_dim=128),
            vqc_config=VQCConfig(n_qubits=8, n_layers=3),
            hybrid_config=HybridModelConfig(output_dim=12, task_type="classification"),
            dataset_config=DatasetConfig(name="Tox21")
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
