"""
Quantum-Enhanced Drug Discovery Package
Hybrid quantum-classical models for molecular property prediction
"""

__version__ = "1.0.0"
__author__ = "Quantum ML Lab"

from src.config import ExperimentConfig, get_experiment_config
from src.hybrid_model import HybridQGNNModel, ClassicalGNNBaseline
from src.data_loader import DrugDiscoveryDataLoader, MoleculeNetLoader
from src.trainer import Trainer
from src.benchmark import BenchmarkRunner

__all__ = [
    'ExperimentConfig',
    'get_experiment_config',
    'HybridQGNNModel',
    'ClassicalGNNBaseline',
    'DrugDiscoveryDataLoader',
    'MoleculeNetLoader',
    'Trainer',
    'BenchmarkRunner',
]
