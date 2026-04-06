#!/usr/bin/env python3
"""
Quick start script for Quantum-Enhanced Drug Discovery System
"""
import logging
from src.utils import setup_logging
from src.config import get_experiment_config
from src.data_loader import DrugDiscoveryDataLoader
from src.hybrid_model import HybridQGNNModel
from src.trainer import Trainer
from src.utils import set_seed, get_device

# Setup
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
set_seed(42)
device = get_device()

def quickstart():
    """Quick start training example"""
    logger.info("🧬 Quantum-Enhanced Drug Discovery - Quick Start")
    logger.info("=" * 80)
    
    # Configuration
    config = get_experiment_config('hybrid')
    
    # Validate config fields are not None
    assert config.training_config is not None, "TrainingConfig cannot be None"
    assert config.gnn_config is not None, "GNNConfig cannot be None"
    assert config.vqc_config is not None, "VQCConfig cannot be None"
    assert config.hybrid_config is not None, "HybridModelConfig cannot be None"
    
    config.training_config.num_epochs = 10  # Short demo
    config.training_config.batch_size = 16
    
    # Load data
    logger.info("\n[1/4] Loading ESOL dataset...")
    data_loader = DrugDiscoveryDataLoader(
        dataset_name='ESOL',
        batch_size=config.training_config.batch_size
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # Build model
    logger.info("\n[2/4] Building Hybrid Quantum-Classical Model...")
    model = HybridQGNNModel(
        config.gnn_config,
        config.vqc_config,
        config.hybrid_config
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {n_params:,}")
    
    # Train
    logger.info("\n[3/4] Training (10 epochs for demo)...")
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
    
    # Test
    logger.info("\n[4/4] Evaluating on test set...")
    test_metrics = trainer.test(test_loader)
    
    logger.info(f"\nQuick Start Complete! ✓")
    logger.info(f"Test Results:")
    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.6f}")
    
    logger.info(f"\nFor full benchmark, run:")
    logger.info(f"  python train.py --benchmark --epochs 100")

if __name__ == "__main__":
    quickstart()
