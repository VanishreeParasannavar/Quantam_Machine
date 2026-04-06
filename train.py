"""
Main training script for Quantum-Enhanced Drug Discovery System
"""
import torch
import logging
import argparse
from pathlib import Path

from src.config import get_experiment_config, ExperimentConfig
from src.data_loader import DrugDiscoveryDataLoader
from src.hybrid_model import HybridQGNNModel
from src.trainer import Trainer
from src.benchmark import BenchmarkRunner
from src.utils import setup_logging, set_seed, get_device

def main(args):
    """Main training function"""
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(args.seed)
    device = get_device()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration for: {args.experiment}")
    config = get_experiment_config(args.experiment)
    
    # Validate config fields are not None
    assert config.dataset_config is not None, "DatasetConfig cannot be None"
    assert config.training_config is not None, "TrainingConfig cannot be None"
    assert config.gnn_config is not None, "GNNConfig cannot be None"
    assert config.vqc_config is not None, "VQCConfig cannot be None"
    assert config.hybrid_config is not None, "HybridModelConfig cannot be None"
    
    config.dataset_config.name = args.dataset
    config.training_config.batch_size = args.batch_size
    config.training_config.num_epochs = args.epochs
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    data_loader = DrugDiscoveryDataLoader(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    if args.benchmark:
        # Run full benchmark
        logger.info("\nRunning Hybrid Quantum-Classical vs Classical Baseline Benchmark")
        benchmark = BenchmarkRunner(config, output_dir=args.output_dir)
        results = benchmark.run_benchmark(train_loader, val_loader, test_loader)
        
        # Analyze noise effects
        if args.analyze_noise:
            benchmark.analyze_quantum_noise_effects(test_loader)
    else:
        # Train single model
        logger.info(f"\nTraining {args.experiment} model on {args.dataset}...")
        
        model = HybridQGNNModel(
            config.gnn_config,
            config.vqc_config,
            config.hybrid_config
        )
        
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        trainer = Trainer(model, config, device)
        trainer.fit(train_loader, val_loader)
        
        # Test
        logger.info("\nTesting on test set...")
        test_metrics = trainer.test(test_loader)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_checkpoint(args.experiment)
        trainer.save_training_history(str(output_dir / f'{args.experiment}_history.json'))
        
        logger.info(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum-Enhanced Drug Discovery System"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="hybrid",
        choices=["hybrid", "gnn_baseline", "tox21"],
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ESOL",
        choices=["ESOL", "Tox21", "HIV", "BBBP"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark comparing hybrid and classical models"
    )
    
    parser.add_argument(
        "--analyze-noise",
        action="store_true",
        help="Analyze quantum noise effects on predictions"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    main(args)
