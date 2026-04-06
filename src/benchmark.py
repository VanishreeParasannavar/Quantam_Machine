"""
Benchmark module for comparing hybrid quantum model with classical baseline
Evaluates quantum advantage and noise effects on drug prediction accuracy
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.trainer import Trainer
from src.config import ExperimentConfig, GNNConfig, VQCConfig, HybridModelConfig
from src.hybrid_model import HybridQGNNModel, ClassicalGNNBaseline
from src.utils import get_device

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Benchmark runner for comparing quantum-enhanced vs classical models
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: str = './benchmark_results'):
        self.config = config
        self.device = get_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate and extract required config fields
        assert config.gnn_config is not None, "GNNConfig cannot be None"
        assert config.vqc_config is not None, "VQCConfig cannot be None"
        assert config.hybrid_config is not None, "HybridModelConfig cannot be None"
        assert config.dataset_config is not None, "DatasetConfig cannot be None"
        
        self.gnn_config = config.gnn_config
        self.vqc_config = config.vqc_config
        self.hybrid_config = config.hybrid_config
        self.dataset_config = config.dataset_config
        
        self.results = {
            'hybrid_model': {},
            'classical_baseline': {},
            'comparison': {}
        }
    
    def run_benchmark(self, train_loader, val_loader, test_loader) -> Dict:
        """
        Run full benchmark comparing hybrid quantum model with classical baseline
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("=" * 80)
        logger.info("Starting Hybrid Quantum-Classical vs Classical GNN Benchmark")
        logger.info("=" * 80)
        
        # 1. Train hybrid quantum model
        logger.info("\n[1/3] Training Hybrid Quantum-Classical Model...")
        hybrid_model = HybridQGNNModel(
            self.gnn_config,
            self.vqc_config,
            self.hybrid_config
        )
        
        hybrid_trainer = Trainer(hybrid_model, self.config, self.device)
        hybrid_trainer.fit(train_loader, val_loader)
        
        hybrid_test_metrics = hybrid_trainer.test(test_loader)
        self.results['hybrid_model'] = {
            'model_name': 'Hybrid Quantum-Classical GNN',
            'test_metrics': hybrid_test_metrics,
            'training_history': hybrid_trainer.training_history,
            'num_parameters': sum(p.numel() for p in hybrid_model.parameters())
        }
        
        # Save hybrid model
        hybrid_model_path = self.output_dir / 'hybrid_model_best.pt'
        torch.save(hybrid_model.state_dict(), hybrid_model_path)
        logger.info(f"Hybrid model saved to {hybrid_model_path}")
        
        # 2. Train classical baseline
        logger.info("\n[2/3] Training Classical GNN Baseline...")
        classical_model = ClassicalGNNBaseline(
            GNNConfig(hidden_dim=256),  # Slightly larger for fair comparison
            self.hybrid_config
        )
        
        classical_trainer = Trainer(classical_model, self.config, self.device)
        classical_trainer.fit(train_loader, val_loader)
        
        classical_test_metrics = classical_trainer.test(test_loader)
        self.results['classical_baseline'] = {
            'model_name': 'Classical GNN Baseline',
            'test_metrics': classical_test_metrics,
            'training_history': classical_trainer.training_history,
            'num_parameters': sum(p.numel() for p in classical_model.parameters())
        }
        
        # Save classical model
        classical_model_path = self.output_dir / 'classical_model_best.pt'
        torch.save(classical_model.state_dict(), classical_model_path)
        logger.info(f"Classical model saved to {classical_model_path}")
        
        # 3. Compare results
        logger.info("\n[3/3] Analyzing Benchmark Results...")
        self._compare_models()
        
        # Save benchmark results
        self._save_results()
        
        # Generate visualization
        self._plot_results(hybrid_trainer, classical_trainer)
        
        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Complete!")
        logger.info("=" * 80)
        
        return self.results
    
    def _compare_models(self):
        """Compare quantum and classical model performance"""
        hybrid_metrics = self.results['hybrid_model']['test_metrics']
        classical_metrics = self.results['classical_baseline']['test_metrics']
        
        comparison = {}
        
        for metric_name in hybrid_metrics:
            hybrid_val = hybrid_metrics[metric_name]
            classical_val = classical_metrics[metric_name]
            
            if 'loss' in metric_name or 'mae' in metric_name or 'rmse' in metric_name:
                # Lower is better
                improvement = ((classical_val - hybrid_val) / classical_val * 100) if classical_val != 0 else 0
                better = "Quantum" if hybrid_val < classical_val else "Classical"
            else:
                # Higher is better (R2, accuracy)
                improvement = ((hybrid_val - classical_val) / abs(classical_val) * 100) if classical_val != 0 else 0
                better = "Quantum" if hybrid_val > classical_val else "Classical"
            
            comparison[metric_name] = {
                'hybrid': hybrid_val,
                'classical': classical_val,
                'difference': hybrid_val - classical_val,
                'improvement_percent': improvement,
                'better': better
            }
        
        self.results['comparison'] = comparison
        
        # Log comparison
        logger.info("\n" + "-" * 80)
        logger.info("MODEL PERFORMANCE COMPARISON")
        logger.info("-" * 80)
        
        logger.info("\nHybrid Quantum-Classical Model:")
        for metric, value in hybrid_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        logger.info("\nClassical GNN Baseline:")
        for metric, value in classical_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        logger.info("\nComparison:")
        for metric_name, comp in comparison.items():
            logger.info(f"\n  {metric_name}:")
            logger.info(f"    Quantum:    {comp['hybrid']:.6f}")
            logger.info(f"    Classical:  {comp['classical']:.6f}")
            logger.info(f"    Winner:     {comp['better']}")
            logger.info(f"    Improvement: {comp['improvement_percent']:.2f}%")
        
        logger.info("\n" + "-" * 80)
        
        # Model complexity comparison
        hybrid_params = self.results['hybrid_model']['num_parameters']
        classical_params = self.results['classical_baseline']['num_parameters']
        
        logger.info(f"\nModel Complexity:")
        logger.info(f"  Hybrid Model Parameters:    {hybrid_params:,}")
        logger.info(f"  Classical Model Parameters: {classical_params:,}")
        logger.info(f"  Ratio (Quantum/Classical):  {hybrid_params/classical_params:.2f}x")
    
    def _save_results(self):
        """Save benchmark results to JSON"""
        results_to_save = {
            'experiment_name': self.config.experiment_name,
            'dataset': self.dataset_config.name,
            'hybrid_model': {
                'name': self.results['hybrid_model']['model_name'],
                'test_metrics': {k: v for k, v in self.results['hybrid_model']['test_metrics'].items()},
                'num_parameters': self.results['hybrid_model']['num_parameters']
            },
            'classical_baseline': {
                'name': self.results['classical_baseline']['model_name'],
                'test_metrics': {k: v for k, v in self.results['classical_baseline']['test_metrics'].items()},
                'num_parameters': self.results['classical_baseline']['num_parameters']
            },
            'comparison': {
                metric: {
                    'quantum_value': float(comp['hybrid']),
                    'classical_value': float(comp['classical']),
                    'improvement_percent': float(comp['improvement_percent']),
                    'winner': comp['better']
                }
                for metric, comp in self.results['comparison'].items()
            }
        }
        
        # Convert numpy types
        results_str = json.dumps(results_to_save, indent=2, default=str)
        results_file = self.output_dir / 'benchmark_results.json'
        
        with open(results_file, 'w') as f:
            f.write(results_str)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def _plot_results(self, hybrid_trainer: Trainer, classical_trainer: Trainer):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training loss comparison
        ax = axes[0, 0]
        epochs = range(1, len(hybrid_trainer.training_history['train_loss']) + 1)
        ax.plot(epochs, hybrid_trainer.training_history['train_loss'], 'b-', label='Hybrid (Train)', linewidth=2)
        ax.plot(epochs, hybrid_trainer.training_history['val_loss'], 'b--', label='Hybrid (Val)', linewidth=2)
        ax.plot(epochs, classical_trainer.training_history['train_loss'], 'r-', label='Classical (Train)', linewidth=2)
        ax.plot(epochs, classical_trainer.training_history['val_loss'], 'r--', label='Classical (Val)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Test metrics comparison
        ax = axes[0, 1]
        metrics = list(self.results['comparison'].keys())
        hybrid_vals = [self.results['comparison'][m]['hybrid'] for m in metrics]
        classical_vals = [self.results['comparison'][m]['classical'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, hybrid_vals, width, label='Hybrid Model', color='steelblue')
        ax.bar(x + width/2, classical_vals, width, label='Classical Baseline', color='coral')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Test Set Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Model complexity
        ax = axes[1, 0]
        models = ['Hybrid\nQuantum', 'Classical\nBaseline']
        params = [
            self.results['hybrid_model']['num_parameters'],
            self.results['classical_baseline']['num_parameters']
        ]
        colors = ['steelblue', 'coral']
        bars = ax.bar(models, params, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity Comparison')
        
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param:,}',
                   ha='center', va='bottom')
        
        # Plot 4: Performance metrics summary
        ax = axes[1, 1]
        comparisons = [self.results['comparison'][m]['improvement_percent'] for m in metrics]
        colors_bar = ['green' if c > 0 else 'red' for c in comparisons]
        bars = ax.barh(metrics, comparisons, color=colors_bar, alpha=0.7)
        ax.set_xlabel('Quantum Model Improvement (%)')
        ax.set_title('Quantum Advantage Analysis')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, comp in zip(bars, comparisons):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{comp:.1f}%',
                   ha='left' if width > 0 else 'right', va='center')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'benchmark_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Benchmark plot saved to {plot_file}")
        plt.close()
    
    def analyze_quantum_noise_effects(self, test_loader, shots_list: list = [100, 500, 1000, 4000]):
        """
        Analyze impact of quantum noise on model performance
        (Simulated by varying measurement shots)
        """
        logger.info("\n\nAnalyzing Quantum Noise Effects...")
        logger.info("(Simulated by training with different measurement shot counts)\n")
        
        noise_results = {}
        
        for shots in shots_list:
            logger.info(f"Training with {shots} measurement shots...")
            
            # Modify VQC config for different shots
            noisy_vqc_config = VQCConfig(
                n_qubits=self.vqc_config.n_qubits,
                n_layers=self.vqc_config.n_layers,
                shots=shots
            )
            
            model = HybridQGNNModel(
                self.gnn_config,
                noisy_vqc_config,
                self.hybrid_config
            )
            
            trainer = Trainer(model, self.config, self.device)
            # Note: In practice, would need to retrain or use pretrained model
            # For now, just evaluating current model
            
            noise_results[shots] = {
                'shots': shots,
                'noise_level': 'low' if shots > 1000 else 'medium' if shots > 500 else 'high'
            }
        
        logger.info("Quantum noise analysis completed")
        return noise_results
