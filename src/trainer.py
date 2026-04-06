"""
Training pipeline for hybrid quantum-classical models
"""
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Optional, Tuple, Dict, List
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from src.utils import EarlyStopping, get_device, denormalize_targets
from src.config import TrainingConfig, ExperimentConfig

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for hybrid quantum-classical models
    Handles training, validation, and checkpointing
    """
    
    def __init__(self, model: nn.Module, config: ExperimentConfig, device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            config: ExperimentConfig with all configurations
            device: Device to train on (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        
        # Validate and extract required configs
        assert config.training_config is not None, "TrainingConfig cannot be None"
        assert config.hybrid_config is not None, "HybridModelConfig cannot be None"
        self.training_config = config.training_config
        self.hybrid_config = config.hybrid_config
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        if self.hybrid_config.task_type == 'classification':
            if self.hybrid_config.output_dim == 1:
                self.loss_fn = nn.BCELoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience
        )
        
        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = config.checkpoint_dir
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            device_batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, intermediates = self.model(
                device_batch['node_features'],
                device_batch['edge_index'],
                device_batch['edge_features'],
                device_batch['batch']
            )
            
            # Compute loss
            if self.hybrid_config.task_type == 'classification' and self.hybrid_config.output_dim == 1:
                loss = self.loss_fn(predictions, device_batch['targets'].float())
            else:
                loss = self.loss_fn(predictions, device_batch['targets'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            pbar.set_postfix({'loss': total_loss / batch_count})
        
        avg_loss = total_loss / batch_count
        return {'loss': avg_loss}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for batch in pbar:
                # Move batch to device
                device_batch = self._move_batch_to_device(batch)
                
                # Forward pass
                predictions, _ = self.model(
                    device_batch['node_features'],
                    device_batch['edge_index'],
                    device_batch['edge_features'],
                    device_batch['batch']
                )
                
                # Compute loss
                if self.hybrid_config.task_type == 'classification' and self.hybrid_config.output_dim == 1:
                    loss = self.loss_fn(predictions, device_batch['targets'].float())
                else:
                    loss = self.loss_fn(predictions, device_batch['targets'])
                
                total_loss += loss.item()
                batch_count += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(device_batch['targets'].cpu().numpy())
                
                pbar.set_postfix({'loss': total_loss / batch_count})
        
        avg_loss = total_loss / batch_count
        
        # Compute additional metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = {'loss': avg_loss}
        metrics.update(self._compute_metrics(predictions, targets))
        
        return metrics
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        if self.hybrid_config.task_type == 'regression':
            mae = float(np.mean(np.abs(predictions - targets)))
            rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
            r2 = float(1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)))
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        else:
            # Classification metrics
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = float(np.mean(predictions_binary == targets))
            
            return {
                'accuracy': accuracy
            }
    
    def fit(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """
        Fit model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.training_config.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_metrics'].append(val_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.6f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.6f}")
            for key, value in val_metrics.items():
                if key != 'loss':
                    logger.info(f"  Val {key}: {value:.6f}")
            
            # Save checkpoint if improved
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if self.early_stopping(val_metrics['loss'], epoch):
                logger.info("Early stopping triggered")
                break
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
    
    def test(self, test_loader) -> Dict[str, float]:
        """
        Test model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        logger.info("Running test evaluation...")
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for batch in pbar:
                device_batch = self._move_batch_to_device(batch)
                
                predictions, _ = self.model(
                    device_batch['node_features'],
                    device_batch['edge_index'],
                    device_batch['edge_features'],
                    device_batch['batch']
                )
                
                if self.hybrid_config.task_type == 'classification' and self.hybrid_config.output_dim == 1:
                    loss = self.loss_fn(predictions, device_batch['targets'].float())
                else:
                    loss = self.loss_fn(predictions, device_batch['targets'])
                
                total_loss += loss.item()
                batch_count += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(device_batch['targets'].cpu().numpy())
                
                pbar.set_postfix({'loss': total_loss / batch_count})
        
        avg_loss = total_loss / batch_count
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = {'loss': avg_loss}
        metrics.update(self._compute_metrics(predictions, targets))
        
        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        return metrics
    
    def save_checkpoint(self, checkpoint_name: str = 'model', is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = Path(self.checkpoint_dir) / f"{checkpoint_name}_epoch_{self.current_epoch}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = Path(self.checkpoint_dir) / f"{checkpoint_name}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def save_training_history(self, output_path: str):
        """Save training history to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        history = {
            'train_loss': [float(x) for x in self.training_history['train_loss']],
            'val_loss': [float(x) for x in self.training_history['val_loss']],
            'training_config': {
                'learning_rate': self.training_config.learning_rate,
                'batch_size': self.training_config.batch_size,
                'num_epochs': self.current_epoch + 1,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {output_path}")
