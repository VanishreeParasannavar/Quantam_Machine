"""
Hybrid Quantum-Classical Model for Drug Discovery
Combines GNN encoder with Variational Quantum Circuit
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import HybridModelConfig, GNNConfig, VQCConfig
from src.gnn_encoder import GNNEncoder, AttentionGNNEncoder
from src.quantum_circuit import QuantumCircuitLayer, ClassicalQuantumHybrid

class HybridQGNNModel(nn.Module):
    """
    Hybrid Quantum-Classical GNN Model for Molecular Property Prediction
    
    Architecture:
    1. Input: Molecular graph (SMILES or 3D structure)
    2. GNN Encoder: Extract structural features using Graph Neural Networks
    3. Quantum Layer: Process features through Variational Quantum Circuit
    4. Classical Head: Output prediction for molecular property
    """
    
    def __init__(self, gnn_config: GNNConfig, vqc_config: VQCConfig,
                 hybrid_config: HybridModelConfig, **kwargs):
        super().__init__()
        self.gnn_config = gnn_config
        self.vqc_config = vqc_config
        self.hybrid_config = hybrid_config
        
        # 1. GNN Encoder
        self.gnn_encoder = GNNEncoder(gnn_config)
        
        # 2. Classical-Quantum Hybrid Layer
        self.use_quantum = hybrid_config.use_quantum
        if self.use_quantum:
            # Map GNN output to quantum input
            self.to_quantum = nn.Linear(gnn_config.hidden_dim, vqc_config.n_qubits)
            
            # Quantum circuit
            self.quantum_layer = QuantumCircuitLayer(vqc_config)
            
            # Map quantum output back to classical
            quantum_output_dim = 1  # Single measurement
            self.from_quantum = nn.Linear(quantum_output_dim, gnn_config.hidden_dim)
        else:
            # Skip quantum layer
            self.to_quantum = None
            self.quantum_layer = None
            self.from_quantum = None
        
        # 3. Classical output head
        head_dim = gnn_config.hidden_dim + (gnn_config.hidden_dim if self.use_quantum else 0)
        self.output_head = nn.Sequential(
            nn.Linear(gnn_config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(hybrid_config.dropout_output),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(hybrid_config.dropout_output),
            nn.Linear(128, hybrid_config.output_dim)
        )
        
        # Task-specific output layer
        if hybrid_config.task_type == "classification":
            self.final_activation = nn.Sigmoid() if hybrid_config.output_dim == 1 else nn.Softmax(dim=-1)
        else:
            self.final_activation = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through hybrid model
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            predictions: Output predictions [batch_size, output_dim]
            intermediates: Dictionary with intermediate activations for analysis
        """
        intermediates = {}
        
        # 1. GNN Encoding
        node_embeddings, graph_embedding = self.gnn_encoder(x, edge_index, edge_attr, batch)
        intermediates['gnn_embeddings'] = graph_embedding.detach()
        
        # 2. Optional Quantum Processing
        quantum_output = None
        if self.use_quantum:
            # Ensure quantum layers are initialized
            assert self.to_quantum is not None, "to_quantum layer must be initialized"
            assert self.quantum_layer is not None, "quantum_layer must be initialized"
            assert self.from_quantum is not None, "from_quantum layer must be initialized"
            
            # Transform to quantum space
            quantum_input = torch.tanh(self.to_quantum(graph_embedding))  # Normalize to [-1, 1]
            
            # Process through quantum circuit
            quantum_output = self.quantum_layer(quantum_input)  # [batch, 1]
            intermediates['quantum_output'] = quantum_output.detach()
            
            # Transform back from quantum space
            quantum_features = self.from_quantum(quantum_output)
            
            # Combine with classical features
            combined_features = graph_embedding + quantum_features * 0.5
            intermediates['combined_features'] = combined_features.detach()
        else:
            combined_features = graph_embedding
        
        # 3. Classical Output Head
        output = self.output_head(combined_features)
        
        # 4. Task-specific activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output, intermediates

class ClassicalGNNBaseline(nn.Module):
    """
    Classical GNN baseline for comparison
    Same GNN architecture but without quantum layer
    """
    
    def __init__(self, gnn_config: GNNConfig, hybrid_config: HybridModelConfig):
        super().__init__()
        self.gnn_config = gnn_config
        self.hybrid_config = hybrid_config
        
        # Use attention-based GNN for stronger baseline
        self.gnn_encoder = AttentionGNNEncoder(gnn_config)
        
        # Larger output head for classical model
        self.output_head = nn.Sequential(
            nn.Linear(gnn_config.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(hybrid_config.dropout_output),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(hybrid_config.dropout_output),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(hybrid_config.dropout_output),
            nn.Linear(128, hybrid_config.output_dim)
        )
        
        # Task-specific output layer
        if hybrid_config.task_type == "classification":
            self.final_activation = nn.Sigmoid() if hybrid_config.output_dim == 1 else nn.Softmax(dim=-1)
        else:
            self.final_activation = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through classical GNN baseline
        """
        intermediates = {}
        
        # GNN Encoding
        node_embeddings, graph_embedding = self.gnn_encoder(x, edge_index, edge_attr, batch)
        intermediates['gnn_embeddings'] = graph_embedding.detach()
        
        # Output prediction
        output = self.output_head(graph_embedding)
        
        # Task-specific activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output, intermediates

class EnsembleHybridModel(nn.Module):
    """
    Ensemble of hybrid models
    Improves robustness and generalization
    """
    
    def __init__(self, gnn_config: GNNConfig, vqc_config: VQCConfig,
                 hybrid_config: HybridModelConfig, num_models: int = 3):
        super().__init__()
        self.num_models = num_models
        
        self.models = nn.ModuleList()
        for i in range(num_models):
            model = HybridQGNNModel(gnn_config, vqc_config, hybrid_config)
            self.models.append(model)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Ensemble forward pass - average predictions
        """
        outputs = []
        all_intermediates = {}
        
        for i, model in enumerate(self.models):
            output, intermediates = model(x, edge_index, edge_attr, batch)
            outputs.append(output)
            all_intermediates[f'model_{i}'] = intermediates
        
        # Average ensemble predictions
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        ensemble_std = torch.std(torch.stack(outputs), dim=0)
        
        all_intermediates['ensemble_std'] = ensemble_std.detach()
        
        return ensemble_output, all_intermediates

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: nn.Module) -> dict:
    """Get model summary"""
    return {
        'total_parameters': count_parameters(model),
        'model_type': model.__class__.__name__,
        'trainable': sum(p.requires_grad for p in model.parameters()),
    }
