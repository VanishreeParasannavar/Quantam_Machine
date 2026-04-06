"""
Unit tests for Quantum-Enhanced Drug Discovery System
"""
import unittest
import torch
import numpy as np
from pathlib import Path

from src.config import GNNConfig, VQCConfig, HybridModelConfig, get_experiment_config
from src.gnn_encoder import GNNEncoder, AttentionGNNEncoder
from src.quantum_circuit import QuantumCircuitLayer, ClassicalQuantumHybrid
from src.hybrid_model import HybridQGNNModel, ClassicalGNNBaseline
from src.utils import smiles_to_mol, mol_to_graph_data, pad_graph_data

class TestGNNEncoder(unittest.TestCase):
    """Test GNN encoder"""
    
    def setUp(self):
        self.config = GNNConfig()
        self.encoder = GNNEncoder(self.config)
    
    def test_forward_pass(self):
        """Test GNN forward pass"""
        batch_size = 2
        num_nodes = 50
        
        x = torch.randn(num_nodes, self.config.num_node_features)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        batch = torch.tensor([0]*25 + [1]*25)
        
        node_embeddings, graph_embedding = self.encoder(x, edge_index, batch=batch)
        
        self.assertEqual(node_embeddings.shape[0], num_nodes)
        self.assertEqual(graph_embedding.shape[0], batch_size)
        self.assertEqual(graph_embedding.shape[1], self.config.hidden_dim)

class TestQuantumCircuit(unittest.TestCase):
    """Test quantum circuit layer"""
    
    def setUp(self):
        self.config = VQCConfig(n_qubits=4, n_layers=1)
        self.layer = QuantumCircuitLayer(self.config)
    
    def test_forward_pass(self):
        """Test quantum circuit forward pass"""
        x = torch.randn(2, 4)
        output = self.layer(x)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1)

class TestHybridModel(unittest.TestCase):
    """Test hybrid model"""
    
    def setUp(self):
        self.config = get_experiment_config('hybrid')
        
        # Validate and extract config fields as typed attributes
        assert self.config.gnn_config is not None, "GNNConfig cannot be None"
        assert self.config.vqc_config is not None, "VQCConfig cannot be None"
        assert self.config.hybrid_config is not None, "HybridModelConfig cannot be None"
        
        self.gnn_config = self.config.gnn_config
        self.vqc_config = self.config.vqc_config
        self.hybrid_config = self.config.hybrid_config
        
        self.model = HybridQGNNModel(
            self.gnn_config,
            self.vqc_config,
            self.hybrid_config
        )
    
    def test_forward_pass(self):
        """Test hybrid model forward pass"""
        batch_size = 2
        num_nodes = 50
        
        x = torch.randn(num_nodes, self.gnn_config.num_node_features)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        edge_attr = torch.randn(100, 4)
        batch = torch.tensor([0]*25 + [1]*25)
        
        predictions, intermediates = self.model(x, edge_index, edge_attr, batch)
        
        self.assertEqual(predictions.shape[0], batch_size)
        self.assertEqual(predictions.shape[1], self.hybrid_config.output_dim)

class TestMolecularProcessing(unittest.TestCase):
    """Test molecular processing utilities"""
    
    def test_smiles_to_mol(self):
        """Test SMILES to molecule conversion"""
        smiles = "CCO"  # Ethanol
        mol = smiles_to_mol(smiles)
        
        self.assertIsNotNone(mol)
        assert mol is not None, "Molecule conversion failed"
        self.assertGreater(mol.GetNumAtoms(), 0)
    
    def test_mol_to_graph(self):
        """Test molecule to graph conversion"""
        smiles = "CCO"
        mol = smiles_to_mol(smiles)
        
        graph_data = mol_to_graph_data(mol)
        
        self.assertIn('node_features', graph_data)
        self.assertIn('edge_index', graph_data)
        self.assertGreater(graph_data['n_nodes'], 0)
    
    def test_pad_graph(self):
        """Test graph padding"""
        smiles = "CCO"
        mol = smiles_to_mol(smiles)
        
        graph_data = mol_to_graph_data(mol)
        padded = pad_graph_data(graph_data, max_nodes=50)
        
        self.assertEqual(padded['node_features'].shape[0], 50)

class TestClassicalBaseline(unittest.TestCase):
    """Test classical baseline model"""
    
    def setUp(self):
        self.config = get_experiment_config('gnn_baseline')
        
        # Validate and extract config fields as typed attributes
        assert self.config.gnn_config is not None, "GNNConfig cannot be None"
        assert self.config.hybrid_config is not None, "HybridModelConfig cannot be None"
        
        self.gnn_config = self.config.gnn_config
        self.hybrid_config = self.config.hybrid_config
        
        self.model = ClassicalGNNBaseline(
            self.gnn_config,
            self.hybrid_config
        )
    
    def test_forward_pass(self):
        """Test classical model forward pass"""
        batch_size = 2
        num_nodes = 50
        
        x = torch.randn(num_nodes, self.gnn_config.num_node_features)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        batch = torch.tensor([0]*25 + [1]*25)
        
        predictions, intermediates = self.model(x, edge_index, batch=batch)
        
        self.assertEqual(predictions.shape[0], batch_size)
        self.assertEqual(predictions.shape[1], self.hybrid_config.output_dim)

if __name__ == "__main__":
    unittest.main()
