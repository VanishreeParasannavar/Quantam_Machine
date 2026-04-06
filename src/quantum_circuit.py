"""
Variational Quantum Circuit (VQC) for molecular feature encoding
Uses PennyLane for quantum circuit construction and execution
"""
import pennylane as qml
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import numpy as np
from src.config import VQCConfig

class QuantumCircuitLayer(nn.Module):
    """
    Parametrized Variational Quantum Circuit layer
    Encodes classical features into quantum states
    """
    
    def __init__(self, config: VQCConfig):
        super().__init__()
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.entangling_layers = config.entangling_layers
        self.measurement_qubits = config.measurement_qubits
        self.shots = config.shots
        
        # Initialize device
        if config.simulator == "default.qubit":
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            # Use Qiskit simulator
            try:
                from qiskit_aer import AerSimulator
                self.dev = qml.device("qiskit_aer", wires=self.n_qubits, shots=self.shots)
            except ImportError:
                print("Qiskit not available, using default.qubit simulator")
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum circuit parameters
        # Encoding parameters: one per qubit per encoding layer
        self.encoding_params = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits) * 0.1
        )
        
        # Variational parameters: weights for rotation gates
        self.variational_params = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3) * 0.1
        )
        
        # Entangling parameters
        self.entangling_params = nn.Parameter(
            torch.randn(self.n_layers, self.entangling_layers, self.n_qubits // 2) * 0.1
        )
        
    @classmethod
    def _create_circuit(cls, x: np.ndarray, encoding_params: np.ndarray,
                       variational_params: np.ndarray, entangling_params: np.ndarray,
                       n_qubits: int, n_layers: int, measurement_qubit: int = 0):
        """
        Define the parametrized quantum circuit
        Uses PyTorch Geometric features as input
        """
        # Normalize input features to [-π, π]
        x_normalized = np.tanh(x) * np.pi
        
        for layer in range(n_layers):
            # Feature encoding: angle encoding
            for i in range(min(len(x_normalized), n_qubits)):
                qml.RX(x_normalized[i], wires=i)
            
            # Add encoding parameters
            for i in range(n_qubits):
                qml.RZ(encoding_params[layer, i], wires=i)
            
            # Variational rotations
            for i in range(n_qubits):
                qml.RX(variational_params[layer, i, 0], wires=i)
                qml.RY(variational_params[layer, i, 1], wires=i)
                qml.RZ(variational_params[layer, i, 2], wires=i)
            
            # Entangling layer (CNOT ladder)
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            
            # Additional entangling with parameter
            for j in range(0, n_qubits - 1, 2):
                if j + 1 < n_qubits:
                    qml.CRZ(entangling_params[layer, j // 2, 0] if j // 2 < entangling_params.shape[1] else 0,
                            wires=[j, j + 1])
        
        return qml.expval(qml.PauliZ(measurement_qubit))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit
        
        Args:
            x: Feature tensor [batch_size, feature_dim] or [feature_dim]
            
        Returns:
            Measurement expectation values [batch_size, measurement_qubits]
        """
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        feature_dim = x.shape[1]
        
        # Pad or truncate features to n_qubits
        if feature_dim < self.n_qubits:
            x_padded = torch.cat([x, torch.zeros(batch_size, self.n_qubits - feature_dim, device=x.device)], dim=1)
        else:
            x_padded = x[:, :self.n_qubits]
        
        results = []
        
        # Define quantum function with QML
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(features, enc_params, var_params, ent_params):
            # Feature encoding
            for i in range(min(len(features), self.n_qubits)):
                qml.RX(features[i], wires=i)
            
            for layer in range(self.n_layers):
                # Parametrized rotations
                for i in range(self.n_qubits):
                    qml.RY(enc_params[layer, i], wires=i)
                    qml.RZ(var_params[layer, i, 0], wires=i)
                    qml.RX(var_params[layer, i, 1], wires=i)
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                for i in range(self.entangling_layers):
                    idx = i % (self.n_qubits - 1)
                    if idx + 1 < self.n_qubits:
                        qml.CRZ(ent_params[layer, i], wires=[idx, idx + 1])
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        # Execute circuit for each sample in batch
        for i in range(batch_size):
            output = quantum_circuit(
                x_padded[i],
                self.encoding_params,
                self.variational_params,
                self.entangling_params
            )
            results.append(output)
        
        result_tensor = torch.stack(results)
        
        if squeeze_output:
            result_tensor = result_tensor.squeeze(0)
        
        return result_tensor.unsqueeze(-1)

class QuantumFeatureMap(nn.Module):
    """
    Quantum feature map for encoding classical data into quantum states
    """
    
    def __init__(self, config: VQCConfig):
        super().__init__()
        self.config = config
        self.n_qubits = config.n_qubits
        
        if config.simulator == "default.qubit":
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            try:
                self.dev = qml.device("qiskit_aer", wires=self.n_qubits, shots=config.shots)
            except ImportError:
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical features using quantum feature map
        
        Args:
            x: Input features [batch_size, feature_dim]
            
        Returns:
            Quantum feature map output
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def feature_map_circuit(features):
            # Angle encoding
            for i in range(min(len(features), self.n_qubits)):
                qml.RY(features[i] * np.pi, wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Return multiple measurements for richer representation
            return [qml.expval(qml.PauliZ(i)) for i in range(min(self.n_qubits, 4))]
        
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            output = feature_map_circuit(x[i])
            if isinstance(output, (list, tuple)):
                output = torch.stack(output)
            else:
                output = output.unsqueeze(0)
            results.append(output)
        
        return torch.stack(results)

class ClassicalQuantumHybrid(nn.Module):
    """
    Hybrid classical-quantum layer
    Transfers classical features to quantum circuit and back
    """
    
    def __init__(self, input_dim: int, config: VQCConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.n_qubits = config.n_qubits
        
        # Classical preprocessing
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, config.n_qubits * 2),
            nn.ReLU(),
            nn.Linear(config.n_qubits * 2, config.n_qubits),
        )
        
        # Quantum layer
        self.quantum_layer = QuantumCircuitLayer(config)
        
        # Classical post-processing
        self.classical_decoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_qubits),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid layer
        """
        # Classical encoding
        encoded = self.classical_encoder(x)  # [batch, n_qubits]
        
        # Quantum processing
        quantum_out = self.quantum_layer(encoded)  # [batch, 1]
        
        # Classical decoding
        decoded = self.classical_decoder(quantum_out)  # [batch, n_qubits]
        
        return decoded
