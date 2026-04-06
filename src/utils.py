"""
Utility functions for Quantum Drug Discovery System
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from typing import Union, Tuple, Optional, List
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get device (GPU if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit molecule object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        return Chem.AddHs(mol)  # Add explicit hydrogens
    except Exception as e:
        logger.error(f"Error converting SMILES {smiles}: {e}")
        return None

def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """Convert molecule to SMILES string"""
    if canonical:
        return Chem.MolToSmiles(mol)
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_node_features(mol: Chem.Mol) -> np.ndarray:
    """Extract node (atom) features from molecule"""
    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
    atom_types = {sym: i for i, sym in enumerate(atom_symbols)}
    
    features = []
    for atom in mol.GetAtoms():
        feature = []
        
        # Atom type (one-hot encoded)
        atom_type = atom.GetSymbol()
        atom_type_feature = np.zeros(len(atom_symbols))
        if atom_type in atom_types:
            atom_type_feature[atom_types[atom_type]] = 1
        else:
            atom_type_feature[-1] = 1  # Unknown
        feature.extend(atom_type_feature)
        
        # Atom features
        feature.append(atom.GetTotalDegree())  # Degree
        feature.append(atom.GetFormalCharge())  # Charge
        feature.append(int(atom.GetIsAromatic()))  # Aromaticity
        feature.append(atom.GetHybridization())  # Hybridization
        feature.append(atom.GetTotalNumHs())  # Number of hydrogens
        feature.append(atom.GetConnectivity())  # Connectivity
        
        features.append(feature)
    
    return np.array(features, dtype=np.float32)

def get_edge_features(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """Extract edge indices and features from molecule"""
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    bond_type_dict = {bond: i for i, bond in enumerate(bond_types)}
    
    edges = []
    edge_features = []
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Add edge in both directions for undirected graph
        edges.append([begin_idx, end_idx])
        edges.append([end_idx, begin_idx])
        
        # Bond features
        bond_type = bond.GetBondType()
        feature = np.zeros(len(bond_types))
        if bond_type in bond_type_dict:
            feature[bond_type_dict[bond_type]] = 1
        else:
            feature[-1] = 1  # Unknown
        
        edge_features.append(feature)
        edge_features.append(feature)  # Same for reverse edge
    
    if len(edges) == 0:
        edge_index = np.array([], dtype=np.int64).reshape(2, 0)
        edge_attr = np.array([], dtype=np.float32).reshape(0, len(bond_types))
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.float32)
    
    return edge_index, edge_attr

def compute_molecular_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute molecular descriptors"""
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'N_rotatable': Descriptors.NumRotatableBonds(mol),
        'N_h_donors': Descriptors.NumHDonors(mol),
        'N_h_acceptors': Descriptors.NumHAcceptors(mol),
        'N_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'N_atoms': mol.GetNumAtoms(),
        'N_bonds': mol.GetNumBonds(),
    }
    
    return np.array(list(descriptors.values()), dtype=np.float32)

def normalize_features(features: np.ndarray, global_min: Optional[np.ndarray] = None,
                      global_max: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to [0, 1] range"""
    if global_min is None:
        global_min = np.min(features, axis=0)
    if global_max is None:
        global_max = np.max(features, axis=0)
    
    # Ensure numpy arrays
    global_min = np.asarray(global_min)
    global_max = np.asarray(global_max)
    
    # Avoid division by zero
    global_max = np.where(global_max == global_min, np.ones_like(global_max), global_max).astype(np.float32)
    
    normalized = (features - global_min) / (global_max - global_min)
    return np.asarray(normalized, dtype=np.float32), np.asarray(global_min, dtype=np.float32), np.asarray(global_max, dtype=np.float32)

def normalize_targets(targets: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize target values to [0, 1]"""
    min_val = np.min(targets)
    max_val = np.max(targets)
    
    if min_val == max_val:
        return targets, min_val, 1.0
    
    normalized = (targets - min_val) / (max_val - min_val)
    return normalized, min_val, max_val - min_val

def denormalize_targets(normalized_targets: np.ndarray, min_val: float, scale: float) -> np.ndarray:
    """Denormalize targets back to original scale"""
    return normalized_targets * scale + min_val

def mol_to_graph_data(mol: Chem.Mol, global_feature_stats: Optional[dict] = None) -> dict:
    """Convert molecule to graph data"""
    node_features = get_node_features(mol)
    edge_index, edge_features = get_edge_features(mol)
    mol_descriptors = compute_molecular_descriptors(mol)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'mol_descriptors': mol_descriptors,
        'n_nodes': mol.GetNumAtoms(),
        'n_edges': mol.GetNumBonds(),
    }

def pad_graph_data(graph_data: dict, max_nodes: int = 50, max_edges: int = 200) -> dict:
    """Pad graph data to fixed size"""
    n_nodes = graph_data['n_nodes']
    n_edges = graph_data['n_edges']
    
    node_feat_dim = graph_data['node_features'].shape[1]
    edge_feat_dim = graph_data['edge_features'].shape[1]
    
    # Pad node features
    padded_nodes = np.zeros((max_nodes, node_feat_dim), dtype=np.float32)
    padded_nodes[:n_nodes] = graph_data['node_features']
    
    # Pad edge features
    padded_edges = np.zeros((max_edges, edge_feat_dim), dtype=np.float32)
    if n_edges > 0:
        padded_edges[:n_edges] = graph_data['edge_features']
    
    # Pad edge index
    padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
    if n_edges > 0:
        padded_edge_index[:, :n_edges] = graph_data['edge_index']
    
    return {
        'node_features': padded_nodes,
        'edge_index': padded_edge_index,
        'edge_features': padded_edges,
        'mol_descriptors': graph_data['mol_descriptors'],
        'n_nodes': n_nodes,
        'n_edges': n_edges,
    }

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 15, delta: float = 0.0, best_score: Optional[float] = None):
        self.patience = patience
        self.delta = delta
        self.patience_counter = 0
        self.best_score = best_score
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop"""
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.patience_counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}. Best score at epoch {self.best_epoch}")
                return True
            return False
