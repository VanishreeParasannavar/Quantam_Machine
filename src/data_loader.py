"""
Data loading and preprocessing for drug discovery datasets
Supports MoleculeNet benchmarks: ESOL, Tox21, HIV, BBBP
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Union
import os
import logging
from tqdm import tqdm
import pickle

try:
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
except ImportError:
    Data = None
    PyGDataLoader = None

from src.utils import (
    smiles_to_mol, mol_to_graph_data, pad_graph_data,
    normalize_features, normalize_targets, denormalize_targets
)

logger = logging.getLogger(__name__)

class MoleculeNetDataset(Dataset):
    """
    PyTorch Dataset wrapper for MoleculeNet benchmarks
    """
    
    def __init__(self, smiles_list: List[str], targets: np.ndarray,
                 max_nodes: int = 50, normalize: bool = True,
                 task_type: str = "regression"):
        """
        Initialize dataset
        
        Args:
            smiles_list: List of SMILES strings
            targets: Target values (toxicity, solubility, etc.)
            max_nodes: Maximum number of nodes in graphs
            normalize: Whether to normalize targets
            task_type: "regression" or "classification"
        """
        self.smiles_list = smiles_list
        self.max_nodes = max_nodes
        self.task_type = task_type
        
        # Process molecules
        self.graphs = []
        self.valid_indices = []
        
        logger.info(f"Processing {len(smiles_list)} molecules...")
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Preprocessing molecules")):
            mol = smiles_to_mol(smiles)
            if mol is not None:
                try:
                    graph_data = mol_to_graph_data(mol)
                    padded_graph = pad_graph_data(graph_data, max_nodes)
                    self.graphs.append(padded_graph)
                    self.valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to process SMILES {smiles}: {e}")
        
        # Filter targets to valid molecules only
        self.targets = targets[self.valid_indices]
        
        # Normalize targets
        if normalize and task_type == "regression":
            self.targets, self.target_min, self.target_scale = normalize_targets(self.targets.astype(np.float32))
        else:
            self.target_min = 0.0
            self.target_scale = 1.0
        
        logger.info(f"Successfully processed {len(self.graphs)} molecules")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get data sample
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with graph data and target
        """
        graph = self.graphs[idx]
        target = self.targets[idx]
        
        return {
            'node_features': torch.FloatTensor(graph['node_features']),
            'edge_index': torch.LongTensor(graph['edge_index']),
            'edge_features': torch.FloatTensor(graph['edge_features']),
            'mol_descriptors': torch.FloatTensor(graph['mol_descriptors']),
            'n_nodes': graph['n_nodes'],
            'n_edges': graph['n_edges'],
            'target': torch.FloatTensor([target]) if self.task_type == "regression" else torch.LongTensor([target])
        }

class MoleculeNetLoader:
    """
    Loader for MoleculeNet benchmark datasets
    Handles automatic download and preprocessing
    """
    
    DATASETS = {
        'ESOL': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/ESOL.csv',
            'task': 'regression',
            'target_col': 'measured log solubility in mols/L'
        },
        'Tox21': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
            'task': 'classification',
            'target_cols': 12  # Multi-task classification
        },
        'HIV': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
            'task': 'classification',
            'target_col': 'HIV_active'
        },
        'BBBP': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
            'task': 'classification',
            'target_col': 'p_np'
        }
    }
    
    def __init__(self, dataset_name: str = 'ESOL', path: str = './data', download: bool = True):
        """
        Initialize dataset loader
        
        Args:
            dataset_name: Name of dataset (ESOL, Tox21, HIV, BBBP)
            path: Path to cache datasets
            download: Whether to download datasets
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(self.DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.path = path
        self.download = download
        
        os.makedirs(path, exist_ok=True)
    
    def load_dataset(self, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                    max_nodes: int = 50) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and split dataset
        
        Args:
            split_ratio: (train, val, test) split ratios
            max_nodes: Maximum nodes in molecular graphs
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        cache_file = os.path.join(self.path, f'{self.dataset_name}_processed.pkl')
        
        # Try to load cached version
        if os.path.exists(cache_file):
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                smiles_list, targets = pickle.load(f)
        else:
            # Download and process
            logger.info(f"Downloading {self.dataset_name} dataset...")
            smiles_list, targets = self._download_and_process()
            
            # Cache processed data
            with open(cache_file, 'wb') as f:
                pickle.dump((smiles_list, targets), f)
        
        # Create dataset
        full_dataset = MoleculeNetDataset(
            smiles_list, targets,
            max_nodes=max_nodes,
            task_type=self.DATASETS[self.dataset_name]['task']
        )
        
        # Split dataset
        n_samples = len(full_dataset)
        train_size = int(n_samples * split_ratio[0])
        val_size = int(n_samples * split_ratio[1])
        test_size = n_samples - train_size - val_size
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        logger.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _download_and_process(self) -> Tuple[List[str], np.ndarray]:
        """Download and process dataset"""
        import pandas as pd
        
        dataset_info = self.DATASETS[self.dataset_name]
        csv_file = os.path.join(self.path, f'{self.dataset_name}.csv')
        
        # Download if not exists
        if not os.path.exists(csv_file):
            import urllib.request
            url = dataset_info['url']
            logger.info(f"Downloading from {url}")
            urllib.request.urlretrieve(url, csv_file)
        
        # Load and process
        df = pd.read_csv(csv_file)
        
        smiles_list = df['smiles'].tolist() if 'smiles' in df.columns else df['SMILES'].tolist()
        
        if self.dataset_name == 'Tox21':
            # Multi-task: use first task as example
            target_cols = [col for col in df.columns if col.startswith('Tox21')]
            targets = np.asarray(df[target_cols[0]].fillna(0).values.astype(np.int32))
        else:
            target_col = dataset_info['target_col']
            targets = np.asarray(df[target_col].fillna(0).values.astype(np.float32))
        
        return smiles_list, targets

def create_graph_batch_data(batch: List[dict], device: torch.device):
    """
    Create batched graph data for PyTorch Geometric style processing
    
    Args:
        batch: List of sample dictionaries
        device: Device to move data to
        
    Returns:
        Batched graph data
    """
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    targets_list = []
    batch_indices = []
    node_offset = 0
    
    for i, sample in enumerate(batch):
        n_nodes = sample['n_nodes']
        
        node_features_list.append(sample['node_features'][:n_nodes])
        targets_list.append(sample['target'])
        
        # Adjust edge indices for batch
        n_edges = sample['n_edges']
        if n_edges > 0:
            edge_index = sample['edge_index'][:2, :n_edges] + node_offset
            edge_index_list.append(edge_index)
            edge_features_list.append(sample['edge_features'][:n_edges])
        
        # Batch assignment
        batch_indices.extend([i] * n_nodes)
        node_offset += n_nodes
    
    # Concatenate batch
    node_features = torch.cat(node_features_list, dim=0).to(device)
    targets = torch.cat(targets_list, dim=0).to(device)
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
    
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1).to(device)
        edge_features = torch.cat(edge_features_list, dim=0).to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_features = torch.zeros((0, 4), dtype=torch.float32, device=device)  # Assuming 4-dim edge features
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'batch': batch_tensor,
        'targets': targets
    }

class DrugDiscoveryDataLoader:
    """
    Wrapper for creating data loaders for drug discovery tasks
    """
    
    def __init__(self, dataset_name: str = 'ESOL', batch_size: int = 32, 
                 num_workers: int = 0, cache_path: str = './data'):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load datasets
        loader = MoleculeNetLoader(dataset_name, cache_path)
        self.train_dataset, self.val_dataset, self.test_dataset = loader.load_dataset()
    
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda batch: create_graph_batch_data(batch, torch.device('cpu'))
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: create_graph_batch_data(batch, torch.device('cpu'))
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: create_graph_batch_data(batch, torch.device('cpu'))
        )
