"""
Graph Neural Network (GNN) Encoder for molecular feature extraction
Uses PyTorch Geometric for graph convolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from typing import Optional, Tuple
from src.config import GNNConfig

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder that processes molecular graphs
    and extracts structural features
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.num_node_features
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        
        # Graph convolution layers
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convolutions.append(GCNConv(self.input_dim, self.hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convolutions.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Output layer (for graph-level representation)
        self.convolutions.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Attention layer (optional)
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=self.dropout_rate,
                batch_first=True
            )
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch indices for graphs in mini-batch
            
        Returns:
            node_embeddings: Embeddings for each node [num_nodes, hidden_dim]
            graph_embedding: Graph-level embedding [batch_size, hidden_dim]
        """
        node_embeddings = x
        
        # Process through graph convolution layers
        for i, (conv, batch_norm) in enumerate(zip(self.convolutions[:-1], self.batch_norms[:-1])):
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = batch_norm(node_embeddings)
            node_embeddings = F.relu(node_embeddings)
            node_embeddings = self.dropout(node_embeddings)
        
        # Final convolution layer
        final_embeddings = self.convolutions[-1](node_embeddings, edge_index)
        final_embeddings = self.batch_norms[-1](final_embeddings)
        final_embeddings = F.relu(final_embeddings)
        
        # Optional attention mechanism
        if self.use_attention:
            # Reshape for multi-head attention
            if batch is not None:
                # Group nodes by graph
                graph_embeddings_list = []
                num_graphs = int(batch.max().item()) + 1
                for i in range(num_graphs):
                    mask = batch == i
                    graph_nodes = final_embeddings[mask]  # [num_nodes_in_graph, hidden_dim]
                    
                    if graph_nodes.shape[0] > 0:
                        # Self-attention within graph
                        attn_out, _ = self.attention(
                            graph_nodes.unsqueeze(0),  # Add batch dimension
                            graph_nodes.unsqueeze(0),
                            graph_nodes.unsqueeze(0)
                        )
                        graph_embeddings_list.append(attn_out.squeeze(0))
                
                if graph_embeddings_list:
                    final_embeddings = torch.cat(graph_embeddings_list, dim=0)
        
        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(final_embeddings, batch)
        else:
            graph_embedding = global_mean_pool(final_embeddings, torch.zeros(final_embeddings.shape[0], dtype=torch.long, device=final_embeddings.device))
        
        return final_embeddings, graph_embedding

class AttentionGNNEncoder(nn.Module):
    """
    Graph Attention Network (GAT) based encoder
    Uses attention mechanism for node aggregation
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.num_node_features
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList()
        
        # First layer
        self.attention_layers.append(
            GATConv(self.input_dim, self.hidden_dim, heads=4, concat=True, dropout=self.dropout_rate)
        )
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.attention_layers.append(
                GATConv(self.hidden_dim * 4, self.hidden_dim, heads=4, concat=True, dropout=self.dropout_rate)
            )
        
        # Output layer
        self.attention_layers.append(
            GATConv(self.hidden_dim * 4, self.hidden_dim, heads=1, concat=False, dropout=self.dropout_rate)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with graph attention
        """
        node_embeddings = x
        
        for i, gat_layer in enumerate(self.attention_layers):
            node_embeddings = gat_layer(node_embeddings, edge_index)
            if i < len(self.attention_layers) - 1:
                node_embeddings = F.elu(node_embeddings)
                node_embeddings = self.dropout(node_embeddings)
        
        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = global_mean_pool(
                node_embeddings,
                torch.zeros(node_embeddings.shape[0], dtype=torch.long, device=node_embeddings.device)
            )
        
        return node_embeddings, graph_embedding

class MultiGNNEncoder(nn.Module):
    """
    Ensemble of different GNN architectures
    Combines predictions from multiple GNN types
    """
    
    def __init__(self, config: GNNConfig, ensemble_size: int = 2):
        super().__init__()
        self.config = config
        self.ensemble_size = ensemble_size
        
        self.gcn_encoder = GNNEncoder(config)
        self.gat_encoder = AttentionGNNEncoder(config)
        
        # Combination layer
        self.combine = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining multiple encoders
        """
        _, gcn_embedding = self.gcn_encoder(x, edge_index, edge_attr, batch)
        _, gat_embedding = self.gat_encoder(x, edge_index, edge_attr, batch)
        
        # Combine embeddings
        combined = torch.cat([gcn_embedding, gat_embedding], dim=1)
        combined = self.combine(combined)
        combined = F.relu(combined)
        
        return _, combined
