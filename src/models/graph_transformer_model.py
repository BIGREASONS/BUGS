import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class BSPGraphTransformer(nn.Module):
    def __init__(self, in_channels: int = 768 + 10, hidden_dim: int = 256, heads: int = 8, num_classes: int = 4, dropout: float = 0.2):
        """
        BSPTEGT-Inspired Graph Transformer Adaptation.
        Input: Node Features (GraphCodeBERT CLS embeddings [768] + 10 complexity metrics [10]) = 778
        Architecture:
        Node Features -> Graph Transformer Layer -> Graph Transformer Layer -> Linear Layer -> 4 Classes
        """
        super(BSPGraphTransformer, self).__init__()
        
        # Graph Transformer Layers
        # TransformerConv expects in_channels, out_channels (per head), heads
        # out_channels * heads should equal hidden_dim to maintain dimension, or we can use concat=False
        
        # Layer 1
        self.conv1 = TransformerConv(
            in_channels=in_channels, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        # Layer 2
        self.conv2 = TransformerConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        # Classification Head (per node classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for per-node classification in a Graph.
        Args:
            x: Node feature matrix of shape (num_nodes, in_channels)
            edge_index: Graph connectivity matrix of shape (2, num_edges)
        Returns:
            logits: Tensor of shape (num_nodes, num_classes)
        """
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)
        
        # Predict class for each node directly
        logits = self.classifier(x)
        
        return logits
