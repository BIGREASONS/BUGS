import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from src.models.modules import MetricEncoder

class BSPGraphTransformer(nn.Module):
    def __init__(self, cls_dim: int = 768, num_classes: int = 4, hidden_dim: int = 256, heads: int = 8, dropout: float = 0.2):
        """
        BSPTEGT-Inspired Graph Transformer Adaptation.
        """
        super(BSPGraphTransformer, self).__init__()
        
        self.metric_encoder = MetricEncoder(dropout=dropout)
        
        in_channels = cls_dim + 64 # 768 + 64 = 832
        
        self.conv1 = TransformerConv(
            in_channels=in_channels, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        self.conv2 = TransformerConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Split raw features
        cls_feat = x[:, :768]
        metrics_raw = x[:, 768:]
        
        # Encode metrics
        metrics_encoded = self.metric_encoder(metrics_raw)
        
        # Recombine
        x = torch.cat([cls_feat, metrics_encoded], dim=1)
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
