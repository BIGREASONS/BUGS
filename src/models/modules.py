import torch.nn as nn

class MetricEncoder(nn.Module):
    """
    Unified architectural block to process structural complexity metrics.
    Ensures feature consistency across Fusion Models and Graph Models.
    Architecture: 10 -> 32 -> BatchNorm -> ReLU -> Dropout -> 64 -> ReLU
    """
    def __init__(self, in_features: int = 10, hidden_dim: int = 32, out_features: int = 64, dropout: float = 0.2):
        super(MetricEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
