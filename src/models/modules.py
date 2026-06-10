import torch.nn as nn


class MetricEncoder(nn.Module):
    """Shared block for processing structural complexity metrics.

    Architecture:
        Linear(in_features -> hidden_dim)
        -> LayerNorm(hidden_dim)
        -> ReLU
        -> Dropout
        -> Linear(hidden_dim -> out_features)
        -> ReLU

    Used in BSPTEGTGraphTransformer (node feature construction)
    and optionally in BSPTEGTGraphBuilder (similarity_plus_metrics edges).
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 32,
        out_features: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)
