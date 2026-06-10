import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from src.models.modules import MetricEncoder


class BSPTEGTGraphTransformer(nn.Module):
    """Graph Transformer for BSPTEGT bug severity prediction.

    Adaptation of the BSPTEGT architecture for source-code inputs.

    Node feature construction (inside forward):
        raw input [N, cls_dim + num_metrics]   (768 + 10 = 778)
        -> split: cls_feat [N, 768], metrics_raw [N, 10]
        -> MetricEncoder: metrics_raw -> [N, 64]
        -> recombine: cat -> [N, 832]

    Message-passing:
        TransformerConv(832 -> hidden_dim)     [N, 256]
        -> ReLU + Dropout
        TransformerConv(hidden_dim -> hidden_dim)  [N, 256]
        -> ReLU + Dropout

    Classifier:
        Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> num_classes)
    """

    def __init__(
        self,
        cls_dim: int = 768,
        num_classes: int = 4,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.2,
        num_metrics: int = 10,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        self._dropout_p   = dropout
        self._cls_dim     = cls_dim
        self._num_metrics = num_metrics

        self.metric_encoder = MetricEncoder(
            in_features=num_metrics, dropout=dropout
        )

        in_channels = cls_dim + 64  # 768 + 64 = 832

        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.conv2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x          : [N, cls_dim + num_metrics]
            edge_index : [2, E]
        Returns:
            logits     : [N, num_classes]
        """
        cls_feat    = x[:, :self._cls_dim]
        metrics_raw = x[:, self._cls_dim:]

        metrics_enc = self.metric_encoder(metrics_raw)
        x = torch.cat([cls_feat, metrics_enc], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout_p, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout_p, training=self.training)

        return self.classifier(x)
