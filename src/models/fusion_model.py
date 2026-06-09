import torch
import torch.nn as nn
from transformers import RobertaModel

class FusionModel(nn.Module):
    def __init__(self, model_name: str = 'microsoft/graphcodebert-base', num_classes: int = 4, dropout_rate: float = 0.2):
        """
        Fusion Model combining GraphCodeBERT and Complexity Metrics.
        Architecture:
        GraphCodeBERT -> CLS Token (768)
        Metrics Branch -> 10 -> 64
        Concatenate -> 832
        MLP -> 256
        Dropout
        Output -> 4 Classes
        """
        super(FusionModel, self).__init__()
        
        # Text Branch
        self.encoder = RobertaModel.from_pretrained(model_name)
        text_dim = self.encoder.config.hidden_size # 768
        
        # Metrics Branch
        self.metrics_branch = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        metrics_dim = 64
        
        fusion_dim = text_dim + metrics_dim # 832
        
        # Fusion MLP
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, metrics_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            attention_mask: Tensor of shape (batch_size, sequence_length)
            metrics_tensor: Tensor of shape (batch_size, 10)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Get text embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] # (batch_size, 768)
        
        # Get metrics embeddings
        metrics_output = self.metrics_branch(metrics_tensor) # (batch_size, 64)
        
        # Concatenate
        fused_features = torch.cat((cls_output, metrics_output), dim=1) # (batch_size, 832)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
