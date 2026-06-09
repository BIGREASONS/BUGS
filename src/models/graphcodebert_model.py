import torch
import torch.nn as nn
from transformers import RobertaModel

class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, model_name: str = 'microsoft/graphcodebert-base', num_classes: int = 4, dropout_rate: float = 0.2):
        """
        GraphCodeBERT Baseline model for Bug Severity Prediction.
        Architecture: GraphCodeBERT -> CLS Token -> Dropout -> Linear Layer -> 4 Classes
        """
        super(GraphCodeBERTClassifier, self).__init__()
        
        # Load pre-trained GraphCodeBERT
        self.encoder = RobertaModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # GraphCodeBERT hidden size is 768
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, metrics_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            attention_mask: Tensor of shape (batch_size, sequence_length)
            metrics_tensor: Ignored in this baseline, kept for API compatibility.
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # GraphCodeBERT outputs: (last_hidden_state, pooler_output)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the CLS token representation (first token)
        # Using outputs.last_hidden_state[:, 0, :] is often preferred over pooler_output
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_cls_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the CLS embeddings without the classification head.
        Useful for graph construction.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
