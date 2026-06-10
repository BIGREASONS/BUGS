import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Optional, Dict


class GraphCodeBERTEmbedder(nn.Module):
    """GraphCodeBERT embedding extractor for BSPTEGT.

    Stage 1 (fine-tuning):
        GraphCodeBERT -> [CLS] [B, 768] -> Dropout -> Linear(768 -> num_classes)
        Trained with classification loss to learn task-specific representations.

    Stage 2 (extraction):
        GraphCodeBERT (frozen) -> [CLS] [B, 768]
        Raw embeddings are used as node features for the Graph Transformer.
    """

    def __init__(
        self,
        model_name: str = 'microsoft/graphcodebert-base',
        num_classes: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.encoder    = RobertaModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metrics_tensor: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids        : [B, seq_len]
            attention_mask   : [B, seq_len]
            metrics_tensor   : ignored (API compatibility)
            return_embeddings: if True, returns raw [CLS] embeddings [B, 768]

        Returns:
            logits [B, num_classes]  or  embeddings [B, 768]
        """
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        if return_embeddings:
            return cls_output
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> nn.Module:
    """Loads a checkpoint, transparently handling DataParallel 'module.' prefix."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    # Strip 'module.' prefix if saved under DataParallel
    cleaned = {
        k.replace('module.', '', 1): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned)
    return model
