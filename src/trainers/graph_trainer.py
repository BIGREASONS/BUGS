import os
import csv
import json
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import numpy as np
from typing import Dict, Any, Tuple, Optional
from torch_geometric.data import Data

from src.utils.losses import FocalLoss
from configs.config import Config


class BSPTEGTTrainer:
    """Training loop for the BSPTEGT Graph Transformer.

    Features:
        - AdamW with weight-decay groups
        - Linear LR schedule with warmup
        - Mixed-precision when CUDA available
        - Focal Loss / Weighted CE
        - Early stopping on validation Macro F1
        - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_graph: Data,
        device: torch.device,
        epochs: int = 20,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = 'outputs/checkpoints/graph',
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model                   = model.to(device)
        self.train_graph             = train_graph.to(device)
        self.device                  = device
        self.epochs                  = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir          = checkpoint_dir
        self.use_amp                 = device.type == 'cuda'

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._save_config()

        # Loss
        if Config.LOSS_TYPE == 'focal':
            alpha = (
                class_weights.to(device)
                if (Config.FOCAL_ALPHA == 'class_weights' and class_weights is not None)
                else None
            )
            self.criterion = FocalLoss(gamma=Config.FOCAL_GAMMA, alpha=alpha)
            print(f"Graph Loss: FocalLoss(gamma={Config.FOCAL_GAMMA})")
        elif Config.LOSS_TYPE == 'weighted_ce' and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print("Graph Loss: Weighted CrossEntropyLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Graph Loss: Standard CrossEntropyLoss")

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(self.epochs * 0.1)),
            num_training_steps=self.epochs,
        )

        # Mixed precision
        self.amp_scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # CSV log
        self._log_path = os.path.join(self.checkpoint_dir, 'training_log.csv')
        with open(self._log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_f1_macro'])

    def _save_config(self) -> None:
        config_dict = {
            k: v for k, v in Config.__dict__.items()
            if not k.startswith('__') and not callable(v)
        }
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

    def train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                logits = self.model(self.train_graph.x, self.train_graph.edge_index)
                loss   = self.criterion(logits, self.train_graph.y)
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            logits = self.model(self.train_graph.x, self.train_graph.edge_index)
            loss   = self.criterion(logits, self.train_graph.y)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        return loss.item()

    def evaluate(
        self, eval_graph: Data, num_train_nodes: int = 0,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluates on eval_graph, slicing out training nodes if attached."""
        self.model.eval()
        eval_g = eval_graph.to(self.device)

        with torch.no_grad():
            logits = self.model(eval_g.x, eval_g.edge_index)

            eval_logits = logits[num_train_nodes:]
            eval_labels = eval_g.y[num_train_nodes:]
            loss = self.criterion(eval_logits, eval_labels)

        y_true = eval_labels.cpu().numpy()
        y_pred = torch.argmax(eval_logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        metrics = {
            'loss':                  loss.item(),
            'Accuracy':              acc,
            'Precision (Macro)':     p_mac,
            'Recall (Macro)':        r_mac,
            'F1 (Macro)':            f1_mac,
            'Precision (Weighted)':  p_wt,
            'Recall (Weighted)':     r_wt,
            'F1 (Weighted)':         f1_wt,
            'MCC':                   mcc,
        }
        return metrics, y_true, y_pred

    def train(
        self, valid_graph: Data, num_train_nodes: int = 0,
    ) -> Dict[str, Any]:
        best_f1          = 0.0
        patience_counter = 0
        best_metrics     = {}
        ckpt_path = os.path.join(self.checkpoint_dir, 'best_graph_model.pt')

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics, _, _ = self.evaluate(valid_graph, num_train_nodes)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Graph Epoch {epoch:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_f1_macro={val_metrics['F1 (Macro)']:.4f}"
                )

            with open(self._log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, train_loss, val_metrics['loss'], val_metrics['F1 (Macro)'],
                ])

            if val_metrics['F1 (Macro)'] > best_f1:
                best_f1          = val_metrics['F1 (Macro)']
                best_metrics     = val_metrics
                patience_counter = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"  Graph early stopping at epoch {epoch} (best val F1={best_f1:.4f})")
                break

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        print(f"\nBest graph model val Macro F1: {best_f1:.4f}")
        return best_metrics
