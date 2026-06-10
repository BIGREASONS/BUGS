import os
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.utils.losses import FocalLoss
from configs.config import Config


class EmbedderTrainer:
    """Trainer for Stage 1: GraphCodeBERT fine-tuning.

    Features:
        - AdamW with per-layer weight-decay groups
        - Linear LR schedule with 10% warmup
        - Mixed-precision (autocast + GradScaler) when CUDA available
        - Gradient accumulation
        - Macro-F1 early stopping
        - Best-checkpoint saving / reloading
        - Per-epoch CSV training log
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        accumulation_steps: int = 1,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = 'outputs/checkpoints/embedder',
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model                   = model.to(device)
        self.train_loader            = train_loader
        self.valid_loader            = valid_loader
        self.device                  = device
        self.epochs                  = epochs
        self.accumulation_steps      = max(1, accumulation_steps)
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
            print(f"Embedder Loss: FocalLoss(gamma={Config.FOCAL_GAMMA})")
        elif Config.LOSS_TYPE == 'weighted_ce' and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print("Embedder Loss: Weighted CrossEntropyLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Embedder Loss: Standard CrossEntropyLoss")

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

        # Scheduler
        total_steps = len(train_loader) * epochs // self.accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # Mixed precision
        self.amp_scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # CSV log
        self._log_path = os.path.join(self.checkpoint_dir, 'training_log.csv')
        with open(self._log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_f1_macro', 'val_mcc',
            ])

    def _save_config(self) -> None:
        config_dict = {
            k: v for k, v in Config.__dict__.items()
            if not k.startswith('__') and not callable(v)
        }
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            metrics_tensor = batch['metrics_tensor'].to(self.device)
            labels         = batch['label'].to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids, attention_mask, metrics_tensor)
                    loss   = self.criterion(logits, labels) / self.accumulation_steps
                self.amp_scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids, attention_mask, metrics_tensor)
                loss   = self.criterion(logits, labels) / self.accumulation_steps
                loss.backward()

            last_step = (step + 1) == len(self.train_loader)
            if (step + 1) % self.accumulation_steps == 0 or last_step:
                if self.use_amp:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(self.train_loader)

    def evaluate(self, loader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metrics_tensor = batch['metrics_tensor'].to(self.device)
                labels         = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask, metrics_tensor)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        metrics = {
            'loss': total_loss / len(loader),
            'Accuracy': acc,
            'F1 (Macro)': f1_mac,
            'MCC': mcc,
        }
        return metrics, y_true, y_pred

    def train(self) -> Dict[str, Any]:
        best_f1          = 0.0
        patience_counter = 0
        best_metrics     = {}
        ckpt_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

        for epoch in range(self.epochs):
            print(f"\nEmbedder Epoch {epoch + 1}/{self.epochs}")
            train_loss        = self.train_epoch()
            val_metrics, _, _ = self.evaluate(self.valid_loader)

            print(
                f"  train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1_macro={val_metrics['F1 (Macro)']:.4f}"
            )

            with open(self._log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch + 1, train_loss,
                    val_metrics['loss'], val_metrics['Accuracy'],
                    val_metrics['F1 (Macro)'], val_metrics['MCC'],
                ])

            if val_metrics['F1 (Macro)'] > best_f1:
                best_f1          = val_metrics['F1 (Macro)']
                best_metrics     = val_metrics
                patience_counter = 0
                # Save without DataParallel prefix
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), ckpt_path)
                print("  -> Checkpoint saved.")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{self.early_stopping_patience})")

            if patience_counter >= self.early_stopping_patience:
                print("  Early stopping triggered.")
                break

        # Reload best
        from src.models.embedding_extractor import load_checkpoint
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        load_checkpoint(base_model, ckpt_path, self.device)
        print(f"\nBest embedder val Macro F1: {best_f1:.4f}")
        return best_metrics
