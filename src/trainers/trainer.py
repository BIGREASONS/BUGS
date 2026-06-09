import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import numpy as np
from typing import Dict, Any

class Trainer:
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
        checkpoint_dir: str = 'outputs/checkpoints',
        class_weights: torch.Tensor = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup Loss
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Setup Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Setup Scheduler
        total_steps = len(train_loader) * epochs // accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # Setup Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            metrics_tensor = batch['metrics_tensor'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with torch.amp.autocast('cuda'):
                logits = self.model(input_ids, attention_mask, metrics_tensor)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
            total_loss += loss.item() * self.accumulation_steps
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metrics_tensor = batch['metrics_tensor'].to(self.device)
                labels = batch['label'].to(self.device)
                
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids, attention_mask, metrics_tensor)
                    loss = self.criterion(logits, labels)
                    
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(loader)
        
        acc = accuracy_score(all_labels, all_preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'Accuracy': acc,
            'Precision (Macro)': p_macro,
            'Recall (Macro)': r_macro,
            'F1 (Macro)': f1_macro,
            'Precision (Weighted)': p_weighted,
            'Recall (Weighted)': r_weighted,
            'F1 (Weighted)': f1_weighted,
            'MCC': mcc
        }

    def train(self) -> Dict[str, Any]:
        best_f1 = 0.0
        patience_counter = 0
        best_metrics = {}
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.valid_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {val_metrics['loss']:.4f} | Valid Macro F1: {val_metrics['F1 (Macro)']:.4f}")
            
            # Early Stopping based on Macro F1
            if val_metrics['F1 (Macro)'] > best_f1:
                best_f1 = val_metrics['F1 (Macro)']
                best_metrics = val_metrics
                patience_counter = 0
                
                # Save checkpoint
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pt'))
                print("Checkpoint saved!")
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered!")
                break
                
        # Load best model before returning
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'best_model.pt')))
        return best_metrics
