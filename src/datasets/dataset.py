import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

class BugSeverityDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: RobertaTokenizer, max_length: int = 512, scaler: StandardScaler = None, fit_scaler: bool = False):
        """
        Args:
            file_path (str): Path to the JSONL data file.
            tokenizer (RobertaTokenizer): The tokenizer to process code.
            max_length (int): Maximum sequence length.
            scaler (StandardScaler): Optional scaler for normalizing metrics.
            fit_scaler (bool): Whether to fit the scaler on this dataset's metrics.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(file_path)

        # The 10 specific complexity metrics requested
        self.metric_keys = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
        
        # Extract and normalize metrics
        all_metrics = []
        for item in self.data:
            metrics = [float(item.get(k, 0.0)) for k in self.metric_keys]
            all_metrics.append(metrics)
            
        self.all_metrics = np.array(all_metrics)
        self.scaler = scaler
        
        if self.scaler is not None:
            if fit_scaler:
                self.all_metrics = self.scaler.fit_transform(self.all_metrics)
            else:
                # Strict check to prevent leakage: Scaler MUST be fit on train data first
                assert hasattr(self.scaler, 'mean_'), "FATAL: Scaler applied to validation/test split before fitting on train!"
                self.all_metrics = self.scaler.transform(self.all_metrics)

    def _load_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Extract features
        code_text = item.get('code_no_comment', '')
        
        # Get normalized metrics
        metrics_tensor = torch.tensor(self.all_metrics[idx], dtype=torch.float)
        
        # Parse label
        label = int(item.get('label', 0))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Tokenize code
        encoding = self.tokenizer(
            code_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension added by return_tensors='pt'
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'metrics_tensor': metrics_tensor,
            'label': label_tensor
        }

def create_dataloaders(
    train_path: str,
    valid_path: str,
    test_path: str,
    tokenizer_name: str = 'microsoft/graphcodebert-base',
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns train, validation, and test dataloaders.
    """
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    scaler = StandardScaler()
    
    train_dataset = BugSeverityDataset(train_path, tokenizer, max_length, scaler=scaler, fit_scaler=True)
    valid_dataset = BugSeverityDataset(valid_path, tokenizer, max_length, scaler=scaler, fit_scaler=False)
    test_dataset = BugSeverityDataset(test_path, tokenizer, max_length, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader
