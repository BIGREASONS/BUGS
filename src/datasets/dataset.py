import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class BugSeverityDataset(Dataset):
    """JSONL dataset for bug-severity prediction.

    Each record must contain:
        - 'code_no_comment' : source code string
        - 'label'           : integer severity class in [0, num_classes)
        - metric fields     : float values for each key in metric_keys
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        metric_keys: Optional[List[str]] = None,
        num_classes: int = 4,
    ):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.num_classes = num_classes
        self.metric_keys = metric_keys or [
            'lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e'
        ]

        self.data = self._load_data(file_path)
        self._validate_labels()

        # Build metric matrix
        all_metrics = [
            [float(item.get(k, 0.0)) for k in self.metric_keys]
            for item in self.data
        ]
        self.all_metrics = np.array(all_metrics, dtype=np.float32)

        # Scale metrics
        self.scaler = scaler
        if self.scaler is not None:
            if fit_scaler:
                self.all_metrics = self.scaler.fit_transform(self.all_metrics)
            else:
                if not hasattr(self.scaler, 'mean_'):
                    raise RuntimeError(
                        "FATAL: Scaler applied to validation/test split "
                        "before being fit on the training split."
                    )
                self.all_metrics = self.scaler.transform(self.all_metrics)

    def _load_data(self, file_path: str) -> List[Dict]:
        data: List[Dict] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at {file_path}:{lineno}: {exc}"
                    ) from exc
        return data

    def _validate_labels(self) -> None:
        for idx, item in enumerate(self.data):
            if 'label' not in item:
                raise ValueError(
                    f"Record {idx} is missing the 'label' field."
                )
            try:
                label_int = int(item['label'])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Record {idx} has non-integer label '{item['label']}'."
                ) from exc
            if not (0 <= label_int < self.num_classes):
                raise ValueError(
                    f"Record {idx} has label {label_int} outside "
                    f"[0, {self.num_classes - 1}]."
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        code_text      = item.get('code_no_comment', '')
        metrics_tensor = torch.tensor(self.all_metrics[idx], dtype=torch.float)
        label_tensor   = torch.tensor(int(item['label']), dtype=torch.long)

        encoding = self.tokenizer(
            code_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'metrics_tensor': metrics_tensor,
            'label':          label_tensor,
        }


def _worker_init_fn(worker_id: int) -> None:
    """Ensures each DataLoader worker has a unique but deterministic seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def create_dataloaders(
    train_path: str,
    valid_path: str,
    test_path: str,
    tokenizer_name: str = 'microsoft/graphcodebert-base',
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    seed: int = 42,
    metric_keys: Optional[List[str]] = None,
    num_classes: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates reproducible train / validation / test DataLoaders.

    Scaler is fit exclusively on the training split.
    """
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    scaler    = StandardScaler()

    train_dataset = BugSeverityDataset(
        train_path, tokenizer, max_length,
        scaler=scaler, fit_scaler=True,
        metric_keys=metric_keys, num_classes=num_classes,
    )
    valid_dataset = BugSeverityDataset(
        valid_path, tokenizer, max_length,
        scaler=scaler, fit_scaler=False,
        metric_keys=metric_keys, num_classes=num_classes,
    )
    test_dataset = BugSeverityDataset(
        test_path, tokenizer, max_length,
        scaler=scaler, fit_scaler=False,
        metric_keys=metric_keys, num_classes=num_classes,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, valid_loader, test_loader
