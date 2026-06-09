import json
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from typing import Dict, List, Tuple

def load_metrics_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads only the complexity metrics and labels for the XGBoost baseline.
    """
    metric_keys = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
    X, y = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                metrics = [float(item.get(k, 0.0)) for k in metric_keys]
                label = int(item.get('label', 0))
                X.append(metrics)
                y.append(label)
                
    return np.array(X), np.array(y)

class XGBoostBaseline:
    def __init__(self, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            eval_metric='mlogloss',
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.metric_keys = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None):
        eval_set = [(X_train, y_train)]
        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        
        # Calculate precision, recall, f1 for macro and weighted averages
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        mcc = matthews_corrcoef(y_test, preds)
        
        return {
            'Accuracy': acc,
            'Precision (Macro)': p_macro,
            'Recall (Macro)': r_macro,
            'F1 (Macro)': f1_macro,
            'Precision (Weighted)': p_weighted,
            'Recall (Weighted)': r_weighted,
            'F1 (Weighted)': f1_weighted,
            'MCC': mcc
        }

    def get_feature_importance(self) -> Dict[str, float]:
        importance = self.model.feature_importances_
        return {self.metric_keys[i]: float(importance[i]) for i in range(len(self.metric_keys))}

from sklearn.preprocessing import StandardScaler

def run_xgboost_baseline(train_path: str, valid_path: str, test_path: str):
    print("Loading data for XGBoost...")
    X_train, y_train = load_metrics_data(train_path)
    X_valid, y_valid = load_metrics_data(valid_path)
    X_test, y_test = load_metrics_data(test_path)
    
    print("Applying StandardScaler to complexity metrics...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    print("Training XGBoost baseline...")
    xgb_baseline = XGBoostBaseline()
    xgb_baseline.train(X_train, y_train, X_valid, y_valid)
    
    print("Evaluating XGBoost baseline on test set...")
    metrics = xgb_baseline.evaluate(X_test, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nFeature Importances:")
    importances = xgb_baseline.get_feature_importance()
    for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")
        
    return metrics, importances
