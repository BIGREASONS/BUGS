import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from typing import Dict, Any, List

class Evaluator:
    def __init__(self, output_dir: str = 'outputs/results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculates all required metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        metrics = {
            'Model': model_name,
            'Accuracy': acc,
            'Precision (Macro)': p_macro,
            'Recall (Macro)': r_macro,
            'F1 (Macro)': f1_macro,
            'Precision (Weighted)': p_weighted,
            'Recall (Weighted)': r_weighted,
            'F1 (Weighted)': f1_weighted,
            'MCC': mcc
        }
        return metrics

    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)
        
    def save_results(self, metrics: Dict[str, Any], conf_matrix: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = 'eval'):
        """
        Exports CSV Results, JSON Results, and Predictions.
        """
        # Save JSON Metrics
        json_path = os.path.join(self.output_dir, f'{prefix}_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save CSV Metrics
        csv_path = os.path.join(self.output_dir, f'{prefix}_metrics.csv')
        df = pd.DataFrame([metrics])
        df.to_csv(csv_path, index=False)
        
        # Save Confusion Matrix
        cm_path = os.path.join(self.output_dir, f'{prefix}_confusion_matrix.csv')
        pd.DataFrame(conf_matrix).to_csv(cm_path, index=False)
        
        # Save Predictions
        preds_path = os.path.join(self.output_dir, f'{prefix}_predictions.csv')
        pd.DataFrame({'True_Label': y_true, 'Predicted_Label': y_pred}).to_csv(preds_path, index=False)
        
        print(f"Results saved to {self.output_dir} with prefix '{prefix}'")
