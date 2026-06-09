import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from typing import Dict, Any, List
from configs.config import Config

class Evaluator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir if output_dir else Config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
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
        
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, prefix: str):
        """Plots a publication-quality confusion matrix."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Critical', 'Major', 'Medium', 'Minor'],
                    yticklabels=['Critical', 'Major', 'Medium', 'Minor'])
        plt.title(f'Confusion Matrix: {prefix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save high-res PNG and PDF
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_cm.png'), dpi=300)
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_cm.pdf'), bbox_inches='tight')
        plt.close()

    def save_results(self, metrics: Dict[str, Any], conf_matrix: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = 'eval'):
        # JSON
        with open(os.path.join(self.output_dir, f'{prefix}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # CSV
        pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, f'{prefix}_metrics.csv'), index=False)
        
        # Plot CM
        self.plot_confusion_matrix(conf_matrix, prefix)
        
        # Predictions
        pd.DataFrame({'True_Label': y_true, 'Predicted_Label': y_pred}).to_csv(os.path.join(self.output_dir, f'{prefix}_predictions.csv'), index=False)
        
        print(f"Results saved to {self.output_dir} with prefix '{prefix}'")
