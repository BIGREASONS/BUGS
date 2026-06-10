import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, Optional, List

from configs.config import Config


class Evaluator:
    """Computes, plots, and persists evaluation artifacts.

    Outputs per prefix:
        {prefix}_metrics.json               scalar metrics
        {prefix}_metrics.csv                 same, one-row CSV
        {prefix}_predictions.csv             (True_Label, Predicted_Label)
        {prefix}_cm.png / .pdf               confusion-matrix heatmap
        {prefix}_classification_report.txt   sklearn per-class report
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.output_dir  = output_dir or Config.RESULTS_DIR
        self.class_names = class_names or Config.CLASS_NAMES
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        return {
            'Model':                model_name,
            'Accuracy':             float(acc),
            'Precision (Macro)':    float(p_mac),
            'Recall (Macro)':       float(r_mac),
            'F1 (Macro)':           float(f1_mac),
            'Precision (Weighted)': float(p_wt),
            'Recall (Weighted)':    float(r_wt),
            'F1 (Weighted)':        float(f1_wt),
            'MCC':                  float(mcc),
        }

    def generate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        self, conf_matrix: np.ndarray, prefix: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=self.class_names,
            yticklabels=self.class_names, ax=ax,
        )
        ax.set_title(f'Confusion Matrix: {prefix}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(
                os.path.join(self.output_dir, f'{prefix}_cm.{ext}'),
                dpi=300, bbox_inches='tight',
            )
        plt.close(fig)

    def save_results(
        self,
        metrics: Dict[str, Any],
        conf_matrix: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = 'eval',
    ) -> None:
        with open(os.path.join(self.output_dir, f'{prefix}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=str)

        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, f'{prefix}_metrics.csv'), index=False
        )

        self.plot_confusion_matrix(conf_matrix, prefix)

        pd.DataFrame({
            'True_Label':      y_true,
            'Predicted_Label': y_pred,
        }).to_csv(
            os.path.join(self.output_dir, f'{prefix}_predictions.csv'),
            index=False,
        )

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
        )
        with open(
            os.path.join(self.output_dir, f'{prefix}_classification_report.txt'), 'w'
        ) as f:
            f.write(f"Model: {prefix}\n\n{report}")

        print(f"Results saved -> {self.output_dir} (prefix='{prefix}')")
