import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configs.config import Config

class ComparisonPipeline:
    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir if results_dir else Config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_metrics(self) -> pd.DataFrame:
        metrics_list = []
        for file in os.listdir(self.results_dir):
            if file.endswith('_metrics.json'):
                path = os.path.join(self.results_dir, file)
                with open(path, 'r') as f:
                    data = json.load(f)
                    metrics_list.append(data)
                    
        df = pd.DataFrame(metrics_list)
        return df

    def generate_performance_tables(self):
        print("Generating Publication-Quality Performance Tables...")
        df = self.load_metrics()
        
        if df.empty:
            print("No results found. Run training scripts first.")
            return
            
        columns = ['Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 'MCC']
        # Filter columns that exist
        columns = [c for c in columns if c in df.columns]
        df_table = df[columns].copy()
        
        # Round to 4 decimal places
        for col in columns[1:]:
            df_table[col] = df_table[col].apply(lambda x: f"{x:.4f}")
            
        # Markdown Table
        md_caption = "Table 1. Comparative performance of all evaluated models on the generated dataset. The highest metric per category represents the optimal architecture for the severity classification task."
        md_table = df_table.to_markdown(index=False)
        with open(os.path.join(self.results_dir, 'performance_table.md'), 'w') as f:
            f.write(f"{md_table}\n\n{md_caption}")
            
        # LaTeX Table
        latex_table = df_table.to_latex(index=False, float_format="%.4f", caption="Comparative performance of all evaluated models.", label="tab:performance")
        with open(os.path.join(self.results_dir, 'performance_table.tex'), 'w') as f:
            f.write(latex_table)
            
        print("Markdown and LaTeX tables saved to results directory.")

    def plot_comparison_chart(self):
        df = self.load_metrics()
        if df.empty or 'F1 (Macro)' not in df.columns:
            return
            
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='Model', y='F1 (Macro)', palette='Blues_d')
        plt.title('Model Comparison: Macro F1 Score')
        plt.ylabel('Macro F1')
        plt.xlabel('Architecture')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'comparison_bar_chart.png'), dpi=300)
        plt.savefig(os.path.join(self.results_dir, 'comparison_bar_chart.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.results_dir, 'comparison_bar_chart.svg'), format='svg')
        plt.close()
        print("Comparison charts generated.")

    def run_ablation_study(self):
        df = self.load_metrics()
        if df.empty:
            return
            
        ablation_map = {
            'XGBoost': 'Metrics Only',
            'GraphCodeBERT': 'GCB Only',
            'FusionModel': '+ Metrics Fusion',
            'GraphTransformer': '+ Graph Transformer'
        }
        
        if 'Model' in df.columns:
            df['Ablation Phase'] = df['Model'].map(ablation_map)
            df = df.dropna(subset=['Ablation Phase'])
            if df.empty:
                return
                
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=df, x='Ablation Phase', y='F1 (Macro)', marker='o', linewidth=2)
            plt.title('Ablation Study: Architectural Progression')
            plt.ylabel('Macro F1 Score')
            plt.xlabel('Phase')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.results_dir, 'ablation_study.png'), dpi=300)
            plt.savefig(os.path.join(self.results_dir, 'ablation_study.pdf'), bbox_inches='tight')
            plt.close()
            print("Ablation charts generated.")
        
if __name__ == "__main__":
    pipeline = ComparisonPipeline()
    pipeline.generate_performance_tables()
    pipeline.plot_comparison_chart()
    pipeline.run_ablation_study()
