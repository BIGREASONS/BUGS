import os
import json
import pandas as pd

class ComparisonPipeline:
    def __init__(self, results_dir: str = 'outputs/results'):
        self.results_dir = results_dir
        
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
        print("Generating Performance Tables...")
        df = self.load_metrics()
        
        if df.empty:
            print("No results found. Run train.py and train_graph.py first.")
            return
            
        print("\n=== Performance Table ===")
        # Reorder columns
        columns = ['Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 'F1 (Weighted)', 'MCC']
        df = df[columns]
        print(df.to_string(index=False))
        
        # Save table
        df.to_csv(os.path.join(self.results_dir, 'final_performance_table.csv'), index=False)
        print(f"\nSaved performance table to {os.path.join(self.results_dir, 'final_performance_table.csv')}")
        
    def generate_rankings(self, metric: str = 'F1 (Macro)'):
        print(f"\nGenerating Model Rankings based on {metric}...")
        df = self.load_metrics()
        if df.empty:
            return
            
        ranked_df = df.sort_values(by=metric, ascending=False)
        print(f"\n=== Rankings ({metric}) ===")
        for i, row in enumerate(ranked_df.itertuples()):
            print(f"{i+1}. {row.Model} - {getattr(row, metric.replace(' ', '_').replace('(', '').replace(')', '')):.4f}")
            
    def run_ablation_study(self):
        print("\nGenerating Ablation Study Results...")
        df = self.load_metrics()
        if df.empty:
            return
            
        print("\n=== Ablation Study ===")
        ablation_map = {
            'XGBoost': 'A: Metrics Only',
            'GraphCodeBERT': 'B: GraphCodeBERT Only',
            'FusionModel': 'C: GraphCodeBERT + Metrics',
            'GraphTransformer': 'D: GraphCodeBERT + Metrics + Graph Transformer'
        }
        
        # Map existing models if their names match
        if 'Model' in df.columns:
            df['Ablation Phase'] = df['Model'].map(ablation_map)
            df = df.dropna(subset=['Ablation Phase'])
            df = df[['Ablation Phase', 'F1 (Macro)', 'MCC']].sort_values(by='Ablation Phase')
            print(df.to_string(index=False))
        
if __name__ == "__main__":
    pipeline = ComparisonPipeline()
    pipeline.generate_performance_tables()
    pipeline.generate_rankings()
    pipeline.run_ablation_study()
