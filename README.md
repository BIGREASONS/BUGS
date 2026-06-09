# Bug Severity Prediction System

A research-grade bug severity prediction system predicting 4 severity classes (Critical, Major, Medium, Minor) using complexity metrics and abstract syntax representations from GraphCodeBERT.

## Repository Structure

```text
project/
├── configs/            # Configuration files
├── data/               # Place train.jsonl, valid.jsonl, test.jsonl here
├── notebooks/          # Exploratory Jupyter notebooks
├── outputs/            # Checkpoints, results, and generated graphs
├── src/
│   ├── baselines/      # XGBoost baseline implementation
│   ├── datasets/       # PyTorch Dataset and DataLoader wrappers
│   ├── evaluation/     # Metrics calculation (F1, MCC, Accuracy, etc.)
│   ├── graphs/         # Cosine-similarity based graph construction
│   ├── models/         # GraphCodeBERT, Fusion Model, Graph Transformer
│   ├── trainers/       # Mixed precision training loop with early stopping
│   └── utils/          # Helper functions
├── comparison.py       # Ablation studies and result generation
├── requirements.txt    # Dependencies
├── train.py            # Primary training script (Baselines & Fusion)
└── train_graph.py      # Graph construction and Transformer training
```

## Setup & Local Verification

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your data files in the `data/` directory:
- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`

3. Verify the training script runs:
```bash
python train.py --help
```

## Kaggle Training Workflow (Recommended)

To leverage 2 Tesla T4 GPUs effectively, use the GitHub to Kaggle workflow.

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial research implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/BugSeverityPrediction.git
git push -u origin main
```

2. **Clone on Kaggle**:
In a new Kaggle Notebook (with GPU & Internet Enabled):
```bash
!git clone https://github.com/YOUR_USERNAME/BugSeverityPrediction.git
%cd BugSeverityPrediction
!pip install -r requirements.txt
```

3. **Mount Data**:
Upload your JSONL files as a Kaggle Dataset and mount it. Update `configs/config.py` paths or create a symlink:
```bash
!ln -s /kaggle/input/your-dataset-name/ data
```

## Experiment Phases

### Phase 1: Baselines & Fusion
Train the sequential and fusion models.
```bash
# XGBoost
python train.py --model xgboost

# GraphCodeBERT
python train.py --model graphcodebert

# Fusion Model
python train.py --model fusion
```

### Phase 2: Graph Transformer
After validating Phase 1, extract embeddings, construct the similarity graph, and train the BSPGraphTransformer:
```bash
python train_graph.py
```

### Phase 3: Ablation Study
Compare all results:
```bash
python comparison.py
```
