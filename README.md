# Bug Severity Prediction via Hybrid Semantics and Graph Transformation

This repository contains the official implementation of our research on software bug severity prediction. We propose a hybrid architecture that leverages abstract syntax token sequences (GraphCodeBERT), structural software complexity metrics, and graph-based relational learning (Graph Transformer) to robustly classify bug severity into four discrete classes: Critical, Major, Medium, and Minor.

## Abstract
Accurate classification of software bug severity remains a critical challenge in software maintenance. Traditional approaches rely solely on historical text or basic metrics, often failing to capture deep contextual logic and structural degradation simultaneously. In this artifact, we present a multi-modal fusion architecture combined with a strictly inductive Graph Transformer. By mapping individual bugs as nodes in a similarity graph, we facilitate message passing between structurally related logic errors. Empirical evaluations demonstrate the architectural progression from XGBoost baselines to our proposed Graph Transformer.

## Dataset
The experiments are conducted on a unified dataset combining Defects4J and Bugs.jar. 
- **Splits:** `train.jsonl`, `valid.jsonl`, `test.jsonl`
- **Features:** Source code (tokenized to length 512) and 10 structural complexity metrics (McCabe, Halstead, etc.).
- **Imbalance:** Natural class imbalance is managed via Focal Loss ($\gamma=2.0$, $\alpha$ derived from inverse class frequencies).

## Methodology
We ablate the system across four stages:
1. **XGBoost Baseline:** Decision trees trained strictly on $Z$-score normalized structural metrics.
2. **GraphCodeBERT:** RoBERTa-based deep semantic tokenization of source code.
3. **Fusion Architecture:** Concatenation of the GraphCodeBERT `[CLS]` embedding with a parallel `MetricEncoder` (Dense $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Dense).
4. **Graph Transformer:** Graph-based learning. Nodes are connected via cosine similarity ($>0.80$) over their hybrid embeddings.

### Leakage Prevention
To prevent train-test data leakage, our system strictly enforces an inductive graph evaluation. Test and validation embeddings are transformed via `StandardScaler` fitted *only* on the training set, and message passing is confined solely to the isolated training graph. Inference on test nodes is inherently 0-hop.

## Experimental Setup
### Environment
- PyTorch 2.x, PyTorch Geometric, HuggingFace Transformers
- GPU: Evaluated on 2x Tesla T4 (16GB VRAM each)
- Training utilizes Mixed Precision (`torch.amp.autocast`) and Gradient Accumulation for memory-safe scaling.

### Installation
```bash
git clone https://github.com/BIGREASONS/BUGS.git
cd BUGS
pip install -r requirements.txt
```

### Reproducibility
A fixed random seed (`42`) is enforced across `numpy` and `torch`. All experiments automatically log hyperparameters to `outputs/publication_v1/config.json`.

## Usage
**1. Train Text & Fusion Baselines**
```bash
python train.py --model all
```

**2. Train Graph Transformer**
```bash
python train_graph.py
```

**3. Generate Publication Artifacts**
```bash
python comparison.py
```
This generates LaTeX tables, Markdown summaries, and matplotlib visualizations (`outputs/publication_v1/results/`).

## Future Work
Future extensions will explore dynamic thresholding for cosine similarity matrices and the integration of AST abstract syntax trees natively into the GraphCodeBERT cross-attention layers.
