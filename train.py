"""
train.py — BSPTEGT unified training pipeline.

Adaptation of BSPTEGT for source-code bug severity prediction
using GraphCodeBERT embeddings and Graph Transformers.

Pipeline:
    Stage 1: Fine-tune GraphCodeBERT on classification (learns embeddings)
    Stage 2: Extract CLS embeddings, build similarity graph
    Stage 3: Train BSPTEGTGraphTransformer, evaluate on test set

Usage:
    python train.py                         # full pipeline
    python train.py --skip-finetune         # use existing embedder checkpoint
    python train.py --skip-extraction       # use cached embeddings
    python train.py --skip-finetune --skip-extraction  # graph-only
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from configs.config import Config
from src.datasets.dataset import create_dataloaders
from src.models.embedding_extractor import GraphCodeBERTEmbedder, load_checkpoint
from src.models.graph_transformer_model import BSPTEGTGraphTransformer
from src.graphs.graph_builder import BSPTEGTGraphBuilder
from src.trainers.embedder_trainer import EmbedderTrainer
from src.trainers.graph_trainer import BSPTEGTTrainer
from src.evaluation.evaluate import Evaluator


# ---------------------------------------------------------------------- #
#  Helpers                                                                 #
# ---------------------------------------------------------------------- #
def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from the training split only."""
    labels = [int(item.get('label', -1)) for item in train_loader.dataset.data]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total  = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1.0))
    print(f"Class counts  : {counts.astype(int).tolist()}")
    print(f"Class weights : {np.round(weights, 4).tolist()}")
    return torch.tensor(weights, dtype=torch.float)


def prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Move to device and wrap in DataParallel if multi-GPU."""
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel).")
        model = nn.DataParallel(model)
    return model


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Extracts CLS embeddings, metrics, and labels from a DataLoader."""
    model.eval()
    all_emb, all_met, all_lbl = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    emb = model(input_ids, attention_mask, return_embeddings=True)
            else:
                emb = model(input_ids, attention_mask, return_embeddings=True)

            all_emb.append(emb.cpu())
            all_met.append(batch['metrics_tensor'])
            all_lbl.append(batch['label'])

    return torch.cat(all_emb), torch.cat(all_met), torch.cat(all_lbl)


def get_cached_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_dir: str,
    split_name: str,
):
    """Loads cached features if available, otherwise extracts and caches."""
    emb_file = os.path.join(cache_dir, f'{split_name}_emb.pt')
    met_file = os.path.join(cache_dir, f'{split_name}_met.pt')
    lbl_file = os.path.join(cache_dir, f'{split_name}_lbl.pt')

    if all(os.path.exists(f) for f in [emb_file, met_file, lbl_file]):
        print(f"  Loading {split_name} features from cache...")
        emb = torch.load(emb_file, map_location='cpu', weights_only=True)
        met = torch.load(met_file, map_location='cpu', weights_only=True)
        lbl = torch.load(lbl_file, map_location='cpu', weights_only=True)
        return emb, met, lbl

    print(f"  Extracting {split_name} features...")
    emb, met, lbl = extract_features(model, loader, device)
    torch.save(emb, emb_file)
    torch.save(met, met_file)
    torch.save(lbl, lbl_file)
    return emb, met, lbl


# ---------------------------------------------------------------------- #
#  Main pipeline                                                           #
# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="BSPTEGT Training Pipeline")
    parser.add_argument(
        '--skip-finetune', action='store_true',
        help='Skip Stage 1; use existing embedder checkpoint.',
    )
    parser.add_argument(
        '--skip-extraction', action='store_true',
        help='Skip Stage 2; use cached embeddings.',
    )
    args = parser.parse_args()

    Config.validate_config()
    Config.set_seed(Config.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {Config.EXPERIMENT_NAME}")

    embedder_ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, 'embedder')
    embedder_ckpt     = os.path.join(embedder_ckpt_dir, 'best_model.pt')
    graph_ckpt_dir    = os.path.join(Config.CHECKPOINT_DIR, 'graph')
    os.makedirs(Config.EMB_CACHE_DIR, exist_ok=True)

    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found: {Config.TRAIN_PATH}\n"
            f"Place your JSONL files in the data/ directory."
        )

    # ================================================================== #
    #  Stage 1: Fine-tune GraphCodeBERT                                    #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  STAGE 1: GraphCodeBERT Embedding Fine-Tuning")
    print("=" * 60)

    print("\n--- Building DataLoaders ---")
    train_loader, valid_loader, test_loader = create_dataloaders(
        Config.TRAIN_PATH, Config.VALID_PATH, Config.TEST_PATH,
        tokenizer_name=Config.MODEL_NAME,
        batch_size=Config.EMB_BATCH_SIZE,
        max_length=Config.MAX_LENGTH,
        num_workers=Config.NUM_WORKERS,
        seed=Config.SEED,
        metric_keys=Config.METRIC_KEYS,
        num_classes=Config.NUM_CLASSES,
    )

    class_weights = compute_class_weights(train_loader, Config.NUM_CLASSES)

    if args.skip_finetune:
        print("\n[SKIP] Stage 1 skipped (--skip-finetune).")
        if not os.path.exists(embedder_ckpt):
            raise FileNotFoundError(
                f"Embedder checkpoint not found: {embedder_ckpt}\n"
                f"Run without --skip-finetune first."
            )
    else:
        embedder = GraphCodeBERTEmbedder(
            Config.MODEL_NAME, Config.NUM_CLASSES, Config.EMB_DROPOUT,
        )
        embedder = prepare_model(embedder, device)

        trainer = EmbedderTrainer(
            model=embedder,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            epochs=Config.EMB_EPOCHS,
            learning_rate=Config.EMB_LEARNING_RATE,
            weight_decay=Config.EMB_WEIGHT_DECAY,
            accumulation_steps=Config.EMB_ACCUMULATION_STEPS,
            early_stopping_patience=Config.EMB_EARLY_STOPPING_PATIENCE,
            checkpoint_dir=embedder_ckpt_dir,
            class_weights=class_weights,
        )
        trainer.train()

    # ================================================================== #
    #  Stage 2: Extract Embeddings & Build Graphs                          #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  STAGE 2: Embedding Extraction & Graph Construction")
    print("=" * 60)

    if args.skip_extraction:
        print("\n[SKIP] Stage 2 extraction skipped (--skip-extraction).")
        # Verify cache exists
        for split in ['train', 'valid', 'test']:
            if not os.path.exists(os.path.join(Config.EMB_CACHE_DIR, f'{split}_emb.pt')):
                raise FileNotFoundError(
                    f"Cached embeddings for '{split}' not found in {Config.EMB_CACHE_DIR}\n"
                    f"Run without --skip-extraction first."
                )

        train_emb = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'train_emb.pt'), map_location='cpu', weights_only=True)
        train_met = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'train_met.pt'), map_location='cpu', weights_only=True)
        train_lbl = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'train_lbl.pt'), map_location='cpu', weights_only=True)
        valid_emb = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'valid_emb.pt'), map_location='cpu', weights_only=True)
        valid_met = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'valid_met.pt'), map_location='cpu', weights_only=True)
        valid_lbl = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'valid_lbl.pt'), map_location='cpu', weights_only=True)
        test_emb  = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'test_emb.pt'), map_location='cpu', weights_only=True)
        test_met  = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'test_met.pt'), map_location='cpu', weights_only=True)
        test_lbl  = torch.load(os.path.join(Config.EMB_CACHE_DIR, 'test_lbl.pt'), map_location='cpu', weights_only=True)
    else:
        # Load fine-tuned embedder for extraction
        print("\n--- Loading fine-tuned GraphCodeBERT ---")
        embedder = GraphCodeBERTEmbedder(
            Config.MODEL_NAME, Config.NUM_CLASSES, Config.EMB_DROPOUT,
        )
        load_checkpoint(embedder, embedder_ckpt, device)
        embedder = prepare_model(embedder, device)

        print("\n--- Extracting features ---")
        train_emb, train_met, train_lbl = get_cached_features(
            embedder, train_loader, device, Config.EMB_CACHE_DIR, 'train',
        )
        valid_emb, valid_met, valid_lbl = get_cached_features(
            embedder, valid_loader, device, Config.EMB_CACHE_DIR, 'valid',
        )
        test_emb, test_met, test_lbl = get_cached_features(
            embedder, test_loader, device, Config.EMB_CACHE_DIR, 'test',
        )

    print(f"\n--- Building graphs (type={Config.GRAPH_TYPE}, eval_mode={Config.GRAPH_EVAL_MODE}) ---")
    graph_builder = BSPTEGTGraphBuilder(Config.GRAPHS_DIR)

    train_graph = graph_builder.build_train_graph(
        train_emb, train_lbl, train_met, Config.GRAPH_SIMILARITY_THRESHOLD,
    )
    print("Train graph:", graph_builder.get_statistics(train_graph))
    graph_builder.save_graph_stats(
        train_graph, 'train_graph_stats.json', Config.GRAPH_SIMILARITY_THRESHOLD,
    )

    valid_graph = graph_builder.build_eval_graph(
        train_emb, train_met, train_lbl,
        valid_emb, valid_met, valid_lbl,
        Config.GRAPH_SIMILARITY_THRESHOLD,
    )

    test_graph = graph_builder.build_eval_graph(
        train_emb, train_met, train_lbl,
        test_emb, test_met, test_lbl,
        Config.GRAPH_SIMILARITY_THRESHOLD,
    )

    # ================================================================== #
    #  Stage 3: Train Graph Transformer                                    #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  STAGE 3: BSPTEGT Graph Transformer Training")
    print("=" * 60)

    gt_model = BSPTEGTGraphTransformer(
        cls_dim=768,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.GT_HIDDEN_DIM,
        heads=Config.GT_HEADS,
        dropout=Config.GT_DROPOUT,
        num_metrics=Config.NUM_METRICS,
    )

    num_train_nodes = (
        train_emb.shape[0]
        if Config.GRAPH_EVAL_MODE == 'inductive_attachment'
        else 0
    )

    gt_trainer = BSPTEGTTrainer(
        model=gt_model,
        train_graph=train_graph,
        device=device,
        epochs=Config.GT_EPOCHS,
        learning_rate=Config.GT_LEARNING_RATE,
        weight_decay=Config.GT_WEIGHT_DECAY,
        early_stopping_patience=Config.GT_EARLY_STOPPING_PATIENCE,
        checkpoint_dir=graph_ckpt_dir,
        class_weights=class_weights,
    )

    gt_trainer.train(valid_graph, num_train_nodes=num_train_nodes)

    # ================================================================== #
    #  Final Evaluation                                                    #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  EVALUATION: GraphCodeBERT Baseline vs. BSPTEGT")
    print("=" * 60)

    evaluator = Evaluator(Config.RESULTS_DIR, class_names=Config.CLASS_NAMES)

    # 1. Evaluate Embedder Baseline (GraphCodeBERT Classifier)
    print("\n--- 1. Evaluating GraphCodeBERT Baseline ---")
    if os.path.exists(embedder_ckpt):
        embedder_eval = GraphCodeBERTEmbedder(
            Config.MODEL_NAME, Config.NUM_CLASSES, Config.EMB_DROPOUT,
        )
        load_checkpoint(embedder_eval, embedder_ckpt, device)
        embedder_eval = prepare_model(embedder_eval, device)

        # Temporary trainer just to use its .evaluate() method
        dummy_trainer = EmbedderTrainer(
            model=embedder_eval,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            checkpoint_dir=embedder_ckpt_dir,
            class_weights=class_weights,
        )
        base_metrics, base_y_true, base_y_pred = dummy_trainer.evaluate(test_loader)
        base_metrics['Model'] = 'GraphCodeBERT_Baseline'

        for k, v in base_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        base_cm = evaluator.generate_confusion_matrix(base_y_true, base_y_pred)
        evaluator.save_results(
            base_metrics, base_cm, base_y_true, base_y_pred, prefix='GraphCodeBERT_Baseline',
        )
    else:
        print(f"  [SKIP] Baseline checkpoint not found: {embedder_ckpt}")

    # 2. Evaluate Graph Transformer
    print("\n--- 2. Evaluating BSPTEGT Graph Transformer ---")
    metrics, y_true, y_pred = gt_trainer.evaluate(
        test_graph, num_train_nodes=num_train_nodes,
    )
    metrics['Model'] = 'BSPTEGTGraphTransformer'

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    cm = evaluator.generate_confusion_matrix(y_true, y_pred)
    evaluator.save_results(
        metrics, cm, y_true, y_pred, prefix='BSPTEGTGraphTransformer',
    )

    print("\n" + "=" * 60)
    print("  BSPTEGT pipeline complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
