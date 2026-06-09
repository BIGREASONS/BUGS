import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from configs.config import Config
from src.datasets.dataset import create_dataloaders
from src.baselines.xgboost_baseline import run_xgboost_baseline
from src.models.graphcodebert_model import GraphCodeBERTClassifier
from src.models.fusion_model import FusionModel
from src.trainers.trainer import Trainer
from src.evaluation.evaluate import Evaluator

def prepare_model(model, device):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
        model = nn.DataParallel(model)
    return model

def main():
    parser = argparse.ArgumentParser(description="Bug Severity Prediction Training")
    parser.add_argument('--model', type=str, choices=['xgboost', 'graphcodebert', 'fusion', 'all'], default='all', help="Model to train")
    args = parser.parse_args()

    Config.set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    evaluator = Evaluator(Config.RESULTS_DIR)

    if args.model in ['xgboost', 'all']:
        print("\n--- Running XGBoost Baseline ---")
        if os.path.exists(Config.TRAIN_PATH):
            xgb_metrics, xgb_importance = run_xgboost_baseline(
                Config.TRAIN_PATH, Config.VALID_PATH, Config.TEST_PATH
            )
            xgb_metrics['Model'] = 'XGBoost'
            evaluator.save_results(xgb_metrics, np.zeros((4,4)), [], [], prefix='XGBoost')
        else:
            print("Data files not found. Please place JSONL files in data/ directory.")

    if args.model in ['graphcodebert', 'fusion', 'all']:
        if not os.path.exists(Config.TRAIN_PATH):
            print("Data files not found. Please place JSONL files in data/ directory.")
            return
            
        print("\n--- Setting up PyTorch DataLoaders ---")
        train_loader, valid_loader, test_loader = create_dataloaders(
            Config.TRAIN_PATH, Config.VALID_PATH, Config.TEST_PATH,
            tokenizer_name=Config.MODEL_NAME,
            batch_size=Config.BATCH_SIZE,
            max_length=Config.MAX_LENGTH,
            num_workers=Config.NUM_WORKERS
        )

        if args.model in ['graphcodebert', 'all']:
            print("\n--- Training GraphCodeBERT Baseline ---")
            gcb_model = GraphCodeBERTClassifier(Config.MODEL_NAME, Config.NUM_CLASSES, Config.DROPOUT)
            gcb_model = prepare_model(gcb_model, device)
            
            trainer_gcb = Trainer(
                model=gcb_model, train_loader=train_loader, valid_loader=valid_loader,
                device=device, epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY, accumulation_steps=Config.ACCUMULATION_STEPS,
                early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
                checkpoint_dir=os.path.join(Config.CHECKPOINT_DIR, 'gcb')
            )
            gcb_best_metrics = trainer_gcb.train()
            print("\nEvaluating GraphCodeBERT on Test Set...")
            gcb_test_metrics = trainer_gcb.evaluate(test_loader)
            gcb_test_metrics['Model'] = 'GraphCodeBERT'
            evaluator.save_results(gcb_test_metrics, np.zeros((4,4)), [], [], prefix='GraphCodeBERT')

        if args.model in ['fusion', 'all']:
            print("\n--- Training Fusion Model (GraphCodeBERT + Metrics) ---")
            fusion_model = FusionModel(Config.MODEL_NAME, Config.NUM_CLASSES, Config.DROPOUT)
            fusion_model = prepare_model(fusion_model, device)
            
            trainer_fusion = Trainer(
                model=fusion_model, train_loader=train_loader, valid_loader=valid_loader,
                device=device, epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY, accumulation_steps=Config.ACCUMULATION_STEPS,
                early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
                checkpoint_dir=os.path.join(Config.CHECKPOINT_DIR, 'fusion')
            )
            fusion_best_metrics = trainer_fusion.train()
            print("\nEvaluating Fusion Model on Test Set...")
            fusion_test_metrics = trainer_fusion.evaluate(test_loader)
            fusion_test_metrics['Model'] = 'FusionModel'
            evaluator.save_results(fusion_test_metrics, np.zeros((4,4)), [], [], prefix='FusionModel')

if __name__ == "__main__":
    main()

