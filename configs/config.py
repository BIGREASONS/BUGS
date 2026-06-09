import torch
import os
import random
import numpy as np

class Config:
    # Experiment Config
    EXPERIMENT_NAME = 'publication_v1'
    
    # Paths
    TRAIN_PATH = 'data/train.jsonl'
    VALID_PATH = 'data/valid.jsonl'
    TEST_PATH = 'data/test.jsonl'
    OUTPUT_DIR = 'outputs'
    CHECKPOINT_DIR = f'outputs/{EXPERIMENT_NAME}/checkpoints'
    RESULTS_DIR = f'outputs/{EXPERIMENT_NAME}/results'
    GRAPHS_DIR = f'outputs/{EXPERIMENT_NAME}/graphs'
    
    # Model
    MODEL_NAME = 'microsoft/graphcodebert-base'
    NUM_CLASSES = 4
    
    # Hyperparameters
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.2
    
    # Loss Configuration
    # Options: 'ce', 'weighted_ce', 'focal'
    LOSS_TYPE = 'focal'
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 'class_weights'
    
    # Graph Transformer Hyperparameters
    GT_HIDDEN_DIM = 256
    GT_HEADS = 8
    GT_LAYERS = 2
    
    # Graph Construction
    GRAPH_SIMILARITY_THRESHOLD = 0.80
    
    # Optimization
    SEED = 42
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 3
    ACCUMULATION_STEPS = 1 # Update to > 1 if running out of memory
    
    @classmethod
    def validate_config(cls):
        valid_losses = ['ce', 'weighted_ce', 'focal']
        assert cls.LOSS_TYPE in valid_losses, f"LOSS_TYPE must be one of {valid_losses}"
        if cls.LOSS_TYPE == 'focal':
            assert isinstance(cls.FOCAL_GAMMA, float), "FOCAL_GAMMA must be a float"
        assert cls.MAX_LENGTH <= 512, "GraphCodeBERT max length is 512"
        print("Config validation passed.")
        
    @staticmethod
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
