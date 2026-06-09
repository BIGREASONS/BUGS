import torch
import os
import random
import numpy as np

class Config:
    # Paths
    TRAIN_PATH = 'data/train.jsonl'
    VALID_PATH = 'data/valid.jsonl'
    TEST_PATH = 'data/test.jsonl'
    OUTPUT_DIR = 'outputs'
    CHECKPOINT_DIR = 'outputs/checkpoints'
    RESULTS_DIR = 'outputs/results'
    GRAPHS_DIR = 'outputs/graphs'
    
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
    
    # Graph Transformer Hyperparameters
    GT_HIDDEN_DIM = 256
    GT_HEADS = 8
    GT_LAYERS = 2
    
    # Optimization
    SEED = 42
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 3
    ACCUMULATION_STEPS = 1 # Update to > 1 if running out of memory
    
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
