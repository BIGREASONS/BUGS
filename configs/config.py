import torch
import os
import random
import numpy as np


class Config:
    # ------------------------------------------------------------------ #
    #  Experiment                                                          #
    # ------------------------------------------------------------------ #
    EXPERIMENT_NAME = 'bsptegt_v1'

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    TRAIN_PATH = 'data/train.jsonl'
    VALID_PATH = 'data/valid.jsonl'
    TEST_PATH  = 'data/test.jsonl'

    OUTPUT_DIR     = 'outputs'
    CHECKPOINT_DIR = f'outputs/{EXPERIMENT_NAME}/checkpoints'
    RESULTS_DIR    = f'outputs/{EXPERIMENT_NAME}/results'
    GRAPHS_DIR     = f'outputs/{EXPERIMENT_NAME}/graphs'
    EMB_CACHE_DIR  = f'outputs/{EXPERIMENT_NAME}/embeddings'

    # ------------------------------------------------------------------ #
    #  GraphCodeBERT Embedder                                              #
    # ------------------------------------------------------------------ #
    MODEL_NAME  = 'microsoft/graphcodebert-base'
    NUM_CLASSES = 4
    MAX_LENGTH  = 512

    # Complexity metric keys in each JSONL record (fixed order).
    METRIC_KEYS = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
    NUM_METRICS = len(METRIC_KEYS)  # 10

    # Confusion-matrix / report class labels (index -> name)
    CLASS_NAMES = ['Critical', 'Major', 'Medium', 'Minor']

    # ------------------------------------------------------------------ #
    #  Stage 1: Embedder Fine-Tuning                                       #
    # ------------------------------------------------------------------ #
    EMB_BATCH_SIZE   = 16
    EMB_EPOCHS       = 10
    EMB_LEARNING_RATE = 2e-5
    EMB_WEIGHT_DECAY = 0.01
    EMB_DROPOUT      = 0.2
    EMB_ACCUMULATION_STEPS = 1
    EMB_EARLY_STOPPING_PATIENCE = 3

    # ------------------------------------------------------------------ #
    #  Stage 3: Graph Transformer                                          #
    # ------------------------------------------------------------------ #
    GT_HIDDEN_DIM = 256
    GT_HEADS      = 8
    GT_LAYERS     = 2
    GT_EPOCHS     = 20
    GT_LEARNING_RATE = 2e-5
    GT_WEIGHT_DECAY  = 0.01
    GT_DROPOUT       = 0.2
    GT_EARLY_STOPPING_PATIENCE = 5

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #
    LOSS_TYPE   = 'focal'
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 'class_weights'

    # ------------------------------------------------------------------ #
    #  Graph Construction                                                  #
    # ------------------------------------------------------------------ #
    GRAPH_SIMILARITY_THRESHOLD = 0.80

    # 'similarity' | 'similarity_plus_metrics' | 'knn' | 'none'
    GRAPH_TYPE = 'similarity'

    # 'strict_inductive' | 'inductive_attachment'
    GRAPH_EVAL_MODE = 'inductive_attachment'

    GRAPH_KNN_K = 5

    # ------------------------------------------------------------------ #
    #  Reproducibility                                                     #
    # ------------------------------------------------------------------ #
    SEED        = 42
    NUM_WORKERS = 4

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def validate_config(cls) -> None:
        """Raises ValueError (not assert) so -O flag cannot bypass."""
        valid_losses = ['ce', 'weighted_ce', 'focal']
        if cls.LOSS_TYPE not in valid_losses:
            raise ValueError(f"LOSS_TYPE must be one of {valid_losses}")
        if cls.LOSS_TYPE == 'focal' and not isinstance(cls.FOCAL_GAMMA, float):
            raise ValueError("FOCAL_GAMMA must be a float")
        if cls.MAX_LENGTH > 512:
            raise ValueError(f"MAX_LENGTH {cls.MAX_LENGTH} exceeds GraphCodeBERT limit 512")
        if cls.GT_HIDDEN_DIM % cls.GT_HEADS != 0:
            raise ValueError(
                f"GT_HIDDEN_DIM ({cls.GT_HIDDEN_DIM}) must be divisible by "
                f"GT_HEADS ({cls.GT_HEADS})"
            )
        if len(cls.METRIC_KEYS) != cls.NUM_METRICS:
            raise ValueError("METRIC_KEYS / NUM_METRICS mismatch")
        valid_graph_types = ['similarity', 'similarity_plus_metrics', 'knn', 'none']
        if cls.GRAPH_TYPE not in valid_graph_types:
            raise ValueError(f"GRAPH_TYPE must be one of {valid_graph_types}")
        valid_eval_modes = ['strict_inductive', 'inductive_attachment']
        if cls.GRAPH_EVAL_MODE not in valid_eval_modes:
            raise ValueError(f"GRAPH_EVAL_MODE must be one of {valid_eval_modes}")
        print("Config validation passed.")

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
