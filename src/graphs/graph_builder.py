import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Dict, Optional

from configs.config import Config
from src.models.modules import MetricEncoder


class BSPTEGTGraphBuilder:
    """Constructs PyTorch Geometric graphs for BSPTEGT.

    Supported graph types:
        - 'similarity'             : cosine similarity on CLS embeddings
        - 'similarity_plus_metrics': cosine on CLS + MetricEncoder output
        - 'knn'                    : k-nearest neighbours on the similarity space
        - 'none'                   : isolated nodes (MLP ablation baseline)

    Supported eval modes:
        - 'strict_inductive'       : eval nodes get 0 edges (0-hop MLP)
        - 'inductive_attachment'   : eval nodes are attached to the training graph
    """

    def __init__(self, output_dir: str = 'outputs/graphs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Edge computation                                                    #
    # ------------------------------------------------------------------ #
    def _batched_cosine_similarity_edges(
        self,
        embeddings: torch.Tensor,
        threshold: float,
        batch_size: int = 2000,
    ) -> torch.Tensor:
        """Pairwise cosine similarity in batches. Returns edges >= threshold."""
        device     = embeddings.device
        embeddings = F.normalize(embeddings, p=2, dim=1)
        N          = embeddings.shape[0]
        sources, targets = [], []

        for i in range(0, N, batch_size):
            end   = min(i + batch_size, N)
            batch = embeddings[i:end]
            sim   = torch.mm(batch, embeddings.t())

            for j in range(end - i):
                sim[j, i + j] = 0.0  # remove self-loops

            mask = sim >= threshold
            idx  = mask.nonzero(as_tuple=False)
            if len(idx) > 0:
                sources.append(idx[:, 0] + i)
                targets.append(idx[:, 1])

        if not sources:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)

    def _knn_edges(self, embeddings: torch.Tensor, k: int) -> torch.Tensor:
        """Undirected k-NN graph via exact Euclidean distance on L2-normalised vectors."""
        device = embeddings.device
        N = embeddings.shape[0]
        if N <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist = torch.cdist(embeddings, embeddings)
        dist.fill_diagonal_(float('inf'))

        actual_k = min(k, N - 1)
        _, topk_idx = torch.topk(dist, actual_k, dim=1, largest=False)

        sources = torch.arange(N, device=device).view(-1, 1).expand(-1, actual_k).reshape(-1)
        targets = topk_idx.reshape(-1)

        directed  = torch.stack([sources, targets], dim=0)
        reverse   = torch.stack([targets, sources], dim=0)
        edges     = torch.unique(torch.cat([directed, reverse], dim=1), dim=1)
        return edges

    # ------------------------------------------------------------------ #
    #  Similarity space                                                    #
    # ------------------------------------------------------------------ #
    def _get_similarity_space(
        self,
        embeddings: torch.Tensor,
        metrics: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns the feature space used for edge computation."""
        if Config.GRAPH_TYPE == 'similarity':
            return embeddings.to(device)

        elif Config.GRAPH_TYPE == 'similarity_plus_metrics':
            encoder = MetricEncoder(
                in_features=Config.NUM_METRICS, dropout=0.0
            ).to(device)
            encoder.eval()
            with torch.no_grad():
                metric_emb = encoder(metrics.to(device))
            return torch.cat([embeddings.to(device), metric_emb], dim=1)

        return embeddings.to(device)

    def _compute_edges(
        self, sim_space: torch.Tensor, threshold: float,
    ) -> torch.Tensor:
        if Config.GRAPH_TYPE == 'none':
            return torch.empty((2, 0), dtype=torch.long, device=sim_space.device)
        elif Config.GRAPH_TYPE == 'knn':
            return self._knn_edges(sim_space, Config.GRAPH_KNN_K)
        else:
            return self._batched_cosine_similarity_edges(sim_space, threshold)

    def _build_features(
        self,
        embeddings: torch.Tensor,
        metrics: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Node features: [N, 768 + 10] = [N, 778]."""
        return torch.cat([embeddings, metrics], dim=1).to(device)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def build_train_graph(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metrics: torch.Tensor,
        threshold: float,
    ) -> Data:
        """Builds the training graph (edges among training nodes only)."""
        device    = embeddings.device
        x         = self._build_features(embeddings, metrics, device)
        sim_space = self._get_similarity_space(embeddings, metrics, device)
        edge_index = self._compute_edges(sim_space, threshold)
        return Data(x=x, edge_index=edge_index, y=labels.to(device))

    def build_eval_graph(
        self,
        train_emb: torch.Tensor,
        train_met: torch.Tensor,
        train_lbl: torch.Tensor,
        eval_emb: torch.Tensor,
        eval_met: torch.Tensor,
        eval_lbl: torch.Tensor,
        threshold: float,
    ) -> Data:
        """Builds the evaluation graph (validation or test)."""
        device = train_emb.device

        if Config.GRAPH_EVAL_MODE == 'strict_inductive':
            print("BSPTEGTGraphBuilder: strict_inductive — 0 edges for eval nodes.")
            x = self._build_features(eval_emb, eval_met, device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            return Data(x=x, edge_index=edge_index, y=eval_lbl.to(device))

        elif Config.GRAPH_EVAL_MODE == 'inductive_attachment':
            print("BSPTEGTGraphBuilder: inductive_attachment — attaching eval to train graph.")
            comb_emb = torch.cat([train_emb, eval_emb], dim=0)
            comb_met = torch.cat([train_met, eval_met], dim=0)
            comb_lbl = torch.cat([train_lbl, eval_lbl], dim=0)

            x         = self._build_features(comb_emb, comb_met, device)
            sim_space = self._get_similarity_space(comb_emb, comb_met, device)
            edge_index = self._compute_edges(sim_space, threshold)
            return Data(x=x, edge_index=edge_index, y=comb_lbl.to(device))

        else:
            raise ValueError(f"Unknown GRAPH_EVAL_MODE: {Config.GRAPH_EVAL_MODE}")

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #
    def get_statistics(self, data: Data) -> Dict[str, float]:
        N, E = data.num_nodes, data.num_edges
        return {
            'num_nodes':  N,
            'num_edges':  E,
            'density':    E / (N * (N - 1)) if N > 1 else 0.0,
            'avg_degree': E / N             if N > 0 else 0.0,
        }

    def save_graph_stats(
        self, data: Data, filename: str, threshold: float,
    ) -> None:
        stats = self.get_statistics(data)
        stats['threshold']  = threshold
        stats['graph_type'] = Config.GRAPH_TYPE
        stats['eval_mode']  = Config.GRAPH_EVAL_MODE
        if Config.GRAPH_TYPE == 'knn':
            stats['knn_k'] = Config.GRAPH_KNN_K
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Graph stats saved -> {path}")

    def save_graph(self, data: Data, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        torch.save(data, path)
        print(f"Graph saved -> {path}")
