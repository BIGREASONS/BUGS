import os
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List, Tuple, Dict
from configs.config import Config

class GraphBuilder:
    def __init__(self, output_dir: str = 'outputs/graphs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _batched_cosine_similarity_edges(self, embeddings: torch.Tensor, threshold: float, batch_size: int = 2000) -> torch.Tensor:
        """
        Computes pairwise cosine similarity in batches to prevent OOM on large datasets.
        Returns edge_index of shape [2, num_edges] for similarities >= threshold.
        """
        device = embeddings.device
        embeddings = F.normalize(embeddings, p=2, dim=1)
        num_nodes = embeddings.shape[0]
        sources = []
        targets = []
        
        for i in range(0, num_nodes, batch_size):
            end = min(i + batch_size, num_nodes)
            batch = embeddings[i:end]
            
            # [batch_size, D] x [D, num_nodes] -> [batch_size, num_nodes]
            sim = torch.mm(batch, embeddings.t())
            
            # Remove self loops
            for j in range(end - i):
                sim[j, i + j] = 0.0
                
            mask = sim >= threshold
            idx = mask.nonzero(as_tuple=False)
            
            if len(idx) > 0:
                sources.append(idx[:, 0] + i)
                targets.append(idx[:, 1])
                
        if not sources:
            return torch.empty((2, 0), dtype=torch.long, device=device)
            
        sources = torch.cat(sources)
        targets = torch.cat(targets)
        return torch.stack([sources, targets], dim=0)
        
    def build_graph(self, embeddings: torch.Tensor, labels: torch.Tensor, metrics: torch.Tensor, thresholds: List[float] = [0.80], is_train: bool = True) -> Dict[float, Data]:
        """
        Builds graphs based on cosine similarity thresholds.
        Only uses provided embeddings to avoid train-test leakage.
        """
        if not is_train:
            # STRICT INDUCTIVE: 0-hop test inference. No edges drawn between test nodes or train nodes.
            print("Strict Inductive Mode: Test/Valid graph constructed with 0 edges.")
            graphs = {}
            for threshold in thresholds:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=embeddings.device)
                x = torch.cat([embeddings, metrics], dim=1)
                data = Data(x=x, edge_index=edge_index, y=labels)
                graphs[threshold] = data
            return graphs

        print("Calculating batched cosine similarity matrix...")
        graphs = {}
        for threshold in thresholds:
            print(f"Building graph for threshold: {threshold}")
            
            edge_index = self._batched_cosine_similarity_edges(embeddings, threshold)
            
            # Node features: Concatenate embeddings and metrics
            x = torch.cat([embeddings, metrics], dim=1)
            
            # PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=labels)
            graphs[threshold] = data
            
        return graphs

    def get_statistics(self, pyG_data: Data) -> Dict[str, float]:
        num_nodes = pyG_data.num_nodes
        num_edges = pyG_data.num_edges
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "avg_degree": avg_degree
        }
        
    def save_graph(self, pyG_data: Data, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save(pyG_data, path)
        print(f"Graph saved to {path}")
