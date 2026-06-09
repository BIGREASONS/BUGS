import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity

class GraphBuilder:
    def __init__(self, output_dir: str = 'outputs/graphs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def build_graph(self, embeddings: torch.Tensor, labels: torch.Tensor, metrics: torch.Tensor, thresholds: List[float] = [0.75, 0.80, 0.85]) -> Dict[float, Data]:
        """
        Builds graphs based on cosine similarity thresholds.
        Only uses provided embeddings to avoid train-test leakage.
        """
        emb_np = embeddings.cpu().numpy()
        print("Calculating cosine similarity matrix...")
        sim_matrix = cosine_similarity(emb_np)
        
        # We fill diagonal with 0 so nodes don't self-loop automatically unless explicitly added later
        np.fill_diagonal(sim_matrix, 0)
        
        graphs = {}
        for threshold in thresholds:
            print(f"Building graph for threshold: {threshold}")
            
            # Find edges where similarity >= threshold
            sources, targets = np.where(sim_matrix >= threshold)
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            
            # Node features: Concatenate embeddings and metrics
            x = torch.cat([embeddings, metrics], dim=1)
            
            # PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=labels)
            graphs[threshold] = data
            
        return graphs

    def get_statistics(self, pyG_data: Data) -> Dict[str, float]:
        """
        Returns graph statistics.
        """
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
        
    def edge_analysis(self, pyG_data: Data) -> str:
        """
        Provides deeper analysis of the edges using NetworkX.
        """
        # Convert to NetworkX for analysis
        G = nx.Graph()
        G.add_nodes_from(range(pyG_data.num_nodes))
        
        edges = pyG_data.edge_index.t().tolist()
        G.add_edges_from(edges)
        
        # Calculate connected components
        num_components = nx.number_connected_components(G)
        largest_cc = len(max(nx.connected_components(G), key=len)) if num_components > 0 else 0
        
        analysis = (
            f"Edge Analysis:\n"
            f"- Number of connected components: {num_components}\n"
            f"- Size of largest connected component: {largest_cc}\n"
        )
        return analysis

    def save_graph(self, pyG_data: Data, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save(pyG_data, path)
        print(f"Graph saved to {path}")
        
    def load_graph(self, filename: str) -> Data:
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found at {path}")
        return torch.load(path)
