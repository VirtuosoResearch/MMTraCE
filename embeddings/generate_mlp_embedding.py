import torch
import torch.nn as nn
import numpy as np
import os
import argparse

# num_nodes = 1000 
embedding_dim = 256 
mlp_output_dim = 128 

class MLPEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim, mlp_output_dim):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, mlp_output_dim),
        )

    def forward(self, node_ids):
        x = self.node_embedding(node_ids)  # (N, embedding_dim)
        out = self.mlp(x)  # (N, mlp_output_dim)
        return out
parser = argparse.ArgumentParser()
parser.add_argument("--state", type=str, default="DE")
args = parser.parse_args()

data_dir = "../data"
state_name = args.state

adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
edge_index = adj.coalesce().indices()

node = adj.coalesce().size()

num_nodes = node[0]

model = MLPEmbedding(num_nodes, embedding_dim, mlp_output_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

node_ids = torch.arange(num_nodes).to(device)

with torch.no_grad():
    node_embeddings = model(node_ids).cpu().numpy()

os.makedirs("./mlp_embedding", exist_ok=True)
np.save(f"./mlp_embedding/{state_name}.npy", node_embeddings)

print(f"Saved node embeddings with shape: {node_embeddings.shape}")
