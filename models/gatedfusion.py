import torch
import torch.nn as nn
import numpy as np
import os

from models.gnn_models import GNN

class GatedFusion(nn.Module):
    def __init__(
        self,
        node_attr_dim,
        edge_attr_dim,
        gnn_hidden_dim,
        combined_dim = 128,
        mlp_out_dim = 128,
        mlp_hidden_dim = 128,
        num_gnn_layers = 3,
        dropout = 0.3,
        JK = "last",
        gnn_type = "gin",
        num_nodes = -1,
        state = "MA",
        vision_type = "clip_image"
    ):
        super().__init__()
        self.gnn = GNN(
            in_channels_node = node_attr_dim,
            in_channels_edge = edge_attr_dim,
            hidden_channels = gnn_hidden_dim,
            num_layers = num_gnn_layers,
            dropout = dropout,
            JK = JK,
            gnn_type = gnn_type,
            num_nodes = num_nodes,
        )
        self.state = state
        self.vision_features = self.load_vision_features(state=self.state, vision_type=vision_type)
        self.vision_dim = self.vision_features.size(1)
        self.gate_net = nn.Sequential(
            nn.Linear(gnn_hidden_dim + self.vision_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_out_dim)
        )
        self.transfer_dim = nn.Sequential(
            nn.Linear(self.vision_dim, gnn_hidden_dim)
        )
    def load_vision_features(self, state, vision_type):
        dir = f"./embeddings/{vision_type}/{state}.npy"
        feature = np.load(dir)
        return torch.tensor(feature, dtype=torch.float)
    
    def forward(self, node_attr, edge_index, edge_attr):
        gnn_emb = self.gnn(node_attr, edge_index, edge_attr)
        vision_emb = self.vision_features
        device = gnn_emb.device
        vision_emb = vision_emb.to(device)
        combined = torch.cat([gnn_emb, vision_emb], dim=1)
        gate = self.gate_net(combined)
        vision_emb = self.transfer_dim(vision_emb)
        fused = gate * gnn_emb + (1-gate) * vision_emb
        out = self.output_mlp(fused)
        return out