import torch
import torch.nn as nn
import numpy as np
from models.gnn_models import GNN

class GatedNetwork(nn.Module):
    def __init__(self, fusion_dim, out_dim):
        super().__init__()
        self.gated_network= nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return torch.softmax(self.gated_network(x), dim=-1)

class Expert(nn.Module):
    def __init__(self, fusion_dim, out_dim, layer_num=3):
        super().__init__()
        self.mlp = nn.Sequential()
        for layer in range(layer_num):
            if layer == 0: 
                self.mlp.append(nn.Linear(fusion_dim, out_dim))
            else: self.mlp.append(nn.Linear(out_dim, out_dim))
            if layer != layer_num-1:
                self.mlp.append(nn.ReLU())
    def forward(self, x):
        return self.mlp(x)
    
class MoE(nn.Module):
    def __init__(self, 
        node_attr_dim,
        edge_attr_dim,
        gnn_hidden_dim,
        combined_dim = 128,
        mlp_out_dim = 128,
        mlp_hidden_dim = 128,
        num_gnn_layers = 3,
        dropout = 0.3,
        num_experts=4,
        JK = "last",
        gnn_type = "gin",
        num_nodes = -1,
        state = "MA",
        vision_type = "clip_image",
        layer_num=3):
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
        self.vision_features=self.load_vision_features(state=self.state, vision_type=vision_type)
        self.vision_dim = self.vision_features.size(1)
        self.experts = nn.ModuleList([
            Expert(gnn_hidden_dim+self.vision_dim, gnn_hidden_dim) for i in range(num_experts)
        ])
        self.gate = GatedNetwork(gnn_hidden_dim+self.vision_dim, num_experts)

    def load_vision_features(self, state, vision_type):
        dir = f"./embeddings/{vision_type}/{state}.npy"
        feature = np.load(dir)
        return torch.tensor(feature, dtype=torch.float)
    
    def forward(self, node_attr, edge_index, edge_attr):
        gnn_emb = self.gnn(node_attr, edge_index, edge_attr)
        device = gnn_emb.device
        vision_emb = self.vision_features.to(device)
        fusion_input = torch.cat([gnn_emb, vision_emb], dim=1)
        gated_weights = self.gate(fusion_input)
        export_output = torch.stack([expert(fusion_input) for expert in self.experts],dim=1)
        moe_output = (gated_weights.unsqueeze(-1) * export_output).sum(dim=1)
        return moe_output