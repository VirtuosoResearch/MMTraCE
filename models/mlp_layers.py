import torch
import torch.nn as nn
import numpy

class MLP(nn.Module):
    def __init__(self, node_attr_dim, edge_attr_dim, mlp_out_dim=128, mlp_hidden_dim=128, dropout=0.3):
        super().__init__()
        self.output_mlp = nn.Sequential(
            nn.Linear(node_attr_dim + edge_attr_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, node_attr, edge_index, edge_attr):
        print("node_attr.size(): ",node_attr.size())
        print("edge_attr.size(): ",edge_attr.size())
        x = torch.concat([node_attr, edge_attr], dim=0)
        return self.output_mlp(x)
