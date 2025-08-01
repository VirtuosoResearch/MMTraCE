import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import torch_scatter

class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        H, D = self.num_heads, self.head_dim
        E = edge_index.size(1)

        q = self.q_proj(x).view(N, H, D)
        k = self.k_proj(x).view(N, H, D)
        v = self.v_proj(x).view(N, H, D)

        attn_logits = torch.full((H, N, N), float("-inf"), device=x.device)

        edge_bias = self.edge_proj(edge_attr)
        src, dst = edge_index

        dot = (q[src] * k[dst]).sum(dim=-1) / (D ** 0.5)

        total_bias = dot + edge_bias

        for h in range(H):
            attn_logits[h, src, dst] = total_bias[:, h]

        attn_weights = F.softmax(attn_logits, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", attn_weights, v).reshape(N, -1)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x
    
class GraphormerLayerSparse(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(N, H, D)
        k = self.k_proj(x).view(N, H, D)
        v = self.v_proj(x).view(N, H, D)

        src, dst = edge_index
        q_i = q[src]
        k_j = k[dst]
        v_j = v[dst]

        score = (q_i * k_j).sum(dim=-1) / (D ** 0.5)
        score += self.edge_proj(edge_attr)
        attn = F.softmax(score, dim=0)

        out = attn.unsqueeze(-1) * v_j
        out = torch_scatter.scatter_add(out, src, dim=0, dim_size=N)

        out = self.out_proj(out.reshape(N, -1))
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x
    
class Graphormer(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, num_layers, dropout=0.1, JK="last"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            GraphormerLayerSparse(hidden_dim, edge_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        h_list = []

        for l in range(self.num_layers):
            h = self.layers[l](h, edge_index, edge_attr)
            h = self.norms[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.JK == "max":
            return torch.stack(h_list, dim=0).max(dim=0)[0]
        elif self.JK == "concat":
            return torch.cat(h_list, dim=-1)