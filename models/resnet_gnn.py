import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from models.gnn_models import GNN, LinkPredictor
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd

class NodeVisionEncoder(nn.Module):
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))  # 去掉fc
        self.fc = nn.Linear(backbone.fc.in_features, out_dim)

    def forward(self, imgs):  # imgs: [N, 3, H, W]
        feats = self.backbone(imgs)          # [N, 512, 1, 1]
        feats = feats.flatten(1)             # [N, 512]
        out = self.fc(feats)                 # [N, out_dim]
        return out

class VisionGNN(nn.Module):
    def __init__(
        self,
        node_attr_dim,
        edge_attr_dim,
        hidden_channels,
        vision_out_dim=64,
        num_gnn_layers=3,
        dropout=0.3,
        JK="last",
        gnn_type="gcn",
        num_nodes=-1,
        vision_pretrained=True,
        state="MA"
    ):
        super().__init__()
        self.vision_encoder = NodeVisionEncoder(out_dim=vision_out_dim, pretrained=vision_pretrained)
        self.gnn = GNN(
            in_channels_node=node_attr_dim + vision_out_dim,
            in_channels_edge=edge_attr_dim,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            dropout=dropout,
            JK=JK,
            gnn_type=gnn_type,
            num_nodes=num_nodes,
        )
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.state = state

    def load_image_by_node_id(self, node_id, img_dir='/home/michael/project/data/Nodes_', device='cpu'):
        img_dir = img_dir + self.state
        img_path = os.path.join(img_dir, f"{node_id}.png")
        former = torch.zeros(1, 3, 224, 224, device=device)
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
            former = img
            return img
        except:
            return former

    def forward(self, node_attr, edge_index, edge_attr):
        """
        Input:
        - node_attr: [N, node_attr_dim]
        - edge_index: [2, E]
        - edge_attr: [E, edge_attr_dim] or None

        Output: 
        - node embedding [N, hidden_channels]
        """
        csv_path = f'./data/Road_Networks/{self.state}/Road_Network_Nodes_{self.state}.csv'
        df = pd.read_csv(csv_path)
        node_ids = df['node_id'].tolist() 
        device = node_attr.device
        vision_features = []
        for nid in tqdm(node_ids):
            img = self.load_image_by_node_id(nid, device=device)
            feat = self.vision_encoder(img)  # [1, vision_out_dim]
            vision_features.append(feat.squeeze(0))
            del img
        img_emb = torch.stack(vision_features, dim=0)  # [N, vision_out_dim]

        node_input = torch.cat([node_attr, img_emb], dim=1)  # [N, node_attr_dim + vision_out_dim]
        node_emb = self.gnn(node_input, edge_index, edge_attr)  # [N, hidden_channels]
        return node_emb